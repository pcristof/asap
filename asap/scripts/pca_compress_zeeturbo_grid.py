def main():
    from asap.compress_grid import Compressor
    import os
    import argparse ## To read optional arguments
    from numpy.linalg import svd
    import numpy as np
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str, help="Input grid directory")
    parser.add_argument("-o", "--output", type=str, default='./pca_grid/')
    parser.add_argument("-t", "--tol", type=float, default=0.001)
    parser.add_argument("-d", "--debug", action='store_true')
    # parser.add_argument("-m", "--mpi", type=bool, default=False)
    # parser.add_argument("-p", "--profile", type=bool, default=False)
    # parser.add_argument("-d", "--dynesty", type=bool, default=False)
    parser.add_argument("-w", "--overwrite", action='store_true')

    args = parser.parse_args()
    pathtogrid = args.indir
    outputgrid = args.output

    pathtogrid = '/Users/pcristofari/Data/zeeturbo-grids/spectra-zeeturbo-v2/hdf5-spirou-vmic1-allteff/'
    ## Copy the wavelength solution
    import shutil
    shutil.copy(pathtogrid+'wave.hdf5', 'wave.hdf5')

    CP = Compressor()
    _, _, _, _, bvals = CP.interpret_grid_dimensions(pathtogrid)

    if np.mean(bvals)>100: ## We likely have magnetic fields in Gauss
        bvals/=1000

    nb_comp=5 ## Initial min number of components
    FREEZE_NB_COMP = False
    for BVAL in bvals[::-1]: ## in decreasing order (see why later)
        print(f'Loading grid for BVAL={BVAL}')
        CP.bs = BVAL
        nwvls, grid_n, teffs, loggs, mhs, alphas = CP.load_grid(pathtogrid, BVAL)
        nwvls = np.array(nwvls)
        print('Done loading grid')
        ## Get number of wavelengths
        n_lambda = grid_n.shape[-1]
        # Flatten all models
        X = grid_n.reshape(-1, n_lambda)   # (N_models, N_lambda)
        K_full = X.shape[0]
        # Get mean model
        mean_spectrum = X.mean(axis=0)
        X0 = X - mean_spectrum

        # X0 = U S V^T
        U, S, Vt = svd(X0, full_matrices=False)

        basis  = Vt                  # shape (Kmax, N_lambda)
        coeffs = U * S               # shape (N_models, Kmax)

        # Eigenvalues of covariance
        eigenvalues = (S**2) / (X0.shape[0] - 1)
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        idx = np.where(cumvar>1-args.tol)[0]
        if len(idx)==0: raise Exception('Failed guessing nb components')
        else: nb_comp = idx[0]

        if args.debug:
            plt.figure()
            plt.plot(cumvar)
            plt.axvline(nb_comp)
            plt.xlabel("Number of PCA components K")
            plt.ylabel("Cumulative explained variance")
            plt.grid()
            plt.show()

        ## Try to reconstruct spectra based main PCA coeffs
        ## Continue so long as you do not have an acceptable number of 
        ## dimensions.
        ## I note that I generally need a higher number of components for the
        ## magnetic spectra than non-magnetic spectra; hence the fact that we
        ## loop in decreasing magnetic field strengths.

        # from IPython import embed; embed()

        ## Start from 200
        nb_comp = len(basis)
        increase_precision = True
        while increase_precision:
            # Original spectrum
            maxerr = 0
            rms = 0
            for model_idx in range(len(basis)):
                spec_orig = X[model_idx]
                spec_pca = coeffs[model_idx, :nb_comp] @ basis[:nb_comp] \
                           + mean_spectrum
                residual = spec_pca - spec_orig
                _rms = np.sqrt(np.mean(residual**2))
                _maxerr = np.max(np.abs(residual))
                if _maxerr>maxerr:
                    maxerr=_maxerr
                    model_idx_maxerr = model_idx
                if _rms>rms:
                    rms=_rms
                    model_idx_maxrms = model_idx
            if maxerr<=args.tol:
                increase_precision=False
            else:
                if FREEZE_NB_COMP:
                    raise Exception("CAUTION PRECISION INSUFICIENT")
                else:
                    nb_comp+=int(round(0.1*nb_comp)) ## Increase by 10%

        ## Freeze the number of components, because all mini grids must have
        ## the same.
        FREEZE_NB_COMP = True

        model_idx = model_idx_maxerr
        spec_orig = X[model_idx]
        spec_pca = coeffs[model_idx, :nb_comp] @ basis[:nb_comp] + mean_spectrum

        coeffs_grid = coeffs.reshape(len(teffs), len(loggs), len(mhs), len(alphas), len(coeffs))

        import h5py
        import numpy as np
        from datetime import datetime

        # K = 200  # number of PCA components you keep\
        mydtypestr = 'float32'

        if mydtypestr=='float32': mydtype = np.float32
        elif mydtypestr=='float64': mydtype = float


        with h5py.File(f"spectral_pca_grid_{CP.bs}.h5", "w") as f:
            # --- Metadata ---
            f.attrs["created_on"] = datetime.now().isoformat()
            f.attrs["author"] = "Paul Cristofari"
            f.attrs["description"] = ("PCA compressed ZeeTurbo spectral grid. "
                                    + f"Computed for BVAL={BVAL} kG. "
                                    + f"Only PCA_K={nb_comp} components kept out of "
                                    + f"K_FULL={K_full}, yielding MAX_ERR "
                                    + "difference in flux.")
            f.attrs["PCA_K"] = nb_comp
            f.attrs["MAX_ERR"] = maxerr
            f.attrs["BVAL"] = BVAL
            f.attrs["K_FULL"] = K_full
            f.attrs["DTYPE"] = mydtypestr
            

            # --- PCA data ---
            f.create_dataset(
                "mean_spectrum",
                data=mean_spectrum.astype(mydtype),
                compression="gzip",
                compression_opts=4
            )
            
            f['wave'] = h5py.ExternalLink('wave.hdf5', './wave')

            f.create_dataset(
                "basis",
                data=basis[:nb_comp].astype(mydtype),  # (K, Nλ)
                compression="gzip",
                compression_opts=4
            )

            f.create_dataset(
                "coeffs",
                data=coeffs_grid[..., :nb_comp].astype(mydtype),  # (nT, ng, nM, nA, K)
                compression="gzip",
                compression_opts=4,
                chunks=(1, 1, 1, 1, nb_comp)  # very important for fast access
            )

            # --- Grid axes (strongly recommended) ---
            f.create_dataset("teffs", data=teffs)
            f.create_dataset("loggs", data=loggs)
            f.create_dataset("mhs",   data=mhs)
            f.create_dataset("alphas", data=alphas)


    ## Example usage.

    # iT, ig, iM, iA = 0,0,0,0

    # with h5py.File(f"spectral_pca_grid_bval{CP.bs}.h5", "r") as f:
    #     mean_spectrum = f["mean_spectrum"][:]       # (Nλ,)
    #     basis = f["basis"][:]                       # (K, Nλ)
    #     coeffs = f["coeffs"][iT, ig, iM, iA, :]     # (K,)

    # spec = coeffs @ basis + mean_spectrum


    # plt.figure()
    # plt.plot(grid_n[0,0,0,0])
    # plt.plot(spec)
    # plt.show()

    pass

if __name__ == "__main__":
    main()

