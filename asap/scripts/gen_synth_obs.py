def main():
    from asap.SpectralAnalysis import SpectralAnalysis
    import os
    from importlib.resources import files
    from astropy.io import fits
    import numpy as np
    from asap.spectral_analysis_pack import wrap_function_fine_linear_4d
    from asap import analysis_tools as tls
    from asap import effects as effects
    from datetime import datetime
    import sys
    import argparse ## To read optional arguments
    from asap import integrate

    parser = argparse.ArgumentParser()
    # parser.add_argument("star", type=str)
    # parser.add_argument("folderid", nargs='?', type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default='test')
    parser.add_argument("-std", "--std", type=float, default=0.02)
    # parser.add_argument("-m", "--mpi", type=bool, default=False)
    # parser.add_argument("-p", "--profile", type=bool, default=False)
    # parser.add_argument("-d", "--dynesty", type=bool, default=False)
    parser.add_argument("-w", "--overwrite", action='store_true')

    args = parser.parse_args()
    sigma = args.std
    star = args.output
    overwrite = args.overwrite

    locpath = os.getcwd()
    config_file = locpath + '/config.ini'

    SA = SpectralAnalysis()
    SA.read_config(config_file)
    SA.resampleVel = False

    ## Check that the directory exists:
    if not os.path.isdir(SA.pathtogrid):
        print('Requested grid directory does not exist.')
        print('Please update your config.ini file.')
        print('PROGRAM END')
        exit()

    ## output filename
    fname = SA.pathtodata + '/{}.fits'.format(star)
    if os.path.isfile(fname):
        if (not overwrite):
            print(f'File {fname} already exists.')
            print('If you wish to overwrite, use the -w option.')
            print('PROGRAM END')
            return 0

    _t,_l,_m,_a = SA.interpret_grid_dimensions(SA.pathtogrid)

    wavfilename = '240041F6T7c_pp_e2dsff_AB_wavem_fp_AB.fits'
    ref_wvl_file = files("asap.support_data").joinpath(wavfilename)
    
    ##################################
    #### Load typical SPIRou data ####
    med_wvl = fits.getdata(ref_wvl_file)
    med_wvl = np.array(med_wvl, dtype=float)*10

    ## Load the grid of spectra for the whole spetrum
    regions = [[np.min(med_wvl)-100, np.max(med_wvl)+100]]
    SA.regions = regions
    SA.obs_wvl = med_wvl
    SA.get_grid_dims()

    nwvls, grid_n, teffs, loggs, mhs, alphas = SA.load_grid(SA.pathtogrid, regions)
    print('Done loading grid')

    _Bspec = np.zeros((SA.d5, SA.d6, SA.d7))
    for i in range(SA.d5):
        _, s = wrap_function_fine_linear_4d(
                                        SA._T, SA._L, SA._M, SA._A,
                                        teffs, loggs, mhs, alphas,
                                        grid_n[i], 
                                        function=SA.interpFunc)
        _Bspec[i] = s
    tosum = [SA.coeffs[i] * _Bspec[i] for i in range(len(SA.coeffs))]

    ## Here are the wavelength and interp. spectrum we are considering
    ## [0] because we only have one "region" here
    _wvl = nwvls[0]
    _spectrum = np.sum(tosum, axis=0)[0] ## non-broad non-adj magnetic model

    ##########################
    #### Split the arrays ####
    #
    # Now this is where things begin to be tricky. Our spectra now have holes in
    # them, because we only computed regions to be faster on the synthesis.
    # This will be a problem if we simply interpolate the whole thing on a SPIRou
    # grid because the convolution function, or resampling functions will create
    # weird effects.
    #  
    # Let's therefore begin by deviding the spectrum based on the wavelength 
    # solution.

    _wvldiff = np.round(np.diff(_wvl), 4); _wvlstep = np.round(_wvldiff[0], 4); 
    idx = np.where(_wvldiff>_wvlstep)[0]+1 # We split after the wvl jump (+1)
    _wvlsplit = np.split(_wvl, idx)
    _spectrumsplit = np.split(_spectrum, idx)

    _wvlsplit = []
    _spectrumsplit = []
    lims = []
    limlow = 0
    dwvlseg = np.round(_wvl[2], 3) - np.round(_wvl[1], 3)
    for i in range(len(_wvl)-1):
        dwvl = np.round(_wvl[i+1], 3) - np.round(_wvl[i], 3)
        if (i+1)>=(len(_wvl)-1):
            lims.append([limlow, -1])
            _wvlsplit.append(_wvl[limlow:])
            _spectrumsplit.append(_spectrum[limlow:])
        elif dwvl > 1:
            limhigh = i+1
            lims.append([limlow, limhigh])
            _wvlsplit.append(_wvl[limlow:limhigh])
            _spectrumsplit.append(_spectrum[limlow:limhigh])
            dwvlseg = np.round(_wvl[i+2], 3) - np.round(_wvl[i+1], 3)
            limlow = i+1

    ##########################

    #################################
    #### Integrate on SPIRou wvl ####
    #
    # Now that we have correctly splitted the array, we can loop through the
    # typical SPIRou orders and integrate the spectrum on each order.
    #
    # Lets loop
    med_spectrum = np.ones(med_wvl.shape)
    for r in range(len(med_wvl)-1):
        nb = 0
        # Loop through the nealy created regions
        for rr in range(len(_wvlsplit)):
            subwvl = _wvlsplit[rr]
            subflux = _spectrumsplit[rr]
            locmedwvl = med_wvl[r]
            # Is this region NOT in the order?
            if subwvl[-1]<locmedwvl[0]: continue # ignore region for this order
            if subwvl[0]>locmedwvl[-1]: continue # ignore region for this order
            # Is region partially in order?
            if (subwvl[0]<locmedwvl[0]):
                idx = subwvl>locmedwvl[0]
                subwvl = subwvl[idx]
                subflux = subflux[idx]
            if (subwvl[-1]>locmedwvl[-1]):
                idx = subwvl<locmedwvl[-1]
                subwvl = subwvl[idx]
                subflux = subflux[idx]
            # We resample the spectrum in velocity. We assume the region small
            # enough for this to work well.
            subwvl, subflux = tls.resample_vel_interp(subwvl, subflux, kind='cubic')

            # We then apply the effects, and we should be confident with our 
            # fourier method since we are now in velocity.
            # subflux = tls.broadened_profile(subwvl, subflux, vmac=vb, 
            #                                      vinstru=vinstru, vsini=0)
            #
            subflux = effects.broaden_spectrum_2(subwvl, subflux, vmac=SA.vmac, 
                                                vinstru=SA.vinstru, 
                                                vsini=SA.vsini,
                                                vmac_mode='rt')
            # val = np.sqrt(vb**2 + vinstru**2)
            # subflux = convolve.convolve(np.array(subwvl, dtype=float), 
            #                             np.array(subflux, dtype=float),  -val)
            # Now the problem is that we must put the wavelength in place
            idx = np.where((locmedwvl>=subwvl[0]) & (locmedwvl<=subwvl[-1]))
            # locmedflux = inte.fftintegrate(locmedwvl[idx], subwvl, subflux)
            import time
            itime = time.time()
            locmedflux = np.interp(locmedwvl[idx], subwvl, subflux)
            etime = time.time()
            print(f'np.interp time: {(etime-itime)*1000:0.4f} ms')
            itime = time.time()
            locmedflux = integrate.integrate(locmedwvl[idx], subwvl, subflux, 
                                             dspeed=2000., auto=True)
            etime = time.time()
            print(f'integrate time: {(etime-itime)*1000:0.4f} ms')
            # And finally we put populuate the output array
            med_spectrum[r, idx] = locmedflux
    #
    #################################

    ####################################
    #### Add realistic noise to spectrum
    # sigma = 0.05
    med_spectrum_noisy, noise = tls.add_gaussian_noise(med_spectrum, 
                                                         sigma=sigma)
    med_err = sigma*np.ones(med_spectrum_noisy.shape)
    ####################################

    ################################
    #### Save data to fits file ####
    version = -1
    nbOfSpectra = nbOfDates = 0
    mjdays = [0,0,0]
    medDate = 0
    snr = 1/sigma
    snrtemp = snr
    filling_factors = SA.coeffs
    bfield = np.sum(SA.coeffs*SA.bs)
    inparams = [SA._T, SA._L, SA._M, SA._A, bfield]
    berv = 0
    nbOfRejectedSpectra = 0
    nbOfMD = 0
    now = datetime.now()
    #
    hdu = fits.PrimaryHDU()
    hdu.header['VERSION'] = ('{}'.format(version), 'FakeSpectrumGenerator')
    hdu.header['NSPECTRA'] = ('{}'.format(nbOfSpectra), 'number of spectra')
    hdu.header['NEPOCHS'] = ('{}'.format(nbOfDates), 'number of epochs')
    hdu.header['MINDATE'] = ('{}'.format(np.min(mjdays)), 'oldest spectrum mjdate')
    hdu.header['MAXDATE'] = ('{}'.format(np.max(mjdays)), 'latest spectrum mjdate')
    hdu.header['MEANDATE'] = ('{}'.format(np.mean(mjdays)), 'mean mjdate')
    hdu.header['MEDDATE'] = ('{}'.format(medDate), 'median mjdate')
    hdu.header['DATE'] = ('{}'.format(now), 'Date and time template was computed')
    hdu.header['SNRMEAN'] = ('{:0.2f}'.format(np.mean(snr)), 'Average SNR')
    hdu.header['SNRMED'] = ('{:0.2f}'.format(np.median(snr)), 'Median SNR')
    hdu.header['SNRMIN'] = ('{:0.2f}'.format(np.min(snr)), 'Minimum SNR')
    hdu.header['SNRMAX'] = ('{:0.2f}'.format(np.max(snr)), 'Maximum SNR')
    hdu.header['SNRTEMP'] = ('{:0.2f}'.format(snrtemp), 
                                            'Template SNR (SNRMED/SQRT(NSPECTRA))')
    hdu.header['BERVMEAN'] = ('{:0.2f}'.format(np.mean(berv)), 'Average BERV')
    hdu.header['BERVMED'] = ('{:0.2f}'.format(np.median(berv)), 'Median BERV')
    hdu.header['BERVMIN'] = ('{:0.2f}'.format(np.min(berv)), 'Minimum BERV')
    hdu.header['BERVMAX'] = ('{:0.2f}'.format(np.max(berv)), 'Maximum BERV')
    hdu.header['NMD'] = ('{}'.format(nbOfMD), 'Number of spectra with different' +\
                                            ' months and days')
    hdu.header['NREJSPEC'] = ('{}'.format(nbOfRejectedSpectra), 
                                        'Number of spectra rejected (SNR<50)')
    hdu.header['INPARAMS'] = ('{}'.format(inparams), 
                                        'Input teff, logg, mh, alpha and bfield')
    hdu.header['INFILL'] = ('{}'.format(filling_factors), 
                                        'Input filling factors if relevant')
    hdu1 = fits.ImageHDU(data=med_wvl/10, name='WVL')
    hdu2 = fits.ImageHDU(data=med_spectrum_noisy, name='TEMPLATE')
    hdu3 = fits.ImageHDU(data=med_err, name='ERR')
    hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3])
    hdul.writeto(fname, overwrite=overwrite)
    print(f'File created {fname}')
    ################################


if __name__ == "__main__":
    main()

