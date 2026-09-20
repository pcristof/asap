import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from asap.SpectralAnalysis import SpectralAnalysis, read_res_v2
from configparser import ConfigParser

## IMPLEMENTATION BY PIC
## Dion's script relies on the copy_config.ini; which I would not to rely on.
## Dion's script was also loading the whole grid and using the full SA object,
## which let to compatibility clash with a new version.


def main():
    ## TODO: Change this to point to the results file only
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        description='Iteratively drop the highest magnetic component from a '
                    'completed ASAP full run, merge its filling factor into '
                    'the new highest component, and re-evaluate lnlike and '
                    'BIC. Accepts either a single output folder OR a '
                    'retrievals root directory (with --suffix) for batch mode.')
    parser.add_argument('path', type=str,
                        help='Either a single ASAP output folder (contains '
                             'results.txt) OR a retrievals root containing '
                             'per-star subfolders with output_*<suffix> dirs.')
    parser.add_argument('-s', '--suffix', type=str, default=None,
                        help='Required for batch mode: match output_* '
                             'directories ending with this suffix, e.g. '
                             '"filtered_new_grid_opt_llist_v1".')
    parser.add_argument('--skip-existing', action='store_true',
                        help='In batch mode, skip folders that already have '
                             'bic_scan/summary.csv.')
    parser.add_argument('-p', '--plot', action="store_true",
                        help='Plots')
    args = parser.parse_args()


    if args.path[-1]=='/': args.path = args.path[:-1]

    resfile = args.path+'/results.txt'
    SA = SpectralAnalysis()
    ## Start from the results file to load the object.
    SA.init_from_results(resfile)
    ## I need to check the paths, else I need to prompt for the new paths
    local_path_file = 'local_paths.ini'
    if not os.path.isfile(local_path_file):
        local_path_file = '../local_paths.ini'
    if not os.path.isfile(local_path_file):
        local_path_file = '../../local_paths.ini'
    
    if os.path.isfile('local_paths.ini'):
        config = ConfigParser()
        config.read('local_paths.ini')
        if config.has_option('PATHS', 'pathtogrid'):
            SA.set_pathtogrid(config['PATHS']['pathtogrid'])
        if config.has_option('PATHS', 'pathtodata'):
            SA.set_pathtodata(config['PATHS']['pathtodata'])
        if config.has_option('PATHS', 'linelistfile'):
            SA.set_linelist(config['PATHS']['linelistfile'])
        if config.has_option('PATHS', 'normfactorfile'):
            SA.set_normfacfile(config['PATHS']['normfactorfile'])

    ## But I do not want to reload everything...
    if len(SA.teffs)>1:
        _teffs = np.array(SA.teffs)
        if SA._T<_teffs[0]:
            _tefflow = _teffs[0]
            _teffup = _teffs[1]
        elif SA._T>_teffs[-1]:
            _tefflow = _teffs[-2]
            _teffup = _teffs[-1]
        else:
            _tefflow = _teffs[np.array(_teffs)<SA._T][-1]
            _teffup = _teffs[np.array(_teffs)>SA._T][0]
        _newteffs = [_tefflow, _teffup]
        SA.update_teffs(_newteffs)
    if len(SA.loggs)>1:
        ## But I do not want to reload everything...
        _loggs = np.array(SA.loggs)
        print(_loggs)
        if SA._L<_loggs[0]:
            _logglow = _loggs[0]
            _loggup = _loggs[1]
        elif SA._L>_loggs[-1]:
            _logglow = _loggs[-2]
            _loggup = _loggs[-1]
        else:
            _logglow = _loggs[np.array(_loggs)<SA._L][-1]
            _loggup = _loggs[np.array(_loggs)>SA._L][0]
        _newloggs = [_logglow, _loggup]
        SA.update_loggs(_newloggs)
    if len(SA.mhs)>1:
        ## But I do not want to reload everything...
        _mhs = np.array(SA.mhs)
        _mhlow = _mhs[np.array(_mhs)<SA._M][-1]
        _mhup = _mhs[np.array(_mhs)>SA._M][0]
        _newmhs = [_mhlow, _mhup]
    SA.update_mhs(_newmhs)
    if len(SA.alphas)>1:
        ## But I do not want to reload everything...
        _alphas = np.array(SA.alphas)
        _alphalow = _alphas[np.array(_alphas)<SA._A][-1]
        _alphaup = _alphas[np.array(_alphas)>SA._A][0]
        _newalphas = [_alphalow, _alphaup]
        SA.update_alphas(_newalphas)

    ## Read the results file
    res = read_res_v2(resfile)

    ## TODO: this should be changed to the input normfactor
    if 'norm_factor' in res:
        SA.set_normFactor(res['norm_factor'])

    ## Use this to update the path to the observation in case you have
    ## Changed the path
    infile = SA.pathtodata+res['input_filename'].split('/')[-1]


    print('---- Loading observations ----')
    med_wvl, med_spectrum, med_err, berv = SA.load_obs(infile)

    print('done loading observation')
    print('------------------------------')
    obs_wvl, obs_flux, obs_err, nan_mask, regions = SA.create_regions(
                                                        SA.linelist, med_wvl,
                                                        med_spectrum, med_err, 
                                                        berv)
    nwvls, grid_n, teffs, loggs, mhs, alphas = SA.load_grid(SA.pathtogrid, regions)
    print('Done loading grid')
    
    SA.init_PARAMS() ## To get the parameter keys

    lnLs = []
    bics = []
    maxmags = []

    lnL = float(SA.lnprob())
    nof = len(SA.PARAMS_FIT)
    N = SA.IDXTOFIT[0].shape[0]
    bic = float(-2.0 * lnL + nof * np.log(N))
    lnLs.append(lnL)
    bics.append(bic)
    maxmags.append(max(SA.bs))

    ## Now we iteratively redistribute the power of the field
    ## Keep a copy
    copy_coeffs = np.copy(SA.coeffs)
    copy_bs = np.copy(SA.bs)
    while len(SA.coeffs)>2:
        SA.update_bs(SA.bs[:-1])
        lastcoeff = SA.coeffs[-1]
        newcoeffs = SA.coeffs[:-1]
        # newcoeffs /= np.sum(newcoeffs)
        newcoeffs[-1]+=lastcoeff
        # copy_newcoeffs = np.copy(newcoeffs)
        # ## Where should I place the last coefficient?
        # _rec_i = 0
        # _rec_lnL = 0
        # for i in range(len(newcoeffs)):
        #     newcoeffs = np.copy(copy_newcoeffs)
        #     newcoeffs[i]+=lastcoeff
        #     SA.update_fillFactors(newcoeffs)
        #     SA.init_PARAMS()
        #     lnL = float(SA.lnprob())
        #     if lnL>_rec_lnL:
        #         _rec_newcoeffs = np.copy(newcoeffs)
        #         _rec_i = i
        #         _rec_lnL = lnL
        # print(_rec_i)
        # newcoeffs = _rec_newcoeffs
        ## Where should we place the last coefficient?
        SA.update_fillFactors(newcoeffs)
        SA.init_PARAMS()
        lnL = float(SA.lnprob())
        nof = len(SA.PARAMS_FIT)
        bic = float(-2.0 * lnL + nof * np.log(N))
        lnLs.append(lnL)
        bics.append(bic)
        maxmags.append(max(SA.bs))

    maxmags = np.array(maxmags)

    idxbic = np.where(bics==np.min(bics))[0][0]
    idxlnL = np.where(lnLs==np.max(lnLs))[0][0]

    print(f'Position min BIC: {maxmags[idxbic]} kG')
    print(f'Position max lnL: {maxmags[idxlnL]} kG')


    if args.plot:
        plt.figure()
        plt.plot(maxmags, bics)
        plt.show()
        plt.figure()
        plt.plot(maxmags, lnLs)
        plt.show()


    print(maxmags[idxbic])
    return maxmags[idxbic]

# def _regions_key(regions):
#     '''Hashable signature of the per-star region list for grid-cache lookup.'''
#     return tuple(tuple(np.round(np.asarray(r), 6).tolist()) for r in regions)


# def run_one(opath, grid_cache=None):
#     '''Run the BIC collapse scan on one output directory.

#     If `grid_cache` is provided and its key matches the current run's
#     (pathtogrid, regions, teffs, loggs, mhs, alphas), the cached grid
#     arrays are copied onto the fresh SpectralAnalysis instance instead
#     of reloading from disk. Returns the (possibly new) grid_cache so the
#     caller can pass it to subsequent invocations.
#     '''
#     if not opath.endswith('/'):
#         opath += '/'

#     config_file = opath + 'config_copy.ini'
#     if not os.path.isfile(config_file):
#         config_file = opath + 'config.ini'
#     results_file = opath + 'results.txt'
#     if not os.path.isfile(results_file):
#         raise FileNotFoundError(f'results.txt not found in {opath}')

#     SA = SpectralAnalysis()
#     SA.read_config(config_file)
#     res = read_res_v2(results_file)

#     star = res['star'].strip()
#     SA.set_star(star)
#     SA.set_opath(opath)

#     ## Overwrite SA attributes with best-fit values where available in results.txt.
#     ## Keys not written by save_results (e.g. vb, teff2, fillTeffs) stay at their
#     ## config-defined defaults.
#     _overrides = {
#         '_T': 'teff', '_L': 'logg', '_M': 'mh', '_A': 'afe',
#         'rv': 'rv', 'vsini': 'vsini', 'vmac': 'vmac', 'vb': 'vb',
#     }
#     for attr, key in _overrides.items():
#         if key in res:
#             setattr(SA, attr, res[key])
#     if SA.fitVeiling and 'veiling' in res:
#         SA.veilingFacToFit = np.array(res['veiling'])

#     ## Sanity check: mag_components from results must match config's bs
#     bs_from_res = np.array(res['mag_components'])
#     if not np.allclose(bs_from_res, SA.bs):
#         raise ValueError(f'mag_components in results.txt ({bs_from_res}) do not '
#                          f'match bs from config ({SA.bs})')

#     ## results.txt rounds ff to 4 decimals so the sum drifts ~1e-4 off 1.0.
#     ## Restore the sum-to-1 invariant via the derived-0kG rule (same as
#     ## unpackpar at SpectralAnalysis.py:3227).
#     ff = np.array(res['mag_ff'], dtype=float)
#     ff[0] = 1.0 - np.sum(ff[1:])
#     SA.update_fillFactors(ff)

#     ## The normFactor used during the MCMC is stored in results.txt; without
#     ## it, lnlike disagrees with lnlike_max because myerr = obs_err*sqrt(nf)
#     ## (SpectralAnalysis.py:3392). compute_normFactor would otherwise default
#     ## to 1.0 whenever the per-star entry is missing from normFactors.txt.
#     if 'norm_factor' in res:
#         SA.set_normFactor(res['norm_factor'])

#     ## Observations
#     pathtodata = SA.pathtodata
#     if not pathtodata.endswith('/'):
#         pathtodata += '/'
#     infile = pathtodata + f'{star}.fits'
#     infile_templates = pathtodata + f'{star}_templates.fits'
#     if os.path.isfile(infile_templates):
#         infile = infile_templates
#     elif not os.path.isfile(infile):
#         raise FileNotFoundError(
#             f'Observation file not found: {infile} or {infile_templates}')

#     print(f'---- Loading observations from {infile} ----')
#     med_wvl, med_spectrum, med_err, berv = SA.load_obs(infile)

#     obs_wvl, obs_flux, obs_err, nan_mask, regions = SA.create_regions(
#         SA.linelist, med_wvl, med_spectrum, med_err, berv)

#     ## Decide whether to reuse the cached grid or load from disk. Key is built
#     ## from the *pre-load* config values because load_grid mutates SA.teffs /
#     ## SA.loggs / ... when the requested grid is adjusted. Two stars that
#     ## share the same pathtogrid + regions + requested axes will produce the
#     ## same loaded grid, so caching on pre-load inputs is sufficient.
#     cache_key = (
#         os.path.abspath(SA.pathtogrid),
#         _regions_key(regions),
#         tuple(np.asarray(SA.teffs).tolist()),
#         tuple(np.asarray(SA.loggs).tolist()),
#         tuple(np.asarray(SA.mhs).tolist()),
#         tuple(np.asarray(SA.alphas).tolist()),
#     )
#     if grid_cache is not None and grid_cache.get('key') == cache_key:
#         print('---- Reusing cached grid ----')
#         SA.nwvls = grid_cache['nwvls']
#         SA.grid_n = grid_cache['grid_n']
#         SA.teffs = grid_cache['teffs']
#         SA.loggs = grid_cache['loggs']
#         SA.mhs = grid_cache['mhs']
#         SA.alphas = grid_cache['alphas']
#         SA.regions = regions
#         SA.diskIntegrationMode = grid_cache['diskIntegrationMode']
#         SA.get_grid_dims()
#     else:
#         if grid_cache is not None:
#             prev = grid_cache['key']
#             diffs = [name for name, a, b in zip(
#                 ['pathtogrid', 'regions', 'teffs', 'loggs', 'mhs', 'alphas'],
#                 prev, cache_key) if a != b]
#             print(f'---- Cache miss (differs in: {diffs}) — loading grid ----')
#         else:
#             print('---- Loading grid ----')
#         SA.load_grid(SA.pathtogrid, regions)
#         print('Done loading grid')
#         grid_cache = {
#             'key': cache_key,
#             'nwvls': SA.nwvls,
#             'grid_n': SA.grid_n,
#             'teffs': SA.teffs,
#             'loggs': SA.loggs,
#             'mhs': SA.mhs,
#             'alphas': SA.alphas,
#             'diskIntegrationMode': SA.diskIntegrationMode,
#         }

#     ## save_results counts BIC's nof from `len(mcmcs)`, and mcmcs includes
#     ## the derived 0 kG filling factor prepended in nssamples. So the original
#     ## BIC used (# MCMC-fitted params) + 1 when fitFields was True. We match
#     ## that convention here and decrement by 1 per dropped component.
#     nof0 = len(SA.return_labels()) + (1 if SA.fitFields else 0)
#     N = len(SA.obs_flux_tofit[SA.IDXTOFIT])
#     print(f'Original nof0 = {nof0}, N (points fitted) = {N}')

#     bicdir = opath + 'bic_scan/'
#     os.makedirs(bicdir, exist_ok=True)

#     def compute_iter_result(drops):
#         fit = SA.gen_spec(
#             SA.obs_wvl, SA.obs_flux, SA.obs_err,
#             SA.nan_mask, SA.nwvls, SA.grid_n,
#             SA.coeffs, SA._T, SA._L, SA._M, SA._A,
#             SA.teffs, SA.loggs, SA.mhs, SA.alphas,
#             SA.vb, SA.rv, SA.vsini, SA.vmac,
#             SA.veilingFacToFit, SA._T2, SA.fillTeffs,
#         )
#         lnL = SA.lnlike(par=None)
#         nof = nof0 - drops
#         bic = -2.0 * lnL + nof * np.log(N)
#         ## AIC uses the same nof and lnL as BIC; only the penalty term differs
#         ## (2*k instead of k*ln(N)). No extra inputs required.
#         aic = 2.0 * nof - 2.0 * lnL
#         return fit, lnL, nof, bic, aic

#     records = []
#     spectra = []
#     coeffs_history = []

#     drops = 0
#     fit, lnL, nof, bic, aic = compute_iter_result(drops)
#     records.append(dict(iteration=drops, n_components=len(SA.bs),
#                         max_B_kG=float(SA.bs[-1]), n_params=nof,
#                         lnlike_max=float(lnL), BIC=float(bic), AIC=float(aic)))
#     spectra.append(fit)
#     coeffs_history.append(SA.coeffs.copy())
#     baseline_bic = bic
#     baseline_aic = aic
#     print(f'Iter {drops}: n_comp={len(SA.bs)}, maxB={SA.bs[-1]} kG, '
#           f'nof={nof}, lnL={lnL:.4f}, BIC={bic:.4f}, AIC={aic:.4f}')
#     print(f'  results.txt reference: lnlike_max={res["lnlike_max"]:.4f}, '
#           f'BIC={res["bic"]:.4f}')
#     ## results.txt stores lnlike/BIC at 4-decimal precision, so small
#     ## round-trip disagreement is expected; warn only if larger than that.
#     if abs(lnL - res['lnlike_max']) > 5e-2:
#         print(f'  WARNING: baseline lnlike mismatch '
#               f'({lnL} vs {res["lnlike_max"]})')
#     if abs(bic - res['bic']) > 1e-1:
#         print(f'  WARNING: baseline BIC mismatch ({bic} vs {res["bic"]})')

#     while len(SA.bs) > 1:
#         drops += 1
#         new_coeffs = SA.coeffs[:-1].copy()
#         new_coeffs[-1] += SA.coeffs[-1]
#         new_bs = SA.bs[:-1]
#         SA.grid_n = SA.grid_n[:-1]
#         SA.update_bs(new_bs)
#         SA.update_fillFactors(new_coeffs)

#         assert np.isclose(np.sum(SA.coeffs), 1.0), \
#             f'FF sum {np.sum(SA.coeffs)} != 1'
#         assert SA.grid_n.shape[0] == len(SA.bs), \
#             f'grid_n axis mismatch: {SA.grid_n.shape[0]} vs {len(SA.bs)}'

#         fit, lnL, nof, bic, aic = compute_iter_result(drops)
#         records.append(dict(iteration=drops, n_components=len(SA.bs),
#                             max_B_kG=float(SA.bs[-1]), n_params=nof,
#                             lnlike_max=float(lnL), BIC=float(bic),
#                             AIC=float(aic)))
#         spectra.append(fit)
#         coeffs_history.append(SA.coeffs.copy())
#         print(f'Iter {drops}: n_comp={len(SA.bs)}, maxB={SA.bs[-1]} kG, '
#               f'nof={nof}, lnL={lnL:.4f}, BIC={bic:.4f}, AIC={aic:.4f}')

#     assert len(SA.bs) == 1 and SA.bs[0] == 0, \
#         f'End state invalid: bs={SA.bs}'
#     assert np.isclose(SA.coeffs[0], 1.0), \
#         f'End state coeffs not [1.0]: {SA.coeffs}'

#     ## Summary CSV
#     with open(bicdir + 'summary.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['iteration', 'n_components', 'max_B_kG', 'n_params',
#                          'lnlike_max', 'BIC', 'delta_BIC', 'AIC', 'delta_AIC'])
#         for r in records:
#             writer.writerow([r['iteration'], r['n_components'], r['max_B_kG'],
#                              r['n_params'], r['lnlike_max'],
#                              r['BIC'], r['BIC'] - baseline_bic,
#                              r['AIC'], r['AIC'] - baseline_aic])

#     ## Filling-factor history CSV
#     B_all = bs_from_res
#     nit = len(coeffs_history)
#     ff_matrix = np.full((len(B_all), nit), np.nan)
#     for i, ff in enumerate(coeffs_history):
#         ff_matrix[:len(ff), i] = ff
#     with open(bicdir + 'filling_factors.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['B_kG'] + [f'iter_{i}' for i in range(nit)])
#         for bi, B in enumerate(B_all):
#             writer.writerow([float(B)] + [float(v) if not np.isnan(v) else ''
#                                           for v in ff_matrix[bi]])

#     ## BIC + AIC curves on shared axes (same units: -2*lnL + penalty)
#     params = {'xtick.labelsize': 18, 'ytick.labelsize': 18,
#               'axes.labelsize': 20, 'legend.fontsize': 14,
#               'figure.figsize': (6.4, 4.8)}
#     plt.rcParams.update(params)
#     fig, ax = plt.subplots()
#     max_B = [r['max_B_kG'] for r in records]
#     n_comps = [r['n_components'] for r in records]
#     bics = [r['BIC'] for r in records]
#     aics = [r['AIC'] for r in records]
#     min_idx = int(np.argmin(bics))
#     min_aic_idx = int(np.argmin(aics))
#     ax.plot(max_B, bics, marker='o', color='black', label='BIC')
#     ax.plot(max_B, aics, marker='s', color='tab:blue', label='AIC')
#     ax.axvline(max_B[min_idx], color='red', ls='--',
#                label=f'min BIC @ max B = {max_B[min_idx]:.0f} kG')
#     ax.axvline(max_B[min_aic_idx], color='tab:orange', ls=':',
#                label=f'min AIC @ max B = {max_B[min_aic_idx]:.0f} kG')
#     ax.set_xlabel('Max magnetic component (kG)')
#     ax.set_ylabel('Information criterion')
#     ax.legend()
#     ax.tick_params(which='major', direction='in', axis='both',
#                    bottom=True, top=True, left=True, right=True, length=10)
#     ax.tick_params(which='minor', direction='in', axis='both',
#                    bottom=True, top=True, left=True, right=True, length=5)
#     plt.tight_layout()
#     plt.savefig(bicdir + 'bic_curve.pdf')
#     plt.close()

#     ## Spectra FITS
#     hdus = [fits.PrimaryHDU()]
#     hdus[0].header['STAR'] = star
#     hdus[0].header['OPATH'] = opath
#     hdus[0].header['NOF0'] = nof0
#     hdus[0].header['NPOINTS'] = N
#     hdus.append(fits.ImageHDU(data=np.asarray(SA.obs_wvl), name='WVL'))
#     for i, (rec, spec) in enumerate(zip(records, spectra)):
#         h = fits.ImageHDU(data=np.asarray(spec), name=f'FIT_I{i}')
#         h.header['MAXB_KG'] = rec['max_B_kG']
#         h.header['NCOMPS'] = rec['n_components']
#         h.header['LNLIKE'] = rec['lnlike_max']
#         h.header['BIC'] = rec['BIC']
#         h.header['AIC'] = rec['AIC']
#         hdus.append(h)
#     fits.HDUList(hdus).writeto(bicdir + 'spectra.fits', overwrite=True)

#     print(f'\nDone. Outputs in {bicdir}')
#     print(f'Min BIC at n_components={n_comps[min_idx]}, '
#           f'BIC={bics[min_idx]:.4f}, delta={bics[min_idx]-baseline_bic:.4f}')
#     print(f'Min AIC at n_components={n_comps[min_aic_idx]}, '
#           f'AIC={aics[min_aic_idx]:.4f}, '
#           f'delta={aics[min_aic_idx]-baseline_aic:.4f}')

#     return grid_cache


# def _is_output_folder(path):
#     '''An ASAP output folder contains results.txt (new format) or config_copy.ini.'''
#     return (os.path.isfile(os.path.join(path, 'results.txt'))
#             or os.path.isfile(os.path.join(path, 'config_copy.ini')))


# def run_batch(root, suffix, skip_existing=False):
#     '''Iterate run_one over every output_*<suffix> folder under `root`,
#     sharing the grid cache across stars where possible. A tqdm progress bar
#     shows current star + min-BIC postfix; per-star chatter is redirected to
#     bic_scan/run.log inside each target folder.'''
#     import glob
#     import sys
#     import time
#     import traceback
#     from contextlib import redirect_stdout, redirect_stderr

#     try:
#         from tqdm.auto import tqdm
#     except ImportError:
#         tqdm = None

#     pattern = os.path.join(root, '*', f'output_*{suffix}')
#     targets = sorted(glob.glob(pattern))
#     if not targets:
#         print(f'No directories matched {pattern}')
#         sys.exit(1)

#     print(f'Found {len(targets)} target directories under {root}')
#     print(f'Suffix: {suffix}')

#     grid_cache = None
#     failures = []
#     skipped = 0

#     bar = tqdm(targets, desc='BIC scan', unit='star', dynamic_ncols=True) \
#         if tqdm is not None else targets

#     for opath in bar:
#         star_label = os.path.basename(opath.rstrip('/'))
#         if tqdm is not None:
#             bar.set_description(f'BIC scan [{star_label}]')

#         summary = os.path.join(opath, 'bic_scan', 'summary.csv')
#         if skip_existing and os.path.isfile(summary):
#             skipped += 1
#             if tqdm is not None:
#                 bar.set_postfix_str('skipped (exists)')
#                 tqdm.write(f'SKIP {star_label}')
#             else:
#                 print(f'SKIP {star_label}')
#             continue

#         log_dir = os.path.join(opath, 'bic_scan')
#         os.makedirs(log_dir, exist_ok=True)
#         log_path = os.path.join(log_dir, 'run.log')

#         t0 = time.time()
#         try:
#             with open(log_path, 'w') as lf:
#                 with redirect_stdout(lf), redirect_stderr(lf):
#                     grid_cache = run_one(opath, grid_cache=grid_cache)
#             dt = time.time() - t0

#             ## Pull min-BIC summary from the freshly written CSV
#             postfix = f'{dt:.0f}s'
#             try:
#                 import csv as _csv
#                 with open(summary) as sf:
#                     rows = list(_csv.DictReader(sf))
#                 best = min(rows, key=lambda r: float(r['BIC']))
#                 postfix = (f'{dt:.0f}s  min@B<={float(best["max_B_kG"]):.0f}kG '
#                            f'BIC={float(best["BIC"]):.0f}')
#             except Exception:
#                 pass

#             if tqdm is not None:
#                 bar.set_postfix_str(postfix)
#                 tqdm.write(f'OK   {star_label}  ({postfix})')
#             else:
#                 print(f'OK   {star_label}  ({postfix})')

#         except Exception:
#             dt = time.time() - t0
#             err_log = os.path.join(opath, 'bic_scan_error.log')
#             with open(err_log, 'w') as f:
#                 traceback.print_exc(file=f)
#             failures.append(opath)
#             msg = f'FAIL {star_label}  ({dt:.0f}s) — see {err_log}'
#             if tqdm is not None:
#                 bar.set_postfix_str('FAILED')
#                 tqdm.write(msg)
#             else:
#                 print(msg)

#     if tqdm is not None:
#         bar.close()

#     print()
#     total = len(targets)
#     ok = total - len(failures) - skipped
#     print(f'Completed {ok}/{total} successfully '
#           f'({skipped} skipped, {len(failures)} failed).')
#     if failures:
#         print('Failed folders:')
#         for f in failures:
#             print(f'  {f}')
#         sys.exit(2)


# # def main():
# #     import argparse
# #     import sys
# #     parser = argparse.ArgumentParser(
# #         description='Iteratively drop the highest magnetic component from a '
# #                     'completed ASAP full run, merge its filling factor into '
# #                     'the new highest component, and re-evaluate lnlike and '
# #                     'BIC. Accepts either a single output folder OR a '
# #                     'retrievals root directory (with --suffix) for batch mode.')
# #     parser.add_argument('path', type=str,
# #                         help='Either a single ASAP output folder (contains '
# #                              'results.txt) OR a retrievals root containing '
# #                              'per-star subfolders with output_*<suffix> dirs.')
# #     parser.add_argument('-s', '--suffix', type=str, default=None,
# #                         help='Required for batch mode: match output_* '
# #                              'directories ending with this suffix, e.g. '
# #                              '"filtered_new_grid_opt_llist_v1".')
# #     parser.add_argument('--skip-existing', action='store_true',
# #                         help='In batch mode, skip folders that already have '
# #                              'bic_scan/summary.csv.')
# #     args = parser.parse_args()

# #     if _is_output_folder(args.path):
# #         if args.suffix is not None:
# #             print('Note: --suffix is ignored when `path` is a single output folder.')
# #         run_one(args.path)
# #     else:
# #         if args.suffix is None:
# #             parser.error(
# #                 f'{args.path} does not look like an ASAP output folder '
# #                 f'(no results.txt or config_copy.ini). Treating as a '
# #                 f'retrievals root requires --suffix <suffix>.')
# #         run_batch(args.path, args.suffix, skip_existing=args.skip_existing)


if __name__ == '__main__':
    val = main()
    # with open('tmpfile.txt', 'w'):
    #     f.write(val)