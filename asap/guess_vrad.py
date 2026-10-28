import numpy as np
from scipy.interpolate import interp1d
from asap.analysis_tools import doppler
from asap.ccf import ccf_2d
from asap import mask_tools as msk_tls
from importlib.resources import files
import glob

def guess_vrad_5(med_wvl, med_spectrum, teff=None, return_ccf=False):
    '''New attempt at deriving the radial velocity from a star.
    In this version we try to nor resample the observation spectrum
    TODO: Could be improve by fitting a gaussian onto the central peak
    TODO: Handle cases were we have different wavelength ranges !
    '''
    #
    # order = 33 # order used to compute cross corr
    resample=True
    # data_folder = paths.data_folder
    
    ## Here we want to check the temperatures against the available files to take the closest temperature?
    # p = files("asap.support_data")    
    # available_files = glob.glob(str(p)+'/vald*')  
    # teffs = []
    # for file in available_files:
    #     if file.split('/')[-1]=='vald-lines': continue ## Backward compatibility
    #     else: _teff = int(file.split('/')[-1].split('_')[2]); teffs.append(_teff)
    # res = np.abs(teff-np.array(teffs))
    # ii = np.where(res==res.min())[0][0]
    # ## Overwrite the passed teff, because now we are useing the closest available value.
    # _teff = teffs[ii]

    _teff = teff
    
    w1 = np.nanmin(med_wvl); w2 = np.nanmax(med_wvl)
    dlam1 = 300*w1/(3*1e5); dlam2 = 300*w2/(3*1e5) 
    bounds=[w1-dlam1, w2+dlam2]
    w, f = msk_tls.read_vald_lines(teff=_teff, 
                                   wave_bounds=bounds,
                                   units='nm') ## VALD line list
    if w is None:
        ## Raise error if we could not find a proper mask
        message = 'Could not find adequate mask for guess_vrad.\n'\
                  + 'Check that you have the correct support data, '\
                  + 'Contact the author, '\
                  + 'or run the code with guess_vrad=False.\n'
        raise Exception(message)

    ## Find the very raw first estimate
    w = np.array(w, dtype=float)
    f = np.array(f, dtype=float)
    med_wvl = np.array(med_wvl, dtype=float)
    med_spectrum = np.array(med_spectrum, dtype=float)
    rvshifts, corrp, div = ccf_2d(w, f, med_wvl, med_spectrum, step=2, span=500, function='cubic')
    rvshifts1, corrp1, div1 = np.copy(rvshifts), np.copy(corrp), np.copy(div)
    # from IPython import embed
    # embed()
    idx = np.where(corrp/div==np.min(corrp/div))
    rv0 = rvshifts[idx][0]



    rvshifts, corrp, div = ccf_2d(w*doppler(-rv0), f, med_wvl, med_spectrum, step=.1, span=50, function='cubic')

    new_rvshifts = np.arange(rvshifts[1], rvshifts[-2], 0.01)
    fun = interp1d(rvshifts, corrp/div, 'cubic')
    new_corrpdiv = fun(new_rvshifts)

    idx = np.where(new_corrpdiv==np.min(new_corrpdiv))
    rv =new_rvshifts[idx][0]

    # from IPython import embed; embed()

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(rvshifts+rv0, corrp/div, color='black')
    ## plt.plot(rvshifts1, corrp1/div1, color='red')
    #plt.plot(new_rvshifts+rv0, new_corrpdiv, '--',  color='black')
    ## plt.plot(rvshifts_c, corrp_c/div_c, '--',  color='red')
    ## plt.plot(new_rvshifts1, new_corrpdiv1, '--', color='red')
    #plt.show()

    if return_ccf:
        return rv+rv0, rvshifts1, corrp1/div1, rvshifts, corrp/div
    else:
        return rv+rv0
