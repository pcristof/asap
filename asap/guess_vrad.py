import numpy as np
from scipy.interpolate import interp1d
from asap.analysis_tools import doppler
from asap.ccf import ccf_2d
from asap import mask_tools as msk_tls
from importlib.resources import files
import glob

def guess_vrad_5(med_wvl, med_spectrum, teff=None):
    '''New attempt at deriving the radial velocity from a star.
    In this version we try to nor resample the observation spectrum
    TODO: Could be improve by fitting a gaussian onto the central peak
    '''
    #
    # order = 33 # order used to compute cross corr
    resample=True
    # data_folder = paths.data_folder
    
    ## Here we want to check the temperatures against the available files to take the closest temperature?
    p = files("asap.support_data")    
    available_files = glob.glob(str(p)+'/vald*')  
    teffs = []
    for file in available_files:
        if file.split('/')[-1]=='vald-lines': continue ## Backward compatibility
        else: _teff = int(file.split('/')[-1].split('_')[2]); teffs.append(_teff)
    res = np.abs(teff-np.array(teffs))
    ii = np.where(res==res.min())[0][0]
    ## Overwrite the passed teff, because now we are useing the closest available value.
    _teff = teffs[ii]
    
    w, f = msk_tls.read_vald_lines(teff=_teff) ## VALD line list

    ## Find the very raw first estimate
    w = np.array(w, dtype=float)
    f = np.array(f, dtype=float)
    med_wvl = np.array(med_wvl, dtype=float)
    med_spectrum = np.array(med_spectrum, dtype=float)
    rvshifts, corrp, div = ccf_2d(w, f, med_wvl, med_spectrum, step=2, span=500, function='cubic')
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

    #from IPython import embed; embed()

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(rvshifts+rv0, corrp/div, color='black')
    ## plt.plot(rvshifts1, corrp1/div1, color='red')
    #plt.plot(new_rvshifts+rv0, new_corrpdiv, '--',  color='black')
    ## plt.plot(rvshifts_c, corrp_c/div_c, '--',  color='red')
    ## plt.plot(new_rvshifts1, new_corrpdiv1, '--', color='red')
    #plt.show()

    return rv+rv0
