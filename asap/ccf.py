"""Module containing CCF functions specifically desgined to work with masks.
Author: Paul I. Cristofari
Modified: Oct. 24, 2023

Rationale: CCF functions sometimes do simple interpolation over NaNs
or things that I want to avoid. Here I attempt to never encounter a NaN while doing
the CCF. I use a cubic interpolation for the spectrum, using scipy interp1d. The function can
easily be changed (input of the function)

There are two functions. 1 handling a 1d spectrum (like a single order of an echelle spectrum).
The second loops through the orders of a 2d spectrum (first dimension being the orders).

If you find errors or bugs: paul.cristofari@cfa.harvar.edu.

Dependencies:
numpy, scipy, and a perso function that I wrote for the doppler shift.

That is for you Bonnie:
You can comment the jit line if you don't have numba

@jit(nopython=True)
def doppler(v):
    '''Returns relativistic doppler factor in km/s.
    Caution: conventions assume that the v is negative if the source and the 
    object are moving towards each other, positive if they are moving away 
    from each other.'''
    _v = v*1e3
    return np.sqrt( (1 + _v/cst.c) / (1 - _v/cst.c))

"""

import numpy as np
from asap.analysis_tools import doppler
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit
from asap import mask_tools as msk_tls


@jit(nopython=True, cache=True)
def ccf_jit2(XW, XF, W, Y, span=100, step=2, step_ths=5, plot=False, function='cubic', renorm=True):
    '''
    SAME AS CCF but Numba compatible

    return what is needed for the ccf CCF. Note that the normalized CCF is corrp/div.
    What I call line mask is a list of intensities associated to a list of wavelengths.
    The CCF will be computed on a -span/2 +span/2 window, in steps of step km/s.
    The step_ths is used to find gaps to avoid. For example, by default, I do not interpolate
    between two pixels that are more than 5 km/s apart.
    Inputs:
    - XW        : wavelength of a line mask
    - XF        : intensity of each line in the mask
    - W         : wavelengths of the spectrum
    - Y         : spectrum
    - span      : velocity span of the CCF (in km/s)
    - step      : velocity step at which you want the CCF to be sampled (km/s)
    - step_ths  : velocity step used to split the input spectrum (km/s)
    - function  : scipy.interp1d function used to interpolate the spectrum. Default is cubic. 
    - renorm    : renormalize XF and Y by removing the median of XF and Y respectively. Default is True.
    '''
    #
    ## Define the RV shifts
    rvshifts = np.arange(1, span, step)
    rvshifts = rvshifts - span//2
    defaults = np.ones(len(rvshifts))
    
    _XW_full, _XF_full = XW, XF ## Unpack wvl and intensity of mask
    _W_full, _Y_full = W, Y ## Unpack wvl and intensity of mask
    cvel = 3*1e5

    ## Check
    if len(_W_full)!=len(_Y_full):
        print('Failed; shape(_W)!=shape(_Y)')
        return rvshifts, defaults, defaults
    
    ## Check that the input wavelength is increasing
    diff = np.diff(_W_full)
    if np.any(diff<0):
        print('Failed; W must be increasing wavelengths')
        return rvshifts, defaults, defaults
    elif np.any(diff==0):
        print('W contains two consecutive bins with the same wavelengths. Behavior unkonwn.')
        return rvshifts, defaults, defaults
    
    ## First make sure there are no NaNs in the _XW and _XF arrays
    # idx = np.where(~np.isnan(_XF_full))
    _XW = _XW_full#[idx]
    _XF = _XF_full#[idx]

    ## Only use non-NaNs points
    # idx = np.where(~np.isnan(_Y_full))
    _WW = _W_full#[idx] ## select valid points
    _YY = _Y_full#[idx] ## select valid points
    
    ## Remove the median of all:
    if renorm:
        _XF = _XF - np.median(_XF)
        _YY = _YY - np.median(_YY)
    
    if len(W.shape)>1:
        print("ccf: Too many dimensions for W. Don't know what to do.")
        return rvshifts, defaults, defaults
    
    ## Remove the lines in the mask if they are close of the
    ## limit, i.e. they will be out of bounds after doppler shift.
    lowlim = np.min(_WW); uplim = np.max(_WW)
    ## Need to define limits so that interpolation is not out of range
    lowdop_a = lowlim*doppler(np.max(rvshifts)) ## lowest acceptable wavelength
    updop_a = uplim*doppler(np.min(rvshifts)) ## largest acceptable wavelength
    ## Take only the valid part of the mask
    # idx = (_XW>lowdop_a) & (_XW<updop_a)
    _XW_val = _XW#[idx]; 
    _XF_val = _XF#[idx]
    ## Search for the gaps in the spectrum:
    diff = np.diff(_WW)/((_WW[:-1]+_WW[1:])/2)*cvel ## delta_lambda / lambda * c (km/s)
    lowbounds = np.where(diff > step_ths)[0] ## Where should we cut the spectrum ?
    highbounds = lowbounds + 1
    ## lowbounds and highbounds define the regions to avoid
    bounds = [_WW[lowbounds], _WW[highbounds]]
    nbbounds = len(_WW[lowbounds])
    # boundslist = list(zip(*bounds))
    # print(boundslist)
    ## Now we go through the bounds and search for the regions to avoid
    # cond = np.full(len(_XW), False, dtype=bool)
    cond = np.ones(len(_XW))==0
    for ib in range(nbbounds):
        # bounds = boundslist[ib]
        lowdop = bounds[0][ib]*doppler(np.min(rvshifts)) ## lowest acceptable wavelength
        updop = bounds[1][ib]*doppler(np.max(rvshifts)) ## largest acceptable wavelength
        ## Take only the valid part of the mask
        cond = cond | ((_XW>lowdop) & (_XW<updop))
    cond = ~cond ## This is the points to KEEP
    _XW = _XW[cond]; _XF = _XF[cond]

    #### ------------------------------------- ####
    #### ---- Perform CCF on valid points ---- ####
    #### ------------------------------------- ####
    #
    corrp  = np.zeros(len(rvshifts)) ## Numerator
    div    = np.empty(len(rvshifts)) ## Denominator
    #
    # ## Ensure that we are not running on too low points:
    # if (function=='cubic') & (len(_WW)<5):
    #     return  rvshifts, defaults, defaults

    # fun = interp1d(_WW, _YY, function) ## Magic interpolation of the spectrum
    #
    nbins = len(_XF_val)
    for i in range(len(rvshifts)):
        ## Here I use -1 because it's simpler than modifying the min and max there.
        ## That simply comes from the way we build the rvshifts arary...
        dopshift = doppler(-rvshifts[i])
        ## Get the interpolated spectrum
        _new_XW = _XW_val*dopshift
        _YY_shift = np.interp(_new_XW, _WW, _YY)
        ## Normalize the mask (not necessarily useful...)
        # _XF_val = _XF_val/np.sum(_XF_val)
        ## Compute the CCF
        _corrp = 0.
        _div1 = 0.
        _div2 = 0.
        for j in range(nbins):
            if (_XW_val[j]>lowdop_a) & (_XW_val[j]<updop_a):
                if (_YY_shift[j]!=np.nan) & (_XF_val[j]!=np.nan):
                    _corrp+=_XF_val[j]*_YY_shift[j]
                    _div1+=_XF_val[j]**2
                    _div2+=_YY_shift[j]**2
                else:
                    print(_YY_shift[j], _XF_val[j])
        _div = np.sqrt(_div1*_div2)
        # _corrp = np.sum(_XF_val*_YY_shift) ## Sum the values
        # _div   = np.sqrt(np.sum((_XF_val)**2)*np.sum(_YY_shift**2)) # div to norm.
        ## Avoid zero
        if _div==0:
            _div = 1 ## Avoid division by zero.
        ## Apprend the results
        corrp[i] = _corrp
        div[i]   = _div
    return rvshifts, corrp, div

@jit(nopython=True, cache=True)
def ccf_jit(XW, XF, W, Y, span=100, step=2, step_ths=5, plot=False, function='cubic', renorm=True):
    '''
    SAME AS CCF but Numba compatible

    return what is needed for the ccf CCF. Note that the normalized CCF is corrp/div.
    What I call line mask is a list of intensities associated to a list of wavelengths.
    The CCF will be computed on a -span/2 +span/2 window, in steps of step km/s.
    The step_ths is used to find gaps to avoid. For example, by default, I do not interpolate
    between two pixels that are more than 5 km/s apart.
    Inputs:
    - XW        : wavelength of a line mask
    - XF        : intensity of each line in the mask
    - W         : wavelengths of the spectrum
    - Y         : spectrum
    - span      : velocity span of the CCF (in km/s)
    - step      : velocity step at which you want the CCF to be sampled (km/s)
    - step_ths  : velocity step used to split the input spectrum (km/s)
    - function  : scipy.interp1d function used to interpolate the spectrum. Default is cubic. 
    - renorm    : renormalize XF and Y by removing the median of XF and Y respectively. Default is True.
    '''
    #
    ## Define the RV shifts
    rvshifts = np.arange(1, span, step)
    rvshifts = rvshifts - span//2
    defaults = np.ones(len(rvshifts))
    
    _XW_full, _XF_full = np.copy(XW), np.copy(XF) ## Unpack wvl and intensity of mask
    _W_full, _Y_full = np.copy(W), np.copy(Y) ## Unpack wvl and intensity of mask
    cvel = 3*1e5

    ## Check
    if len(_W_full)!=len(_Y_full):
        print('Failed; shape(_W)!=shape(_Y)')
        return rvshifts, defaults, defaults
    
    ## Check that the input wavelength is increasing
    diff = np.diff(_W_full)
    if np.any(diff<0):
        print('Failed; W must be increasing wavelengths')
        return rvshifts, defaults, defaults
    elif np.any(diff==0):
        print('W contains two consecutive bins with the same wavelengths. Behavior unkonwn.')
        return rvshifts, defaults, defaults
    
    ## First make sure there are no NaNs in the _XW and _XF arrays
    idx = np.where(~np.isnan(_XF_full))
    _XW = _XW_full[idx]
    _XF = _XF_full[idx]

    ## Only use non-NaNs points
    idx = np.where(~np.isnan(_Y_full))
    _WW = _W_full[idx] ## select valid points
    _YY = _Y_full[idx] ## select valid points
    
    ## Remove the median of all:
    if renorm:
        _XF = _XF - np.median(_XF)
        _YY = _YY - np.median(_YY)
    
    if len(W.shape)>1:
        print("ccf: Too many dimensions for W. Don't know what to do.")
        return rvshifts, defaults, defaults
    
    ## Remove the lines in the mask if they are close of the
    ## limit, i.e. they will be out of bounds after doppler shift.
    lowlim = np.min(_WW); uplim = np.max(_WW)
    ## Need to define limits so that interpolation is not out of range
    lowdop = lowlim*doppler(np.max(rvshifts)) ## lowest acceptable wavelength
    updop = uplim*doppler(np.min(rvshifts)) ## largest acceptable wavelength
    ## Take only the valid part of the mask
    idx = (_XW>lowdop) & (_XW<updop)
    _XW_val = _XW[idx]; _XF_val = _XF[idx]
    ## Search for the gaps in the spectrum:
    diff = np.diff(_WW)/((_WW[:-1]+_WW[1:])/2)*cvel ## delta_lambda / lambda * c (km/s)
    lowbounds = np.where(diff > step_ths)[0] ## Where should we cut the spectrum ?
    highbounds = lowbounds + 1
    ## lowbounds and highbounds define the regions to avoid
    bounds = [_WW[lowbounds], _WW[highbounds]]
    nbbounds = len(_WW[lowbounds])
    # boundslist = list(zip(*bounds))
    # print(boundslist)
    ## Now we go through the bounds and search for the regions to avoid
    # cond = np.full(len(_XW), False, dtype=bool)
    cond = np.ones(len(_XW))==0
    for ib in range(nbbounds):
        # bounds = boundslist[ib]
        lowdop = bounds[0][ib]*doppler(np.min(rvshifts)) ## lowest acceptable wavelength
        updop = bounds[1][ib]*doppler(np.max(rvshifts)) ## largest acceptable wavelength
        ## Take only the valid part of the mask
        cond = cond | ((_XW>lowdop) & (_XW<updop))
    cond = ~cond ## This is the points to KEEP
    _XW = _XW[cond]; _XF = _XF[cond]

    #### ------------------------------------- ####
    #### ---- Perform CCF on valid points ---- ####
    #### ------------------------------------- ####
    #
    corrp  = np.zeros(len(rvshifts)) ## Numerator
    div    = np.empty(len(rvshifts)) ## Denominator
    #
    # ## Ensure that we are not running on too low points:
    # if (function=='cubic') & (len(_WW)<5):
    #     return  rvshifts, defaults, defaults

    # fun = interp1d(_WW, _YY, function) ## Magic interpolation of the spectrum
    #
    for i in range(len(rvshifts)):
        ## Here I use -1 because it's simpler than modifying the min and max there.
        ## That simply comes from the way we build the rvshifts arary...
        dopshift = doppler(-rvshifts[i])
        ## Get the interpolated spectrum
        _YY_shift = np.interp(_XW_val*dopshift, _WW, _YY)
        ## Normalize the mask (not necessarily useful...)
        # _XF_val = _XF_val/np.sum(_XF_val)
        ## Compute the CCF
        _corrp = np.sum(_XF_val*_YY_shift) ## Sum the values
        _div   = np.sqrt(np.sum((_XF_val)**2)*np.sum(_YY_shift**2)) # div to norm.
        ## Avoid zero
        if _div==0:
            _div = 1 ## Avoid division by zero.
        ## Apprend the results
        corrp[i] = _corrp
        div[i]   = _div

    return rvshifts, corrp, div

@jit(nopython=True, cache=True)
def ccf_2d_jit(XW, XF, W, Y, span=100, step=2, step_ths=5, function='cubic', plot=False, renorm=True):
    '''Same as ccf but for a 2d spectrum.'''
    # if len(W.shape)==2:
    #     mode2d = True
    #     # print('Entering 2D mode.')
    # elif len(W.shape)>2:
    #     raise Exception("Too many dimensions for W. Don't know what to do.")
    
    mask2d = False
    if len(np.shape(XW))==2:
        mask2d = True

    # rvshifts = 0
    x = np.arange(1, span, step)
    # rvshifts = rvshifts - span//2
    #
    corrp = np.zeros(len(x))
    div = np.ones(len(x))
    for order in range(len(W)):
        _W = W[order]; _Y = Y[order]
        if np.all(np.isnan(_Y)):
            continue ## All NaN slice
        rvshifts, _corrp, _div = ccf_jit(XW, XF, _W, _Y, span=span, step=step, step_ths=step_ths, function=function, plot=plot, renorm=renorm)
        corrp+=_corrp
        div+=_div

    corrp_sum = corrp
    div_sum = div

    return rvshifts, corrp_sum, div_sum

# @jit(nopython=False, cache=True)
def ccf(XW, XF, W, Y, span=100, step=2, step_ths=5, plot=False, function='cubic', renorm=True):
    '''
    return what is needed for the ccf CCF. Note that the normalized CCF is corrp/div.
    What I call line mask is a list of intensities associated to a list of wavelengths.
    The CCF will be computed on a -span/2 +span/2 window, in steps of step km/s.
    The step_ths is used to find gaps to avoid. For example, by default, I do not interpolate
    between two pixels that are more than 5 km/s apart.
    Inputs:
    - XW        : wavelength of a line mask
    - XF        : intensity of each line in the mask
    - W         : wavelengths of the spectrum
    - Y         : spectrum
    - span      : velocity span of the CCF (in km/s)
    - step      : velocity step at which you want the CCF to be sampled (km/s)
    - step_ths  : velocity step used to split the input spectrum (km/s)
    - function  : scipy.interp1d function used to interpolate the spectrum. Default is cubic. 
    - renorm    : renormalize XF and Y by removing the median of XF and Y respectively. Default is True.
    '''
    #
    _XW, _XF = np.copy(XW), np.copy(XF) ## Unpack wvl and intensity of mask
    _W, _Y = np.copy(W), np.copy(Y) ## Unpack wvl and intensity of mask
    cvel = 3*1e5

    ## Check
    if len(_W)!=len(_Y):
        raise Exception('W and Y must have the same dimension')

    ## Check that the input wavelength is increasing
    diff = np.diff(_W)
    if np.any(diff<0):
        raise Exception('W must be increasing wavelengths')
    elif np.any(diff==0):
        raise Exception('W contains two consecutive bins with the same wavelengths. Behavior unkonwn.')

    ## First make sure there are no NaNs in the _XW and _XF arrays
    idx = np.where(~np.isnan(_XF))
    _XW = _XW[idx]
    _XF = _XF[idx]

    ## Only use non-NaNs points
    idx = np.where(~np.isnan(_Y))
    _WW = _W[idx] ## select valid points
    _YY = _Y[idx] ## select valid points
    
    ## Remove the median of all:
    if renorm:
        _XF = _XF - np.median(_XF)
        _YY = _YY - np.median(_YY)
    
    if len(W.shape)>1:
        raise Exception("ccf: Too many dimensions for W. Don't know what to do.")

    ## Define the RV shifts
    rvshifts = np.arange(1, span, step)
    rvshifts = rvshifts - span//2

    ## Remove the lines in the mask if they are close of the
    ## limit, i.e. they will be out of bounds after doppler shift.
    lowlim = np.min(_WW); uplim = np.max(_WW)
    ## Need to define limits so that interpolation is not out of range
    lowdop = lowlim*doppler(np.max(rvshifts)) ## lowest acceptable wavelength
    updop = uplim*doppler(np.min(rvshifts)) ## largest acceptable wavelength
    ## Take only the valid part of the mask
    idx = (_XW>lowdop) & (_XW<updop)
    _XW = _XW[idx]; _XF = _XF[idx]
    ## Search for the gaps in the spectrum:
    diff = np.diff(_WW)/((_WW[:-1]+_WW[1:])/2)*cvel ## delta_lambda / lambda * c (km/s)
    lowbounds = np.where(diff > step_ths)[0] ## Where should we cut the spectrum ?
    highbounds = lowbounds + 1
    ## lowbounds and highbounds define the regions to avoid
    bounds = [_WW[lowbounds], _WW[highbounds]]
    boundslist = list(zip(*bounds))
    ## Now we go through the bounds and search for the regions to avoid
    cond = np.full(len(_XW), False, dtype=bool)
    for bounds in boundslist:
        lowdop = bounds[0]*doppler(np.min(rvshifts)) ## lowest acceptable wavelength
        updop = bounds[1]*doppler(np.max(rvshifts)) ## largest acceptable wavelength
        ## Take only the valid part of the mask
        cond = cond | ((_XW>lowdop) & (_XW<updop))
    cond = ~cond ## This is the points to KEEP
    _XW = _XW[cond]; _XF = _XF[cond]

    #### ------------------------------------- ####
    #### ---- Perform CCF on valid points ---- ####
    #### ------------------------------------- ####
    #
    corrp  = np.zeros(len(rvshifts)) ## Numerator
    div    = np.empty(len(rvshifts)) ## Denominator
    #
    ## Ensure that we are not running on too low points:
    if (function=='cubic') & (len(_WW)<5):
        return  rvshifts, rvshifts*0, rvshifts*0

    fun = interp1d(_WW, _YY, function) ## Magic interpolation of the spectrum
    #
    for i in range(len(rvshifts)):
        ## Here I use -1 because it's simpler than modifying the min and max there.
        ## That simply comes from the way we build the rvshifts arary...
        dopshift = doppler(-rvshifts[i])
        ## Get the interpolated spectrum
        _YY_shift = fun(_XW*dopshift)
        ## Normalize the mask (not necessarily useful...)
        # _XF = _XF/np.sum(_XF)
        ## Compute the CCF
        _corrp = np.sum(_XF*_YY_shift) ## Sum the values
        _div   = np.sqrt(np.sum((_XF)**2)*np.sum(_YY_shift**2)) # div to norm.
        ## Avoid zero
        if _div==0:
            _div = 1 ## Avoid division by zero.
        ## Apprend the results
        corrp[i] = _corrp
        div[i]   = _div
        # Control plot if requested
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            # plt.plot(wvl, flx, color='black')
            plt.plot(_WW, _YY/np.max(_YY), color='black')
            plt.bar(_XW*dopshift, _XF, color='red', width = 0.1)
            # plt.plot(_XW*dopshift, _XF, color='red')
            print(_XW*dopshift)
            for bounds in boundslist:
                plt.axvspan(bounds[0], bounds[1])
            plt.title('{}/{}'.format(i, len(rvshifts)))
            plt.show()

    return rvshifts, corrp, div

def ccf_2d(XW, XF, W, Y, span=100, step=2, step_ths=5, function='cubic', plot=False, renorm=True):
    '''Same as ccf but for a 2d spectrum.'''
    if len(W.shape)==2:
        mode2d = True
        # print('Entering 2D mode.')
    elif len(W.shape)>2:
        raise Exception("Too many dimensions for W. Don't know what to do.")
    

    mask2d = False
    if len(np.shape(XW))==2:
        mask2d = True

    # rvshifts = 0
    corrp = []
    div = []
    for order in range(len(W)):
        _W = W[order]; _Y = Y[order]
        if mask2d:
            _XW, _XF = XW[order], XF[order]
        else:
            _XW, _XF = XW, XF
        if np.all(np.isnan(_Y)):
            continue ## All NaN slice
        rvshifts, _corrp, _div = ccf(_XW, _XF, _W, _Y, span=span, step=step, step_ths=step_ths, function=function, plot=plot, renorm=renorm)
        corrp.append(_corrp)
        div.append(_div)

    corrp = np.sum(corrp, 0)
    div = np.sum(div, 0)

    return rvshifts, corrp, div
        
def test_ccf():
    from astropy.io import fits
    import matplotlib.pyplot as plt
    file = '/Users/pcristofari/Data/spirou/lam-data-templates-v7275/gj1012_templates.fits'
    file = '/Users/pcristofari/Data/spirou/lam-data-templates-v7275/citau_templates.fits'
    # file = '/Users/pcristofari/Data/spirou/lam-data-templates-v7254/gl699_templates.fits'
    hdu = fits.open(file)
    wvl = np.array(hdu['WVL'].data, dtype=float)
    flx = np.array(hdu['TEMPLATE'].data, dtype=float)
    hdu.close()


    w, f = msk_tls.read_vald_lines() ## VALD line list

    order = 46

    plt.figure()
    plt.plot(wvl[order], flx[order])
    plt.show()

    lims=[2200, 2400]

    _wvl, _flx = wvl[order], flx[order]
    idx = (_wvl>lims[0])&(_wvl<lims[1])
    _wvl = _wvl[idx]
    _flx = _flx[idx]

    rvshifts0, corrp0, div0 = ccf(w, f, _wvl, _flx-1, step=1, step_ths=20, span=100, plot=False)

    plt.figure()
    # plt.plot(rvshifts0, corrp0, color='black')
    # plt.plot(rvshifts0, div0, color='red')
    plt.plot(rvshifts0, corrp0/div0, color='black')
    # plt.plot(rvshifts0_lin, corrp0_lin/div0_lin, color='red')
    # plt.plot(rvshifts1, corrp1/div1, color='blue')
    # plt.axvline(rvshifts0[corrp0/div0==np.min(corrp0/div0)], color='black')
    # plt.axvline(rvshifts0_lin[corrp0_lin/div0_lin==np.min(corrp0_lin/div0_lin)], color='red', linestyle='--')
    # plt.axhline((corrp0/div0)[corrp0/div0==np.min(corrp0/div0)], color='black')
    # plt.axhline((corrp0_lin/div0_lin)[corrp0_lin/div0_lin==np.min(corrp0_lin/div0_lin)], color='red', linestyle='--')
    # plt.axvline(rvshifts[corrp1==np.min(corrp1)], color='blue', linestyle='--')
    # plt.plot(rvshifts, corrp/div, color='red')
    plt.show()


# test_ccf()