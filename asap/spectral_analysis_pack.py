
from asap import params
import numpy as np
from numba import jit
from asap import analysis_tools as tls
from asap import effects as effects
from asap.c_tools import effects as effects_cy
from asap import normalization_tools as norm_tools
from asap.c_tools import normalization_tools as norm_tools_cy
from asap import polyfit
from numba import types
from numba.typed import Dict

@jit(nopython=params.JIT, cache=params.CACHE)
def interp_axis_0(val, x1, x2, y1, y2, function='linear'):
    '''Function does a LINEAR interpolation between y1 and y2. Interpolation
    evaluated for value val.
    Implementation in pure Python extremly fast.'''
    # if function=='log10':
    #     a = (np.log10(y2)-np.log10(y1))/(x2-x1)
    #     b = np.log10(y1) - a*x1
    #     o = 10**(a*val + b)
    #     o[np.isnan(o)] = 0
    #     return o
    # else:
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a*val + b


@jit(nopython=params.JIT, cache=params.CACHE)
def wrap_function_fine_linear_4d(teff,logg, mh, alpha, teffs, loggs, mhs, alphas, spectra, function='linear'):
    '''This very function performs really fast linear interpolation in 4D.
    It chooses the points around the desired interpolation value, and performs
    linear interpolation on each dimension.
    !!! This function allows to extrapolate to on the lowest end of the grid for Logg and [M/H] only !. '''

    ## Function
    function = function.lower().strip()

    # teff,logg, mh, teffs, loggs, mhs, spectra = arguments
    ## What are the closest coeffs?
    if teff<teffs[0]: ## Allow extrapolation on the fly
        idx_tlow = 0
        idx_thigh = 1
    else:
        idx_tlow = np.where(teffs <= teff)[0][-1]
        idx_thigh = np.where(teffs >= teff)[0][0]
#
    if logg<loggs[0]: ## Allow extrapolation on the fly
        idx_llow = 0
        idx_lhigh = 1
    else:
        idx_llow = np.where(loggs <= logg)[0][-1]
        idx_lhigh = np.where(loggs >= logg)[0][0]
#
    # idx_llow = np.where(loggs <= logg)[0][-1]
    # idx_lhigh = np.where(loggs >= logg)[0][0]
    idx_mlow = np.where(mhs <= mh)[0][-1]
    idx_mhigh = np.where(mhs >= mh)[0][0]
    if alpha<alphas[0]: ## This should allow to extrapolate on the fly
        idx_alow = 0
        idx_alow = 1
    else:
        idx_alow = np.where(alphas <= alpha)[0][-1]
        idx_ahigh = np.where(alphas >= alpha)[0][0]
    tlow = teffs[idx_tlow] 
    thigh = teffs[idx_thigh] 
    llow = loggs[idx_llow] 
    lhigh = loggs[idx_lhigh] 
    mlow = mhs[idx_mlow] 
    mhigh = mhs[idx_mhigh] 
    alow = alphas[idx_alow] 
    ahigh = alphas[idx_ahigh]
    ## We need the spectra that are around the said spectrum
    if function=='log10':
        sllll = np.log10(spectra[idx_tlow, idx_llow, idx_mlow, idx_alow])
    elif (function=='log') | (function=='ln'):
        sllll = np.log(spectra[idx_tlow, idx_llow, idx_mlow, idx_alow])
    else:
        sllll = spectra[idx_tlow, idx_llow, idx_mlow, idx_alow]
    if function=='log10':
        slllh = np.log10(spectra[idx_tlow, idx_llow, idx_mlow, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        slllh = np.log(spectra[idx_tlow, idx_llow, idx_mlow, idx_ahigh])
    else:
        slllh = spectra[idx_tlow, idx_llow, idx_mlow, idx_ahigh]
    if function=='log10':
        sllhl = np.log10(spectra[idx_tlow, idx_llow, idx_mhigh, idx_alow])
    elif (function=='log') | (function=='ln'):
        sllhl = np.log(spectra[idx_tlow, idx_llow, idx_mhigh, idx_alow])
    else:
        sllhl = spectra[idx_tlow, idx_llow, idx_mhigh, idx_alow]
    if function=='log10':
        sllhh = np.log10(spectra[idx_tlow, idx_llow, idx_mhigh, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        sllhh = np.log(spectra[idx_tlow, idx_llow, idx_mhigh, idx_ahigh])
    else:
        sllhh = spectra[idx_tlow, idx_llow, idx_mhigh, idx_ahigh]
    if function=='log10':
        slhll = np.log10(spectra[idx_tlow, idx_lhigh, idx_mlow, idx_alow])
    elif (function=='log') | (function=='ln'):
        slhll = np.log(spectra[idx_tlow, idx_lhigh, idx_mlow, idx_alow])
    else:
        slhll = spectra[idx_tlow, idx_lhigh, idx_mlow, idx_alow]
    if function=='log10':
        slhlh = np.log10(spectra[idx_tlow, idx_lhigh, idx_mlow, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        slhlh = np.log(spectra[idx_tlow, idx_lhigh, idx_mlow, idx_ahigh])
    else:
        slhlh = spectra[idx_tlow, idx_lhigh, idx_mlow, idx_ahigh]
    if function=='log10':
        shlll = np.log10(spectra[idx_thigh, idx_llow, idx_mlow, idx_alow])
    elif (function=='log') | (function=='ln'):
        shlll = np.log(spectra[idx_thigh, idx_llow, idx_mlow, idx_alow])
    else:
        shlll = spectra[idx_thigh, idx_llow, idx_mlow, idx_alow]
    if function=='log10':
        shllh = np.log10(spectra[idx_thigh, idx_llow, idx_mlow, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        shllh = np.log(spectra[idx_thigh, idx_llow, idx_mlow, idx_ahigh])
    else:
        shllh = spectra[idx_thigh, idx_llow, idx_mlow, idx_ahigh]
    if function=='log10':
        slhhl = np.log10(spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_alow])
    elif (function=='log') | (function=='ln'):
        slhhl = np.log(spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_alow])
    else:
        slhhl = spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_alow]
    if function=='log10':
        slhhh = np.log10(spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        slhhh = np.log(spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_ahigh])
    else:
        slhhh = spectra[idx_tlow, idx_lhigh, idx_mhigh, idx_ahigh]
    if function=='log10':
        shhll = np.log10(spectra[idx_thigh, idx_lhigh, idx_mlow, idx_alow])
    elif (function=='log') | (function=='ln'):
        shhll = np.log(spectra[idx_thigh, idx_lhigh, idx_mlow, idx_alow])
    else:
        shhll = spectra[idx_thigh, idx_lhigh, idx_mlow, idx_alow]
    if function=='log10':
        shhlh = np.log10(spectra[idx_thigh, idx_lhigh, idx_mlow, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        shhlh = np.log(spectra[idx_thigh, idx_lhigh, idx_mlow, idx_ahigh])
    else:
        shhlh = spectra[idx_thigh, idx_lhigh, idx_mlow, idx_ahigh]
    if function=='log10':
        shlhl = np.log10(spectra[idx_thigh, idx_llow, idx_mhigh, idx_alow])
    elif (function=='log') | (function=='ln'):
        shlhl = np.log(spectra[idx_thigh, idx_llow, idx_mhigh, idx_alow])
    else:
        shlhl = spectra[idx_thigh, idx_llow, idx_mhigh, idx_alow]
    if function=='log10':
        shlhh = np.log10(spectra[idx_thigh, idx_llow, idx_mhigh, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        shlhh = np.log(spectra[idx_thigh, idx_llow, idx_mhigh, idx_ahigh])
    else:
        shlhh = spectra[idx_thigh, idx_llow, idx_mhigh, idx_ahigh]
    if function=='log10':
        shhhl = np.log10(spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_alow])
    elif (function=='log') | (function=='ln'):
        shhhl = np.log(spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_alow])
    else:
        shhhl = spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_alow]
    if function=='log10':
        shhhh = np.log10(spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_ahigh])
    elif (function=='log') | (function=='ln'):
        shhhh = np.log(spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_ahigh])
    else:
        shhhh = spectra[idx_thigh, idx_lhigh, idx_mhigh, idx_ahigh]
    ## First we reduce the problem to a 3D problem:
    # Let's interpolate in the alpha direction
    # We must have 

    _new_spectra = np.empty(spectra[0,0,0,0].shape)
    ## And we interpolate the spectrum
    for r in range(len(sllll)):
        if alow==ahigh:
            slll = sllll[r]
            sllh = sllhl[r]
            slhl = slhll[r]
            shll = shlll[r]
            shhl = shhll[r]
            slhh = slhhl[r]
            shlh = shlhl[r]
            shhh = shhhl[r]
        else:
            slll = interp_axis_0(alpha, alow, ahigh, sllll[r], slllh[r], function)
            sllh = interp_axis_0(alpha, alow, ahigh, sllhl[r], sllhh[r], function)
            slhl = interp_axis_0(alpha, alow, ahigh, slhll[r], slhlh[r], function)
            shll = interp_axis_0(alpha, alow, ahigh, shlll[r], shllh[r], function)
            shhl = interp_axis_0(alpha, alow, ahigh, shhll[r], shhlh[r], function)
            slhh = interp_axis_0(alpha, alow, ahigh, slhhl[r], slhhh[r], function)
            shlh = interp_axis_0(alpha, alow, ahigh, shlhl[r], shlhh[r], function)
            shhh = interp_axis_0(alpha, alow, ahigh, shhhl[r], shhhh[r], function)
        if tlow == thigh:
            s1 = slll
            s2 = slhl
            s3 = sllh
            s4 = slhh
        else:
            s1 = interp_axis_0(teff, tlow, thigh, slll, shll, function)
            s2 = interp_axis_0(teff, tlow, thigh, slhl, shhl, function)
            s3 = interp_axis_0(teff, tlow, thigh, sllh, shlh, function)
            s4 = interp_axis_0(teff, tlow, thigh, slhh, shhh, function)
        if llow == lhigh:
            s11 = s1
            s33 = s3
        else:
            s11 = interp_axis_0(logg, llow, lhigh, s1, s2, function)
            s33 = interp_axis_0(logg, llow, lhigh, s3, s4, function)
        if mlow == mhigh:
            s = s11
        else:
            s = interp_axis_0(mh, mlow, mhigh, s11, s33, function)
        _new_spectra[r] = s
    
    ## If function is log, we need to adjust the output solution
    if function=='log10':
        _new_spectra = 10**(_new_spectra)
    elif (function=='log') | (function=='ln'):
        _new_spectra = np.exp(_new_spectra)
    
    return (teff, logg, mh), _new_spectra

def broaden_spectra(args, **kwargs):
    '''
    Function to broaden spectra regions.

    Input parameters:
    /! MUST BE A LIST OF ARGUMENTS CONTAINING:
    - wvls      :   Wavelength solution for the model grid
    - spectrum  :   Spectrum to broaden and shift
    - obs_wvl   :   Wavelength grid for the observation spectrum
    - obs_flux  :   Observation spectrum
    - vinstru   :   Instrumental width
    - vmac      :   Macroturbulence (currently assumed gaussian)
    - vsini     :   Rotation velocity
    - vrad      :   Radial velocity value
    - t, l, m   :   Indices used to order outputs after multiprocessing
    '''
    if len(args)>16:
        vrad, wvls, spectrum, obs_wvl, obs_flux, obs_err, mask, \
            vinstru, vmac, vsini, t, l, m, model, adj, function, epsilon = args
    else:
        vrad, wvls, spectrum, obs_wvl, obs_flux, obs_err, mask, \
                vinstru, vmac, vsini, t, l, m, model, adj, function = args
        epsilon = 0.6

    if "macProf" in kwargs.keys(): 
        macProf = kwargs['macProf']
    else:
        macProf = 'g'

    if "payneWaveIdx" in kwargs.keys(): 
        payneWaveIdx = kwargs['payneWaveIdx']
    else:
        payneWaveIdx = None

    nan_mask = np.copy(mask)
    nan_mask[nan_mask==0] = np.nan
    r = 0 # Dummy solution to estimate wavelength step

    # print(vrad)
    # wvls = spirou.correct_berv(wvls, -vrad)
    doppler_factor = tls.doppler(-vrad)

    output = []
    ps = []
    ps2 = []
    cs = []
    cs2 = []
    coeffs = []
    coeffserr = []
    for r in range(len(obs_wvl)):
        if np.any(np.isnan(obs_flux[r])):
            # raise Exception('NaN value in observed flux')
            pass
        elif np.any(np.isnan(spectrum[r])):
            raise Exception('NaN value in model')
        if np.any(np.isnan(wvls[r])):
            raise Exception('NaN value in wvls')
        # if model=='turbospectrum' or model=='turbospectrum_vmic0.3':
        #     _spectra = convolve.convolve(wvls[r], spectrum[r], -np.sqrt(vmac**2))
        # else:
        if payneWaveIdx is not None:
            _wvls = wvls[payneWaveIdx[r]] * doppler_factor
        elif wvls.ndim>1:
            _wvls = wvls[r] * doppler_factor
        else:
            _wvls = wvls * doppler_factor
        # _spectra = effects.broadened_profile(
        #     _wvls, spectrum[r], 
        #     rv=None, epsilon=0.6,
        #     vsini=vsini, vmac=vmac, 
        #     vinstru = vinstru
        #     )
        # _spectra = effects.broaden_spectrum_2(_wvls, spectrum[r], 
        #                                     vinstru=vinstru, 
        #                                     vsini=vsini, epsilon=0.6, 
        #                                     vmac=vmac, vmac_mode=macProf)
        ## Faster with cython:
        if payneWaveIdx is not None:
            _spec = spectrum[payneWaveIdx[r]]
        elif spectrum.ndim>1:
            _spec = spectrum[r]
        else:
            _spec = spectrum
        if ((vinstru==0.) & (vsini==0.)) & (vmac==0.):
            _spectra = _spec
        else:
            _spectra = effects_cy.broaden_spectrum_2_cy(_wvls, _spec, 
                                                vinstru=vinstru, 
                                                vsini=vsini, epsilon=epsilon, 
                                                vmac=vmac, vmac_mode=macProf)      
        _wlim = _wvls[_spec!=0][-1]
        _wlow = _wvls[0]
        
        _wvl = _wvls
        _flux = _spectra
        # _spectrum = inte.fftintegrate(obs_wvl[r], _wvl, _flux)
        _spectrum = np.interp(obs_wvl[r], _wvl, _flux)
        _spectrum[obs_wvl[r]>_wlim] = 0.
        _spectrum[obs_wvl[r]<_wlow] = 0.

        # _spectrum = inte.integrate(obs_wvl[r], _wvl, _flux)
        # Adjust normalization 
        # _c, _ps = norm_tools.numba_fit_continuum_2(
        #     wvl=obs_wvl[r],
        #     flux=obs_flux[r],
        #     window_size=100,
        #     p=50,
        #     degree=1, m=0.05)
        # _c2, _ps2 = norm_tools.numba_fit_continuum_2(
        #     wvl=obs_wvl[r],
        #     flux=_spectrum,
        #     window_size=100,
        #     p=50,
        #     degree=1, m=0.05)
        # _spectrum = _spectrum/(_c2/_c)
        if adj:
            # try:
            ## Get the percentile from the config
            # config = read_config()
            # p = int(float(config['OPTIONS']['P']))
            # try:

            _c, _pss, X, Xerr, X2, X2err = norm_tools.adjust_continuum5(wvl=obs_wvl[r],
                                                obs_flux=obs_flux[r],
                                                model_flux=_spectrum,
                                                window_size=100,
                                                p=90,
                                                degree=1, m=0.05, function=function)
            ## Implementation in Cython, faster and possibly better.
            # _c, wave_points, obs_points, mod_points = norm_tools_cy.adjust_continuum5_fast_inplace(wvl=obs_wvl[r],
            #                                     obs_flux=obs_flux[r],
            #                                     model_flux=_spectrum,
            #                                     p=90,
            #                                     nWindows = 6)
            
            _pss = [0,0,0,0]
            X = np.array([0,0])
            Xerr = np.array([0,0])
            X2 = np.array([0,0])
            X2err = np.array([0,0])

            if len(X)==1:
                X = np.array([0, X[0]])
                Xerr = np.array([0, Xerr[0]])
        else:
            _c = np.ones(_spectrum.shape)
            _pss = [0,0,0,0]
            X = np.array([0,0])
            Xerr = np.array([0,0])
            X2 = np.array([0,0])
            X2err = np.array([0,0])
        # _ps = 0
        # _c = np.ones(len(_spectrum))

        ## chi2 adjustement of continuum under development
        # wargs = (wvls, spectrum, obs_wvl, obs_flux, obs_err, mask,\
        #                     vinstru, vmac, vsini, t, l, m, model, adj, function)
        # result_ls = opt.least_squares(compute_residuals, x0=initial_guess, 
        #                             method='lm', args = wargs)#, xtol=0.1, ftol=1e-15, gtol=1e-15)

        ## Continuum adjustment via least square fit.
        # _c, _ps = norm_tools.adjust_continuum(wvl=obs_wvl[r],
        #                                       obs_flux=obs_flux[r],
        #                                       model_flux=_spectrum,
        #                                       window_size=100,
        #                                       p=60,
        #                                       degree=1, m=0.05)

        _ps = [_pss[0], _pss[1]]
        _ps2 = [_pss[2], _pss[3]]
        _c2 = _c
        _spectrum = _spectrum / _c

        # # Correct the continuum to get closer to min chi2
        # _nan_mask = nan_mask[r]
        # corr = opt_cont(_spectrum, obs_flux[r]*nan_mask[r], obs_err[r])
        # # print(corr)
        # _spectrum = _spectrum * corr

        output.append(_spectrum)
        ps.append(_ps)
        ps2.append(_ps2)
        cs.append(_c)
        cs2.append(_c2)
        coeffs.append([X, X2])
        coeffserr.append([Xerr, X2err])

    output = np.array(output)
    cs = np.array(cs)
    return t, l, m, output, vrad, [ps, ps2], [cs, cs2], coeffs, coeffserr

# @jit(nopython=True, cache=True)
def veiling_function(veilingParams, wvl, veilingBands, mode='interp_between'):
    '''This function takes a number of veiling for the YJHK bands and return the value of the veiling for the considered
    wavaelength ranges.
    This version interpolates LIEARLY through the points'''

    # if len(veilingParams)>4:
    #     raise Exception("veilingParams can only work for YJHK at the moment.\nMust provide exactly 6 veiling coeffs.")
    
    ## Initial check:
    strVeilingBands = veilingBands.strip().replace(' ', '').upper()
    nbRequestedBands = len(strVeilingBands) ## one letter per band
    nbPassedParams = len(veilingParams)
    if nbRequestedBands!=nbPassedParams:
        pass
        # raise Exception('veiling_function: mismatch between passed parameters and number of requested bands.')

    ## Data contaning the central wavelength of each band 

    wavelengths_dic = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )

    fwhms_dic = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64,
    )
    wavelengths_dic['I'] = 8000
    wavelengths_dic['Y'] = 10200
    wavelengths_dic['J'] = 12200
    wavelengths_dic['H'] = 16300
    wavelengths_dic['K'] = 21900
    wavelengths_dic['L'] = 30000
    #
    fwhms_dic['I'] = 1490
    fwhms_dic['Y'] = 1200
    fwhms_dic['J'] = 2130
    fwhms_dic['H'] = 3070
    fwhms_dic['K'] = 3900
    fwhms_dic['L'] = 4720
    
    listOfBands = ['I','Y','J','H','K','L']


    # wavelengths_dic = {'I': 8000, 'Y': 10200, 'J':12200, 'H':16300, 'K': 21900, 'L':30000}
    # fwhms_dic = {'I': 1490, 'Y': 1200, 'J':2130, 'H':3070, 'K': 3900, 'L':4720}
    # nbOfAvailableBands = len(list(wavelengths_dic.keys()))

    # ## Second check:
    # for band in strVeilingBands:
    #     if band not in listOfBands: #wavelengths_dic.keys():
    #         pass
    #         # raise Exception('veiling_function: requested band not available yet.')

    if mode=='interp':
        ## Unpack the values of different bands.
        wavelengths = np.empty(nbRequestedBands)
        for ib, band in enumerate(strVeilingBands):
            wavelengths[ib] = wavelengths_dic[band]
        wavelengths = np.sort(wavelengths)

        ## Now, say that there are NaNs on the edge, that means we want those to equal the closest
        myVeilArray = np.copy(veilingParams)
        if np.isnan(myVeilArray[0]):
            myVeilArray[0] = myVeilArray[1]
        if np.isnan(myVeilArray[-1]):
            myVeilArray[-1] = myVeilArray[-2]

        ## Third check:
        if (np.min(wvl)<wavelengths[0]) | (np.max(wvl)>wavelengths[-1]):
            pass
            # raise Exception('veiling_function: Please provide bands that cover the full wavelength range')

        ## Old stuff
        ## Those are the wavelength of the bands, with edges of the SPIRou domain
        # wavelengths = np.array([8000, 10200, 12200, 16300, 21900, 30000]) ## Angstroms
        ## For now, I simply set the edges values to that of the closest bands 
        # myVeilArray = [veilingParams[0], veilingParams[1], veilingParams[2], veilingParams[3], veilingParams[4], veilingParams[5]]
        # myVeilArray = [veilingParams[0], veilingParams[0], veilingParams[1], veilingParams[2], veilingParams[3], veilingParams[3]]

        myveiling = np.interp(wvl, wavelengths, myVeilArray)
    elif mode=='interp_between':
        ## Now, say that there are NaNs on the edge, that means we want those to equal the closest
        myVeilArray_int = np.copy(veilingParams)
        if np.isnan(myVeilArray_int[0]):
            myVeilArray_int[0] = myVeilArray_int[1]
        if np.isnan(myVeilArray_int[-1]):
            myVeilArray_int[-1] = myVeilArray_int[-2]

        ## Unpack the values of different bands.
        wavelengths = np.empty(nbRequestedBands*2) ## Two values per band
        myVeilArray = np.empty(nbRequestedBands*2)
        iterator = 0
        for ib, band in enumerate(strVeilingBands):
            wavelengths[iterator] = wavelengths_dic[band]-fwhms_dic[band]/2
            myVeilArray[iterator] = myVeilArray_int[ib]
            iterator+=1
            wavelengths[iterator] = wavelengths_dic[band]+fwhms_dic[band]/2
            myVeilArray[iterator] = myVeilArray_int[ib]
            iterator+=1
        wavelengths = np.sort(wavelengths)

        ## Interpolate between the bands
        myveiling = np.interp(wvl, wavelengths, myVeilArray)


    
    # myveiling = np.empty(len(wvl))
    # for i in range(len(wvl)):
    #     myveiling[i] = np.interp(wvl[i], wavelengths, myVeilArray)

    return myveiling

def fill_nans_wavelength(med_wvl):
    ####################################
    if np.any(np.isnan(med_wvl)):
        ## Some people put NaNs in the wavelengths... don't ask.
        ## Here is a fix:
        STDTOL = 1e-6 ## Maximimum deviation from input wavelength allowed
        new_med_wvl = np.empty(med_wvl.shape)
        prev_high = 0 ## highest wvl of the previous order 
        poly_orders = []
        for r in range(len(med_wvl)):
            KEEPGOING = True
            ORDERMAX = len(med_wvl) ## Maximum allowed order
            x = np.arange(len(med_wvl[r]), dtype=float)
            idx = np.where(~np.isnan(med_wvl[r]))
            poly_order = 0
            lastTurn = False
            prev_val = np.inf
            while KEEPGOING:
                poly_order+=1
                if len(x[idx])==2:
                    poly_order = 1
                elif len(x[idx])<2:
                    poly_order = 0
                
                if len(x[idx])>20:
                    x = norm_tools.normalize_axis(x, x[idx])
                    coeffs = polyfit.fit_1d_polynomial(x[idx], 
                                                    np.array(med_wvl[r][idx], 
                                                                dtype=float),poly_order)
                    fit = polyfit.poly1d(x, coeffs)
                    new_med_wvl[r] = fit
                    prev_high = new_med_wvl[r][-1]
                    ## Check that the residuals are sufficiently low:
                    std = np.nanstd(med_wvl[r] - new_med_wvl[r])
                    if std>prev_val:
                        KEEPGOING = False
                    else:
                        prev_val = std
                else:
                    print('Caution, full NaN order')
                    ## In that case we have a problem.
                    ## We start from the last wavelength of the previous order
                    ## We end with the a default value of 25000 (like for SPIRou).
                    ## It shouldn't really matter because this is full of NaNs, that we are going to ignore.
                    next_low = 25000
                    new_med_wvl[r] = np.linspace(prev_high, next_low, len(new_med_wvl[r]), dtype=float)*np.nan
                    std = 1e-20 ## dummy value
                if lastTurn:
                    KEEPGOING = False
                if np.any(np.diff(new_med_wvl[r][idx])<0):
                    if lastTurn:
                        print('FATAL ERROR - non-increasing wavelength')
                        print('We should not be reaching this point...')
                        from IPython import embed;embed()
                        raise Exception('FATAL ERROR - non-increasing wavelength')
                    lastTurn = True
                    poly_order-=2
                elif std<STDTOL: KEEPGOING = False ## Convergence reached
                elif (((std>STDTOL) & (poly_order>=ORDERMAX))
                    | ((std>STDTOL) & lastTurn)):
                    print('ISSUE RECONSTRUCTING THE WAVELENGTHS')
                    # from IPython import embed; embed()
                    # import matplotlib.pyplot as plt
                    # plt.figure()
                    # plt.plot(new_med_wvl[r])
                    # plt.plot(med_wvl[r], '.')
                    # plt.show()
                    print(f'STD = {std}')
                    # raise Exception('Reconstructing wavelength solution: STD too high. Contact Author')
            poly_orders.append(poly_order)
    ########################################## 
    return new_med_wvl, poly_orders

# @jit(nopython=True)
def rebuilt_wavelength(wave):
    '''Function to rearange the wavelength solution.
    The function loops through bins and shifts the solution to insert NaNs
    in the middle of the wavelength solution.
    CAUTION: assumes that input wavelengths are increasing'''

    newwave = np.empty(len(wave)*10)*np.nan ## output array
    nmaxbins = len(newwave)
    rec_diff = np.nan ## considered delta_wvl at given step
    indices = []
    n=0
    for bin in range(len(wave)-1):
        ## Is the diff to come larger than the previous one?
        diff = (wave[bin+1]-wave[bin])
        update_rec_diff = True
        ## append to array
        if n>nmaxbins-1: break
        newwave[n] = wave[bin]
        indices.append(n)
        n+=1
        if np.isnan(diff):
            pass
        if diff<(.5*rec_diff): ## Problem
            ## if True means first step larger than typical
            ## The diff I consider is the previous one
            thediff = rec_diff
            ## Need to set typical to next step:
            rec_diff = diff
            ## Need to set diff to insert NaNs BEFORE the bin
            n-=1
            jump = int(np.floor(thediff/rec_diff))
            ## Need to insert NaNs
            n+=jump ## Move to next bin in output array
        elif diff>(1.5*rec_diff):
            ## How many bins do I need to insert?
            jump = int(np.round(diff/rec_diff))-1
            ## Need to insert NaNs
            n+=jump ## Move to next bin in output array
            # continue
            update_rec_diff = False ## Change the reference wavelength
        if update_rec_diff:
            rec_diff = diff
    return newwave, indices

def rebuilt_wavelength_v2(wave):
    new_wave = wave.copy()
    # med_diff is considered the typical sampling
    med_diff = np.min(np.diff(new_wave[~np.isnan(new_wave)]))
    loc_diff = med_diff
    nstart = 0
    ## first, fill in the middle
    for bin in range(len(new_wave)-1):
        if np.isnan(new_wave[bin]): 
            nstart = bin+1
            # pass
        else:
            if np.isnan(new_wave[bin+1]):
                new_wave[bin+1] = new_wave[bin]+loc_diff
            else:
                loc_diff = new_wave[bin+1]-new_wave[bin]
    if nstart>0:
        loc_diff = new_wave[nstart+1]-new_wave[nstart]
        for bin in range(nstart, 0, -1):
            if np.isnan(new_wave[bin-1]):
                new_wave[bin-1] = new_wave[bin]-med_diff
    return new_wave

def fill_nans_wavelength_v2(med_wvl):
    '''
    Notes:
    Update Sep. 18, 2025: Change of rationale
    ASAP should no longer require wavelength to be evenly spaced, but does
    require non-nan, increasing wavelengths
    fill_nans_wavelength_v3_v2 will replace NaNs based median sampling '''
    ####################################
    if np.any(np.isnan(med_wvl)):
        ## Some people put NaNs in the wavelengths... don't ask.
        ## Here is a fix:
        new_med_wvl = np.empty((len(med_wvl), len(med_wvl[0])*10))
        indices = []
        for r in range(len(med_wvl)):
            ## Here I add NaN bins between wavelength jumps
            new_med_wvl[r], idx = rebuilt_wavelength(med_wvl[r])
            indices.append(idx)
    else:
        new_med_wvl = med_wvl
        indices = np.array([np.arange(len(med_wvl[0])) for i in range(len(med_wvl))])
    ########################################## 
    return new_med_wvl, indices

from scipy.interpolate import interp1d
def fill_nans_wavelength_v3(med_wvl):
    '''
    Notes:
    Update Sep. 18, 2025: Change of rationale
    Same as fill_nans_wavelength, but use cubic spline between pixels.'''
    ####################################
    if np.any(np.isnan(med_wvl)):
        ## Some people put NaNs in the wavelengths... don't ask.
        ## Here is a fix:
        new_med_wvl = np.empty(med_wvl.shape)
        for r in range(len(med_wvl)):
            x = np.arange(len(med_wvl[r]), dtype=float)
            idx = np.where(~np.isnan(med_wvl[r]))
            ## What is the first non-NaN bin?
            for i in range(len(med_wvl[r])):
                if ~np.isnan(med_wvl[r][i]):
                    firstbin = i
                    break
            for i in range(len(med_wvl[r])-1, 0, -1):
                if ~np.isnan(med_wvl[r][i]):
                    lastbin = i
                    break
            fun = interp1d(x[idx], med_wvl[r][idx], kind='cubic')
            new_med_wvl[r, firstbin:lastbin] = fun(x[firstbin:lastbin])
            ## Now handle the edges:
            lowdiff = new_med_wvl[r, firstbin+1]-new_med_wvl[r, firstbin]
            maxdiff = new_med_wvl[r, lastbin-1]-new_med_wvl[r, lastbin-2]
            lowend = np.arange(firstbin)*lowdiff + (new_med_wvl[r, firstbin]-firstbin*lowdiff)
            lenright = len(new_med_wvl[r])-lastbin #
            highend = np.arange(1, lenright+1)*maxdiff + (new_med_wvl[r, lastbin-1])
            new_med_wvl[r, :firstbin] = lowend
            new_med_wvl[r, lastbin:] = highend
            
            # plt.plot(np.diff(new_med_wvl[r]))
            if np.any(np.diff(new_med_wvl[r])<0):
                print('FATAL ERROR - non-increasing wavelength')
                print('We should not be reaching this point...')
                from IPython import embed;embed()
                raise Exception('FATAL ERROR - non-increasing wavelength')
    ########################################## 
    return new_med_wvl