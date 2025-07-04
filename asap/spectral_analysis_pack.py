
from asap import params
import numpy as np
from numba import jit
from asap import analysis_tools as tls
from asap import effects as effects
from asap import normalization_tools as norm_tools
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
    vrad, wvls, spectrum, obs_wvl, obs_flux, obs_err, mask, \
            vinstru, vmac, vsini, t, l, m, model, adj, function = args
    if "macProf" in kwargs.keys(): 
        macProf = kwargs['macProf']
    else:
        macProf = 'g'
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
        _wvls = wvls[r] * doppler_factor
        # _spectra = effects.broadened_profile(
        #     _wvls, spectrum[r], 
        #     rv=None, epsilon=0.6,
        #     vsini=vsini, vmac=vmac, 
        #     vinstru = vinstru
        #     )
        _spectra = effects.broaden_spectrum_2(_wvls, spectrum[r], 
                                            vinstru=vinstru, 
                                            vsini=vsini, epsilon=0.6, 
                                            vmac=vmac, vmac_mode=macProf)

        _wvl = _wvls
        _flux = _spectra
        # _spectrum = inte.fftintegrate(obs_wvl[r], _wvl, _flux)
        _spectrum = np.interp(obs_wvl[r], _wvl, _flux)
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
            # except:
            #     from IPython import embed
            #     embed()
            if len(X)==1:
                X = np.array([0, X[0]])
                Xerr = np.array([0, Xerr[0]])
            # except:
            #     from IPython import embed
            #     embed()
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
        new_med_wvl = np.empty(med_wvl.shape)
        prev_high = 0 ## highest wvl of the previous order 
        for r in range(len(med_wvl)):
            x = np.arange(len(med_wvl[r]), dtype=float)
            idx = np.where(~np.isnan(med_wvl[r]))
            poly_order = 6
            if len(x[idx])==2:
                poly_order = 1
            elif len(x[idx])<2:
                poly_order = 0
            
            ## Let's fit a polynomial to that
            # from irap_tools import polyfit as polyfit
            # from irap_tools import normalizationTools as norm_tools

            if poly_order>0:
                x = norm_tools.normalize_axis(x, x)
                coeffs = polyfit.fit_1d_polynomial(x[idx], 
                                                   np.array(med_wvl[r][idx], 
                                                            dtype=float),poly_order)
                fit = polyfit.poly1d(x, coeffs)
                new_med_wvl[r] = fit
                prev_high = new_med_wvl[r][-1]
            else:
                ## In that case we have a problem.
                ## We start from the last wavelength of the previous order
                ## We end with the a default value of 25000 (like for SPIRou).
                ## It shouldn't really matter because this is full of NaNs, that we are going to ignore.
                next_low = 25000
                new_med_wvl[r] = np.linspace(prev_high, next_low, len(new_med_wvl[r]), dtype=float)

            ## Check that the residuals are sufficiently low:
            std = np.nanstd(med_wvl[r] - new_med_wvl[r])

            if std>1e-6:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(med_wvl[r] - new_med_wvl[r])
                plt.show()
                print(f'STD = {std}')
                raise Exception('Reconstructing wavelength solution: STD too high. Contact Author')
    ########################################## 
    return new_med_wvl