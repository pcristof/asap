from numba import jit
from asap import params
import numpy as np
from asap import analysis_tools as tls
from scipy.optimize import curve_fit

@jit(nopython=True, cache=params.CACHE)
def adjust_continuum5(wvl, obs_flux, model_flux, p=50, window_size=None,
                        degree=1, m=0.5, function='line', nWindows=6):
    '''

    adjust_continuum version 5 !

    This version is the same as adjust_continuum4 but we try to make it 
    compatible with numba.

    Input parameters:
    - flux      :   [1d array] flux to normalize
    - window_size : dummy variable. TODO: remove it
    - p         :   [int, float] percentile value to consider to find points
    - degree    :   [int] degree of the polynomial to fit on the considered 
                    points
    - m=0.5     :   dummy variable.
    '''

    solidmodel = False
    ## Check if the model provided is a single horizontal line (to 0.05 precision):
    if np.sum(np.sqrt((model_flux-np.mean(model_flux))**2))<0.1:
    # if np.sqrt((np.max(model_flux) - np.min(model_flux))**2) > 0.05:
        ## we passed a line
        solidmodel = True
        print('-------------------------------------')
        print('-------------------------------------')
        print('////!!!! SOLIDMODEL = TRUE !!!!!!\\\\\\\\')
        print('-------------------------------------')
        print('-------------------------------------')
    ## CAUTION - The function here assumes that obs_flux and model_flux
    ## have the same sampling on the wvl grid

    ## Make sure we have an even length of input array (required for split):
    _len = int(np.floor(len(obs_flux)/10)*10); ## Even length array
    obs_flux_loc = np.copy(obs_flux[:_len])
    wvl_loc = np.copy(wvl[:_len])
    model_flux_loc = np.copy(model_flux[:_len])
    model_flux_loc[model_flux_loc<0.1] *= np.nan  ## This should also exlude the values at 0.0 \
                                                ## that we've set when data is missing in the models 
    ## Compute boolean arrays to locate non NaN values
    IDXOBS = ~np.isnan(obs_flux_loc)
    IDXMOD = ~np.isnan(model_flux_loc)

    if sum(IDXOBS)<2:
        _c1 = np.array([0., 1.]); 
        _c2 = np.array([0., 1.]);
        _e1 = np.array([0, 0])
        _e2 = np.array([0, 0])
        _X = _c1
        _Xerr = _e1
        _X2 = _c2
        _Xerr2 = _e2
        _droite_model = _c1[0]*wvl + _c1[1]
        _droite_obs = _c2[0]*wvl + _c2[1]
        _w_obs = _p_obs = _w_mod = _p_mod = np.array([0.])
        # print('coucou')
        return _droite_model, [_w_obs, _p_obs, _w_mod, _p_mod], _X, _Xerr, _X2, _Xerr2
    else:

        ## We reject all points above the 98 percentile (emissions)
        if solidmodel:
            IDXMOD2 = IDXMOD
        else:
            IDXMOD2 = model_flux_loc[IDXMOD]<np.percentile(model_flux_loc[IDXMOD], 98)
        IDXOBS2 = obs_flux_loc[IDXOBS]<np.percentile(obs_flux_loc[IDXOBS], 98)
        IDXMOD[IDXMOD] = IDXMOD[IDXMOD] & IDXMOD2 ## intersection booleans
        IDXOBS[IDXOBS] = IDXOBS[IDXOBS] & IDXOBS2

        ## We reject all points below the 50 percentile
        if solidmodel:
            IDXMOD2 = IDXMOD
        else:
            IDXMOD2 = model_flux_loc[IDXMOD]>np.percentile(model_flux_loc[IDXMOD], 10)
        IDXOBS2 = obs_flux_loc[IDXOBS]>np.percentile(obs_flux_loc[IDXOBS], 10)
        IDXMOD[IDXMOD] = IDXMOD[IDXMOD] & IDXMOD2
        IDXOBS[IDXOBS] = IDXOBS[IDXOBS] & IDXOBS2

        ## We place NaNs where the points are not from the continuum
        obs_flux_loc[~IDXOBS] = np.nan
        model_flux_loc[~IDXMOD] = np.nan

        # # ## We make chuncks based on window_size:
        # ## Option 1: we make the split and compute the percentiles with the axis
        # ## argument. This method is INCOMPATIBLE WITH NUMBA.
        # obs_flux_loc[~IDXOBS] = np.nan
        # model_flux_loc[~IDXMOD] = np.nan
        # obs_flux_loc2d = np.array(np.split(obs_flux_loc, 10))
        # model_flux_loc2d = np.array(np.split(model_flux_loc, 10))
        # wvl_loc2d = np.array(np.split(wvl_loc, 10))

        # pobs = np.nanpercentile(obs_flux_loc2d, p, axis=-1)
        # IDXOBSF = obs_flux_loc2d.T > pobs
        # pmod = np.nanpercentile(model_flux_loc2d, p, axis=-1)
        # IDXMODF = model_flux_loc2d.T > pmod

        # w_mod = wvl_loc2d[IDXMODF.T]; w_obs = wvl_loc2d[IDXOBSF.T]
        # p_mod = model_flux_loc2d[IDXMODF.T]; p_obs = obs_flux_loc2d[IDXOBSF.T]
        ## -------
        ## Option 2: we make a loop like we would in C. This method allows to 
        ## use the percentile rather than the nanpercentile and should be 
        ## compatible with numba.
        ## VERY clearly, this function is slower than the previous one in non-jit
        ## mode.
        ## /!\/!\/!\/!\/!\/!\ BROKEN METHOD /!\/!\/!\/!\
        ## /!\/!\ DOES NOT REPRODUCE THE RESULTS OF OPTION 1 /!\/!\
        #
        #
        ## We now want to split the array in 10 windows.
        # nWindows = 6 ## I had 6 befores
        size = _len // nWindows ## Otherwise it returns a float
        ## We initialize arrays of booleans
        IDXMODF = np.array([False for i in range(_len)])
        IDXOBSF = np.array([False for i in range(_len)])
        for j in range(nWindows):
            i = j*size

            SEC = IDXOBS[i:i+size]
            pobs = np.percentile(obs_flux_loc[i:i+size][SEC], p)
            # Check that we have enough points -- if not we are likely not in the
            # continuum
            IDXCROSSOBS = obs_flux_loc[i:i+size]>pobs
            if np.sum(IDXCROSSOBS)<4: IDXCROSSOBS[IDXCROSSOBS] = False
            IDXOBSF[i:i+size] = IDXOBS[i:i+size] & (obs_flux_loc[i:i+size]>pobs)

            SEC = IDXMOD[i:i+size]
            # Check that the model in the region is not all zeros. If that's the case, we probably
            # have a missing segment of the model.
            if np.all(SEC==False):
                IDXMODF[i:i+size] = [False for i in range(len(IDXMODF[i:i+size]))]
                IDXOBSF[i:i+size] = [False for i in range(len(IDXOBSF[i:i+size]))]
                continue
            else:
                pmod = np.percentile(model_flux_loc[i:i+size][SEC], p)
                IDXCROSSMOD = model_flux_loc[i:i+size]>pmod
                if solidmodel:
                    IDXCROSSMOD = model_flux_loc[i:i+size]==model_flux_loc[i:i+size]
                if np.sum(IDXCROSSMOD)<4: IDXCROSSMOD[IDXCROSSMOD] = False
                IDXMODF[i:i+size] = IDXMOD[i:i+size] & (model_flux_loc[i:i+size]>pmod)

        w_mod = wvl_loc[IDXMODF]; w_obs = wvl_loc[IDXOBSF]
        p_mod = model_flux_loc[IDXMODF]; p_obs = obs_flux_loc[IDXOBSF]

        #### ---- END OF OPTION CHOICE

        ## Go super fast with a linear least square solve:
        ## - Matrices
        matmod = np.empty((2, len(w_mod)))
        matobs = np.empty((2, len(w_obs)))
        matmod[0] = w_mod; matmod[1] = np.ones(w_mod.shape) ## For a straight line
        matobs[0] = w_obs; matobs[1] = np.ones(w_obs.shape) ## For a straight line
        ## - Solving the least square
        ##   Returning the covariance matrix is extremely slow. Only
        ##   advisable for debugging or if you really need it.
        returnCov = False
        if returnCov:
            pass
            ## The following should work, but fails on Titan for some reason.
            ## For now I just bypass it
            #
            # c1, cov1 = tls.jitlinsolve(matmod.T, p_mod, np.ones(p_mod.shape))
            # c2, cov2 = tls.jitlinsolve(matobs.T, p_obs, np.ones(p_obs.shape))
            # e1 = np.sqrt(np.diag(cov1))
            # e2 = np.sqrt(np.diag(cov2))
        else:
            if solidmodel:
                c1 = np.array([0, np.mean(model_flux)])
            else:
                c1 = tls.jitlinsolve_nocov(matmod.T, p_mod, np.ones(p_mod.shape))
            c2 = tls.jitlinsolve_nocov(matobs.T, p_obs, np.ones(p_obs.shape))
            e1 = np.array([0, 0])
            e2 = np.array([0, 0])
        #
        ## Bypass for test
        # w_obs, p_obs, w_mod, p_mod = 0,0,0,0
        # c1 = np.array([0, 1]); e1 = np.array([1, 1])
        # c2 = np.array([0, 1]); e2 = np.array([1, 1])
        #
        ## Compute the continua
        droite_model = c1[0]*wvl + c1[1]
        droite_obs = c2[0]*wvl + c2[1]
        X = c1
        Xerr = e1
        X2 = c2
        Xerr2 = e2

        # droite_model = np.ones(wvl.shape)
        # droite_obs = np.ones(wvl.shape)

        ## -----
        ## Old method
        ## -----
        # # X = tls.fit_line_dxdy(wvl_points2, diff2)
        # X, Xerr = tls.fit_to_data(w_mod, p_mod, function=function)
        # if len(X)>1:
        #     droite_model = X[0]*wvl + X[1]
        # elif len(X)==1:
        #     droite_model = 0*wvl + X[0]
        # X2, Xerr2 = tls.fit_to_data(w_obs, p_obs, function=function)
        # if len(X)>1:
        #     droite_obs = X2[0]*wvl + X2[1]
        # elif len(X)==1:
        #     droite_obs = 0*wvl + X2[0]

        ## Here we could reject the points that were the furthest from the fitted
        ## line

        
        # continuum = droite * model_flux
    return droite_model/droite_obs, [w_obs, p_obs, w_mod, p_mod], X, Xerr, X2, Xerr2

@jit(nopython=True, cache=params.CACHE)
def moving_median(Im,hws,btd=None, p=50):    
    """
    Compute the moving median of Im and divide Im by the resulting moving median computing within
    [W[N_bor],W[-N_bor]] with N_best points
    
    Inputs:
    - Im    :       1D exposure to normalize
    - hws   :       Half Window Size of the window used to compute the median
    - btd   :       Bins to delete on the edges [default is hws].
    
    Outputs:
    - I_nor: Normalised exposure (size: len(I_tm))
    - I_bin: Resulting moving median used to normalize the Im 
    """

    if btd is None:
        btd = hws
    ## Init the moving median
    # W_bin = np.empty(len(Wm)-N_bor)
    I_bin = np.empty(len(Im)) * np.nan
    
    ## Apply moving median
    for k in range(len(Im)):  
        ## Handle edges
        if k < hws:
            N_inf = 0
            N_sup = int(k+hws)
        elif k + hws > len(Im):
            N_inf = int(k-hws)
            N_sup = -1
        else:
            N_inf = int(k-hws)
            N_sup = int(k+hws)       
        # W_bin[k] = np.nanmedian(Wm[N_inf:N_sup])
        r = Im[N_inf:N_sup]
        # idx = sigma_clip(r,5,5) #Apply sigma-clipping
        # r = np.delete(r, idx)
        # if np.all(np.isnan(r)):
        #     I_bin[k] = np.nan    
        # else:
        # idx = ~np.isnan(r)
        I_bin[k] = np.nanpercentile(r, p)  #Take median

    ## Set borders to NaNs    
    I_bin[:btd] = np.nan
    I_bin[len(I_bin)-btd:] = np.nan
    I_nor = Im/I_bin

    return I_nor,I_bin

@jit(nopython=True, cache=params.CACHE)
def moving_median_vel(Wm, Im,hws,btd=None, p=50):    
    """
    Compute the moving median of Im and divide Im by the resulting moving median computing within
    [W[N_bor],W[-N_bor]] with N_best points
    
    Inputs:
    - Wm    :       This is the wavelength
    - Im    :       1D exposure to normalize
    - hws   :       Half Window Size of the window used to compute the median
    - btd   :       Bins to delete on the edges [default is hws].
    
    Outputs:
    - I_nor: Normalised exposure (size: len(I_tm))
    - I_bin: Resulting moving median used to normalize the Im 
    """
    
    ## In this one I assume that hws is in km/s
    windowsize = hws

    if btd is None:
        btd = hws
    ## Init the moving median
    # W_bin = np.empty(len(Wm)-N_bor)
    I_bin = np.empty(len(Im)) * np.nan
    
    ## Apply moving median
    for k in range(1, len(Im)-1):
        dspeed = (Wm[k+1]-Wm[k-1])/2/Wm[k]*3*1e5
        hws = windowsize / dspeed
        ## Handle edges
        if k < hws:
            N_inf = 0
            N_sup = int(k+hws)
        elif k + hws > len(Im):
            N_inf = int(k-hws)
            N_sup = -1
        else:
            N_inf = int(k-hws)
            N_sup = int(k+hws)       
        # W_bin[k] = np.nanmedian(Wm[N_inf:N_sup])
        r = Im[N_inf:N_sup]
        # idx = sigma_clip(r,5,5) #Apply sigma-clipping
        # r = np.delete(r, idx)
        # if np.all(np.isnan(r)):
        #     I_bin[k] = np.nan    
        # else:
        idx = ~np.isnan(r)
        I_bin[k] = np.percentile(r[idx], p)  #Take median

    ## Set borders to NaNs    
    I_bin[:btd] = np.nan
    I_bin[len(I_bin)-btd:] = np.nan
    I_nor = Im/I_bin

    return I_nor,I_bin

def fit_poly(x, z, degree=3):
    '''
    Returns the coefficients of the best fit obtained with 
    scipy.optimize.curve_fit. 

    Input parameters:
    - x         :   x values for data to fit
    - y         :   y values for data to fit
    - degree    :   degree of the polynomial used for fit

    Output parameters:
    - popt      :   optimal coefficients of the polynomial. The length of 
                    popt depends on the degree of the polynomial used.
    - pcov      :   covariance matrix on the parameters. The size of pcov
                    depends on the degree of the polynomial used. 
    '''
    ## Define local function based on the polynomial degree 
    if degree==8: 
        def _poly(x,k,a,b,c,d,e,f,g,h): return k*x**8+a*x**7+b*x**6+c*x**5+d*x**4+e*x**3+f*x**2+g*x+h
    elif degree==7: 
        def _poly(x,k,a,b,c,d,e,f,g): return k*x**7+a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
    elif degree==6: 
        def _poly(x,k,a,b,c,d,e,f): return k*x**6+a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==5: 
        def _poly(x,a,b,c,d,e,f): return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==4: 
        def _poly(x,b,c,d,e,f): return b*x**4+c*x**3+d*x**2+e*x+f
    elif degree==3: 
        def _poly(x,c,d,e,f): return c*x**3+d*x**2+e*x+f
    elif degree==2: 
        def _poly(x,d,e,f): return d*x**2+e*x+f
    elif degree==1: 
        def _poly(x,e,f): return e*x+f
    elif degree==0: 
        def _poly(x,f): return f
    else : raise Exception("fit_poly --> Polynomial degree not supported")

    ## Fit function on data    
    popt, pcov = curve_fit(_poly, x, z) 
    return popt, pcov

@jit(nopython=True)
def normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns (a-np.mean(b))/(np.max(a)-np.min(b))    
    '''
    c = (a-np.mean(b))/(np.max(b)-np.min(b))
    return c

@jit(nopython=True)
def revert_normalize_axis(a,b):
    '''
    method that modify array based on second array
    returns a*(np.max(b)-np.min(b))+np.mean(b)   
    '''
    c = a*(np.max(b)-np.min(b))+np.mean(b)
    return c