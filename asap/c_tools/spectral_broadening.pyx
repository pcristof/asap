# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan

# Typedefs
ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t ITYPE_t

# Assume these Cython functions exist and are cpdef/cdef
from asap.c_tools.effects import broaden_spectrum_2_cy
from asap.c_tools.normalization_tools import adjust_continuum5_fast_inplace, adjust_continuum6_fast_inplace

# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport sqrt

# Speed of light in km/s (adjust if needed)
cdef double C_LIGHT_KM_S = 2.99792458e5  # km/s

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double doppler_cy(double v):
    """
    Returns relativistic Doppler factor in km/s.
    
    Conventions:
    - v < 0 : source approaching observer
    - v > 0 : source receding from observer
    """
    cdef double _v = v
    cdef double beta, factor

    if _v >= C_LIGHT_KM_S:
        print("Caution, star going faster than light !")
        return float('inf')
    elif _v <= -C_LIGHT_KM_S:
        print("Caution, star going faster than light !")
        return 0.0

    beta = _v / C_LIGHT_KM_S
    factor = sqrt((1.0 + beta) / (1.0 - beta))
    return factor

@cython.boundscheck(False)
@cython.wraparound(False)
def broaden_spectra_cy_old(
    # list args, dict kwargs
    np.ndarray[DTYPE_t, ndim=2] wvls, 
    np.ndarray[DTYPE_t, ndim=2] spectrum, 
    np.ndarray[DTYPE_t, ndim=2] obs_wvl, 
    np.ndarray[DTYPE_t, ndim=2] obs_flux,
    np.ndarray[DTYPE_t, ndim=2] obs_err,
    double vinstru,
    double vmac,
    double vsini,
    double vrad,
    bint adj,
    vmac_mode='g'
):
    cdef double doppler_factor
    cdef ITYPE_t r, i, nregions, npts
    cdef np.ndarray[DTYPE_t, ndim=1] _wvls, _spectrum_b, _spectrum_b_interp
    # cdef np.ndarray[DTYPE_t, ndim=1] obs_wvl_r, obs_flux_r, 
    cdef np.ndarray[DTYPE_t, ndim=2] output
    cdef double[:] _c
    cdef double _wlim, _wlow

    # Doppler factor
    doppler_factor = doppler_cy(-vrad)

    nregions = obs_wvl.shape[0]
    npts = wvls.shape[1]
    npts_obs = obs_wvl.shape[1]
    output = np.empty((nregions,npts_obs))

    for r in range(nregions):
        # obs_wvl_r = obs_wvl[r]
        # obs_flux_r = obs_flux[r]
        # spectrum_r = spectrum[r]
        # wvls_r = wvls[r]

        # Skip if observation has NaNs
        if np.any(np.isnan(obs_flux[r])):
            continue
        if np.any(np.isnan(spectrum[r])):
            raise ValueError("NaN in model spectrum")
        if np.any(np.isnan(wvls[r])):
            raise ValueError("NaN in wvls")

        # Doppler shift
        _wvls = wvls[r] * doppler_factor

        # Broaden spectrum (Cython function)
        _spectrum_b = broaden_spectrum_2_cy(_wvls, spectrum[r],
                                         vinstru=vinstru,
                                         vsini=vsini, epsilon=0.6,
                                         vmac=vmac, vmac_mode=vmac_mode)

        # Wavelength limits
        # npts = _wvls.shape[0]
        _wlim = 0.
        _wlow = _wvls[0]
        for i in range(npts-1, -1, -1):
            if spectrum[r][i] != 0.:
                _wlim = _wvls[i]
                break

        # Interpolate to observation grid
        _spectrum_b_interp = np.interp(obs_wvl[r], _wvls, _spectrum_b)

        # Mask outside wavelength range
        for i in range(obs_wvl[r].shape[0]):
            if obs_wvl[r][i] > _wlim or obs_wvl[r][i] < _wlow:
                _spectrum_b_interp[i] = 0.

        # Continuum adjustment
        if adj:
            _c, wave_points, obs_points, mod_points = adjust_continuum5_fast_inplace(
                wvl=obs_wvl[r], obs_flux=obs_flux[r], model_flux=_spectrum_b_interp, p=90, nWindows=6)
        else:
            # Allocate _c as memoryview to avoid np.ones(_spectrum.shape) issues
            _c = np.empty(npts_obs, dtype=np.float64)
            for i in range(npts_obs):
                _c[i] = 1.0
        # Scale spectrum
        for i in range(npts_obs):
            _spectrum_b_interp[i] /= _c[i]

        output[r] = _spectrum_b_interp

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def broaden_spectra_cy(
    np.ndarray[DTYPE_t, ndim=2] wvls, 
    np.ndarray[DTYPE_t, ndim=2] spectrum, 
    np.ndarray[DTYPE_t, ndim=2] obs_wvl, 
    np.ndarray[DTYPE_t, ndim=2] obs_flux,
    np.ndarray[DTYPE_t, ndim=2] obs_err,
    double vinstru,
    double vmac,
    double vsini,
    double vrad,
    bint adj,
    vmac_mode='g'
):
    cdef ITYPE_t r, i, nregions, npts, npts_obs
    cdef double doppler_factor, _wlim, _wlow
    cdef np.ndarray[DTYPE_t, ndim=1] _spectrum_b, _spectrum_b_interp
    cdef double[:] _c
    cdef double[:] _wvls

    nregions = obs_wvl.shape[0]
    npts = wvls.shape[1]
    npts_obs = obs_wvl.shape[1]

    # Output preallocated
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.zeros((nregions, npts_obs), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] output_c = np.zeros((nregions, npts_obs), dtype=np.float64)

    # Compute doppler factor once
    doppler_factor = doppler_cy(-vrad)

    # memoryviews for speed
    cdef double[:, :] wvls_mv = wvls
    cdef double[:, :] spec_mv = spectrum
    cdef double[:, :] obs_wvl_mv = obs_wvl
    cdef double[:, :] obs_flux_mv = obs_flux
    cdef bint skip
    cdef double[:] spec_interp_mv
    cdef double[:] obs_wvl_row

    _wvls = np.empty(npts)

    for r in range(nregions):
        # Skip NaN rows
        skip = False
        for i in range(npts_obs):
            if isnan(obs_flux_mv[r,i]):
                skip = True
                break
        if skip:
            continue

        for i in range(npts):
            if isnan(spec_mv[r,i]):
                raise ValueError("NaN in model spectrum")
            if isnan(wvls_mv[r,i]):
                raise ValueError("NaN in wvls")

        # Doppler shift
        # _wvls = wvls_mv[r] * doppler_factor
        for i in range(npts):
            _wvls[i] = wvls_mv[r, i] * doppler_factor

        # Broaden spectrum
        _spectrum_b = broaden_spectrum_2_cy(_wvls, spec_mv[r],
                                            vinstru=vinstru,
                                            vsini=vsini,
                                            epsilon=0.6,
                                            vmac=vmac,
                                            vmac_mode=vmac_mode)

        # Wavelength limits (last non-zero)
        _wlow = _wvls[0]
        _wlim = 0.
        for i in range(npts-1, -1, -1):
            if spec_mv[r,i] != 0.:
                _wlim = _wvls[i]
                break

        # Interpolate to observation grid (cannot avoid np.interp here)
        _spectrum_b_interp = np.interp(obs_wvl_mv[r], _wvls, _spectrum_b)

        # Mask outside wavelength range
        spec_interp_mv = _spectrum_b_interp
        obs_wvl_row = obs_wvl_mv[r]
        for i in range(npts_obs):
            if obs_wvl_row[i] < _wlow or obs_wvl_row[i] > _wlim:
                spec_interp_mv[i] = 0.

        # Continuum adjustment
        if adj:
            _c, wave_points, obs_points, mod_points = adjust_continuum5_fast_inplace(
                wvl=obs_wvl_mv[r],
                obs_flux=obs_flux_mv[r],
                model_flux=_spectrum_b_interp,
                p=90,
                nWindows=6)
            # _c = adjust_continuum6_fast_inplace(obs_wvl_mv[r], 
            #                                     obs_flux_mv[r], 
            #                                     _spectrum_b_interp)
        else:
            # preallocate _c as memoryview
            _c = np.empty(npts_obs, dtype=np.float64)
            for i in range(npts_obs):
                _c[i] = 1.0

        # Scale spectrum
        for i in range(npts_obs):
            _spectrum_b_interp[i] /= _c[i]

        # Save to output
        output[r] = _spectrum_b_interp
        output_c[r] = _c

    return output, output_c