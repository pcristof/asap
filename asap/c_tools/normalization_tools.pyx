# adjust_continuum5_cy.pyx
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.view cimport array

# You may need to cimport your tls module or keep Python call
# from asap import analysis_tools as tls

# def linsolve_nocov_c(np.ndarray[double, ndim=2] mat,
#                            np.ndarray[double, ndim=1] Y):
#     """
#     Fast linear least squares for 2xN matrix:
#         mat[0,:] = x values
#         mat[1,:] = ones
#     Solves min ||mat.T @ coeffs - Y||^2
#     Returns coeffs = [a, b] for y = a*x + b
#     """

#     cdef Py_ssize_t N = mat.shape[1]
#     cdef double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Sxy = 0.0
#     cdef double a, b
#     cdef Py_ssize_t i
#     cdef double det

#     # Loop over points
#     for i in range(N):
#         Sx += mat[0, i]
#         Sy += Y[i]
#         Sxx += mat[0, i] * mat[0, i]
#         Sxy += mat[0, i] * Y[i]

#     # Solve 2x2 normal equations explicitly:
#     # [Sxx  Sx] [a] = [Sxy]
#     # [Sx    N] [b]   [Sy]
#     det = Sxx * N - Sx * Sx
#     if det == 0.0:
#         # Degenerate case: vertical line? Return zeros
#         a = 0.0
#         b = 0.0
#     else:
#         a = (Sxy * N - Sy * Sx) / det
#         b = (Sxx * Sy - Sx * Sxy) / det

#     return np.array([a, b], dtype=np.float64)

# def adjust_continuum5_c(
#         np.ndarray[double, ndim=1] wvl,
#         np.ndarray[double, ndim=1] obs_flux,
#         np.ndarray[double, ndim=1] model_flux,
#         int p=50,
#         int nWindows=6,
#         int degree=1,
#         double m=0.5,
#         function='line'
#     ):
#     """
#     Cython version of adjust_continuum5
#     """

#     cdef Py_ssize_t _len = (obs_flux.shape[0] // 10) * 10
#     cdef np.ndarray[double, ndim=1] obs_flux_loc = np.copy(obs_flux[:_len])
#     cdef np.ndarray[double, ndim=1] model_flux_loc = np.copy(model_flux[:_len])
#     cdef np.ndarray[double, ndim=1] wvl_loc = np.copy(wvl[:_len])
#     cdef np.ndarray[double, ndim=1] droite_model = np.empty(_len)
#     cdef np.ndarray[double, ndim=1] droite_obs = np.empty(_len)
    
#     cdef np.ndarray[double, ndim=1] IDXOBS = ~np.isnan(obs_flux_loc)
#     cdef np.ndarray[double, ndim=1] IDXOBS2 = ~np.isnan(obs_flux_loc)
#     cdef np.ndarray[double, ndim=1] IDXMOD = ~np.isnan(model_flux_loc)
#     cdef np.ndarray[double, ndim=1] IDXMOD2 = ~np.isnan(model_flux_loc)
#     cdef bint solidmodel

#     cdef double[::1] c1 = np.empty(2)
#     cdef double[::1] c2 = np.empty(2)
#     cdef double[::1] e1 = np.empty(2)
#     cdef double[::1] e2 = np.empty(2)

#     cdef Py_ssize_t size
#     cdef Py_ssize_t i, j

#     cdef double[::1] wvl_loc_view = wvl_loc

#     model_flux_loc[model_flux_loc < 0.1] = np.nan

#     solidmodel = False
#     if np.sum(np.sqrt((model_flux_loc - np.mean(model_flux_loc))**2)) < 0.1:
#         solidmodel = True
#         # print("//// SOLIDMODEL = TRUE ////")

#     if np.sum(IDXOBS) < 2:
#         c1 = np.array([0., 1.])
#         c2 = np.array([0., 1.])
#         e1 = np.array([0., 0.])
#         e2 = np.array([0., 0.])
#         w_obs = p_obs = w_mod = p_mod = np.array([0.])
#         droite_model = c1[0]*wvl + c1[1]
#         droite_obs = c2[0]*wvl + c2[1]
#         return droite_model, [w_obs, p_obs, w_mod, p_mod], c1, e1, c2, e2

#     ## Reject extremes
#     if solidmodel:
#         IDXMOD2 = IDXMOD
#     else:
#         IDXMOD2 = model_flux_loc[IDXMOD] < np.percentile(model_flux_loc[IDXMOD], 98)
#     IDXOBS2 = obs_flux_loc[IDXOBS] < np.percentile(obs_flux_loc[IDXOBS], 98)
#     IDXMOD[IDXMOD] = IDXMOD[IDXMOD] & IDXMOD2
#     IDXOBS[IDXOBS] = IDXOBS[IDXOBS] & IDXOBS2

#     if solidmodel:
#         IDXMOD2 = IDXMOD
#     else:
#         IDXMOD2 = model_flux_loc[IDXMOD] > np.percentile(model_flux_loc[IDXMOD], 10)
#     IDXOBS2 = obs_flux_loc[IDXOBS] > np.percentile(obs_flux_loc[IDXOBS], 10)
#     IDXMOD[IDXMOD] = IDXMOD[IDXMOD] & IDXMOD2
#     IDXOBS[IDXOBS] = IDXOBS[IDXOBS] & IDXOBS2

#     obs_flux_loc[~IDXOBS] = np.nan
#     model_flux_loc[~IDXMOD] = np.nan

#     ## Split into windows
#     size = _len // nWindows
#     IDXMODF = np.zeros(_len, dtype=np.bool_)
#     IDXOBSF = np.zeros(_len, dtype=np.bool_)
    
#     # cdef Py_ssize_t i, j
#     for j in range(nWindows):
#         i = j*size
#         SEC = IDXOBS[i:i+size]
#         pobs = np.percentile(obs_flux_loc[i:i+size][SEC], p)
#         IDXCROSSOBS = obs_flux_loc[i:i+size] > pobs
#         if np.sum(IDXCROSSOBS) < 4:
#             IDXCROSSOBS[IDXCROSSOBS] = False
#         IDXOBSF[i:i+size] = IDXOBS[i:i+size] & IDXCROSSOBS

#         SEC = IDXMOD[i:i+size]
#         if np.all(SEC == False):
#             IDXMODF[i:i+size] = [False] * len(IDXMODF[i:i+size])
#             IDXOBSF[i:i+size] = [False] * len(IDXOBSF[i:i+size])
#             continue
#         else:
#             pmod = np.percentile(model_flux_loc[i:i+size][SEC], p)
#             IDXCROSSMOD = model_flux_loc[i:i+size] > pmod
#             if solidmodel:
#                 IDXCROSSMOD = model_flux_loc[i:i+size] == model_flux_loc[i:i+size]
#             if np.sum(IDXCROSSMOD) < 4:
#                 IDXCROSSMOD[IDXCROSSMOD] = False
#             IDXMODF[i:i+size] = IDXMOD[i:i+size] & IDXCROSSMOD

#     w_mod = wvl_loc[IDXMODF]
#     w_obs = wvl_loc[IDXOBSF]
#     p_mod = model_flux_loc[IDXMODF]
#     p_obs = obs_flux_loc[IDXOBSF]

#     ## Linear least squares
#     matmod = np.vstack([w_mod, np.ones(len(w_mod))])
#     matobs = np.vstack([w_obs, np.ones(len(w_obs))])
    
#     if solidmodel:
#         c1 = np.array([0, np.mean(model_flux)])
#     else:
#         # Python call to your linear solve function
#         c1 = linsolve_nocov_c(matmod.T, p_mod, np.ones(p_mod.shape))
#     c2 = linsolve_nocov_c(matobs.T, p_obs, np.ones(p_obs.shape))
#     e1 = np.array([0,0])
#     e2 = np.array([0,0])

#     droite_model = c1[0]*wvl + c1[1]
#     droite_obs = c2[0]*wvl + c2[1]

#     return droite_model/droite_obs, [w_obs, p_obs, w_mod, p_mod], c1, e1, c2, e2




# adjust_continuum5_cy.pyx
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.math cimport floor  # floor function from C
cimport cython

# cdef double percentile(double[::1] arr, int n):
#     """
#     Compute the nth percentile of a 1D array ignoring NaNs.
#     Simple implementation using sorting (for small arrays this is fine).
#     """
#     cdef Py_ssize_t i, count = 0, length = arr.shape[0]
#     cdef double[:] tmp = np.empty(length, dtype=np.float64)
    
#     # Copy non-NaN values
#     for i in range(length):
#         if not np.isnan(arr[i]):
#             tmp[count] = arr[i]
#             count += 1

#     if count == 0:
#         return np.nan

#     tmp = tmp[:count]
#     tmp.sort()
#     cdef double rank = (n/100.0)*(count-1)
#     cdef Py_ssize_t low = int(np.floor(rank))
#     cdef Py_ssize_t high = int(np.ceil(rank))
#     cdef double weight = rank - low
#     return (1.0 - weight)*tmp[low] + weight*tmp[high]


@cython.boundscheck(False)
@cython.wraparound(False)
def percentile(double[::1] arr, np.uint8_t[::1] mask, int p):
    """
    Compute the p-th percentile of arr, considering only entries where mask is True.
    Fully typed memoryview-based, no Python object allocation inside hot loops.
    """
    cdef Py_ssize_t n = arr.shape[0]
    cdef Py_ssize_t count = 0
    cdef Py_ssize_t i, j, k, idx
    cdef double tmp, perc

    # First pass: count valid elements
    for i in range(n):
        if mask[i]:
            count += 1

    if count == 0:
        return 0.0

    # Allocate fixed-size C array on stack (safe if count is reasonable, < ~10^6)
    cdef double[::1] vals = arr[0:count]  # temporary memoryview

    # Copy valid elements into vals
    k = 0
    for i in range(n):
        if mask[i]:
            vals[k] = arr[i]
            k += 1

    # Simple in-place selection sort
    for i in range(count):
        for j in range(i+1, count):
            if vals[j] < vals[i]:
                tmp = vals[i]
                vals[i] = vals[j]
                vals[j] = tmp

    # Compute index for percentile
    idx = <Py_ssize_t>floor((p / 100.0) * (count - 1))
    perc = vals[idx]

    return perc



# Linear least squares for 2xN array
cdef np.ndarray[double, ndim=1] linsolve_nocov_c(double[:, ::1] mat, double[::1] Y):
    cdef Py_ssize_t N = mat.shape[1]
    cdef double Sx=0.0, Sy=0.0, Sxx=0.0, Sxy=0.0
    cdef Py_ssize_t i
    cdef double det, a, b

    for i in range(N):
        Sx += mat[0,i]
        Sy += Y[i]
        Sxx += mat[0,i]*mat[0,i]
        Sxy += mat[0,i]*Y[i]

    det = Sxx*N - Sx*Sx
    if det == 0.0:
        a = 0.0
        b = 0.0
    else:
        a = (Sxy*N - Sy*Sx)/det
        b = (Sxx*Sy - Sx*Sxy)/det

    return np.array([a,b], dtype=np.float64)


def adjust_continuum5_cy(
        np.ndarray[double, ndim=1] wvl,
        np.ndarray[double, ndim=1] obs_flux,
        np.ndarray[double, ndim=1] model_flux,
        int p=50,
        int nWindows=6
    ):
    """
    Full Cython version of adjust_continuum5 with memoryviews.
    """

    cdef Py_ssize_t _len = (obs_flux.shape[0]//10)*10
    cdef double[::1] wvl_loc = wvl[:_len]
    cdef double[::1] obs_flux_loc = np.copy(obs_flux[:_len])
    cdef double[::1] model_flux_loc = np.copy(model_flux[:_len])
    cdef np.uint8_t[::1] IDXOBS = array(
        shape=(_len,),
        itemsize=sizeof(np.uint8_t),
        format="B",   # unsigned char
    )    
    cdef np.uint8_t[::1] IDXMOD = array(
        shape=(_len,),
        itemsize=sizeof(np.uint8_t),
        format="B",   # unsigned char
    )    
    cdef np.uint8_t[::1] IDXOBSF = array(
        shape=(_len,),
        itemsize=sizeof(np.uint8_t),
        format="B",   # unsigned char
    )    
    cdef np.uint8_t[::1] IDXMODF = array(
        shape=(_len,),
        itemsize=sizeof(np.uint8_t),
        format="B",   # unsigned char
    )    
    # cdef _Bool[::1] IDXMOD = array(shape=(_len,), itemsize=sizeof(_Bool), format="?") # b for boolean
    # cdef _Bool[::1] IDXOBSF = array(shape=(_len,), itemsize=sizeof(_Bool), format="?") # b for boolean
    # cdef _Bool[::1] IDXMODF = array(shape=(_len,), itemsize=sizeof(_Bool), format="?") # b for boolean

    cdef Py_ssize_t i,j,size,count
    cdef double val
    cdef bint solidmodel = 0
    cdef double mean_model
    cdef double perc
    cdef double tmpval
    cdef bint keep

    ## Initilialize the boolean arrays:
    for i in range(_len):
        IDXOBS[i] = 0
        IDXMOD[i] = 0
        IDXOBSF[i] = 0
        IDXMODF[i] = 0

    # Replace model_flux < 0.1 with nan
    for i in range(_len):
        if model_flux_loc[i] < 0.1:
            model_flux_loc[i] = np.nan

    # Check if model is nearly flat
    val = 0.0
    mean_model = 0.0
    for i in range(_len):
        if not np.isnan(model_flux_loc[i]):
            mean_model += model_flux_loc[i]
    count = 0
    for i in range(_len):
        if not np.isnan(model_flux_loc[i]):
            count += 1
    if count > 0:
        mean_model /= count
    for i in range(_len):
        if not np.isnan(model_flux_loc[i]):
            val += sqrt((model_flux_loc[i]-mean_model)**2)
    if val < 0.1:
        solidmodel = 1

    # Masks for valid observations
    for i in range(_len):
        IDXOBS[i] = not np.isnan(obs_flux_loc[i])
        IDXMOD[i] = not np.isnan(model_flux_loc[i])

    # Early return for very few points
    count = 0
    for i in range(_len):
        if IDXOBS[i]:
            count += 1
    if count < 2:
        c1 = np.array([0.,1.])
        c2 = np.array([0.,1.])
        e1 = np.array([0.,0.])
        e2 = np.array([0.,0.])
        droite_model = c1[0]*wvl + c1[1]
        droite_obs = c2[0]*wvl + c2[1]
        w_obs = p_obs = w_mod = p_mod = np.array([0.])
        return droite_model, [w_obs, p_obs, w_mod, p_mod], c1, e1, c2, e2

    # Reject top and bottom percentiles
    
    perc = percentile(model_flux_loc, IDXMOD, 98)
    for i in range(_len):
        if IDXMOD[i] and not solidmodel:
            IDXMOD[i] = model_flux_loc[i] < perc
    perc = percentile(obs_flux_loc, IDXOBS, 98)
    for i in range(_len):
        if IDXOBS[i]:
            IDXOBS[i] = obs_flux_loc[i] < perc

    perc = percentile(model_flux_loc, IDXMOD, 10)
    for i in range(_len):
        if IDXMOD[i] and not solidmodel:
            IDXMOD[i] = model_flux_loc[i] > perc
    perc = percentile(obs_flux_loc, IDXOBS, 10)
    for i in range(_len):
        if IDXOBS[i]:
            IDXOBS[i] = obs_flux_loc[i] > perc

    # Set NaNs for rejected points
    for i in range(_len):
        if not IDXOBS[i]:
            obs_flux_loc[i] = np.nan
        if not IDXMOD[i]:
            model_flux_loc[i] = np.nan

    # Windowed percentile selection
    size = _len // nWindows

    for j in range(nWindows):
        i = j*size
        # Obs
        count = 0
        for k in range(i, i+size):
            if IDXOBS[k]:
                count += 1
        if count == 0:
            continue
        tmp = np.empty(count, dtype=np.float64)
        count = 0
        for k in range(i, i+size):
            if IDXOBS[k]:
                tmp[count] = obs_flux_loc[k]
                count += 1
        pobs = percentile(tmp, p)
        for k in range(i, i+size):
            keep = 1
            if IDXOBS[k] and obs_flux_loc[k] <= pobs:
                keep = 0
            IDXOBSF[k] = IDXOBS[k] and keep

        # Model
        count = 0
        for k in range(i, i+size):
            if IDXMOD[k]:
                count += 1
        if count == 0:
            for k in range(i, i+size):
                IDXMODF[k] = False
                IDXOBSF[k] = False
            continue
        tmp = np.empty(count, dtype=np.float64)
        count = 0
        for k in range(i, i+size):
            if IDXMOD[k]:
                tmp[count] = model_flux_loc[k]
                count += 1
        pmod = percentile(tmp, p)
        for k in range(i, i+size):
            keep = 1
            if IDXMOD[k] and model_flux_loc[k] <= pmod and not solidmodel:
                keep = 0
            IDXMODF[k] = IDXMOD[k] and keep

    # Collect final points
    w_mod = np.array([wvl_loc[i] for i in range(_len) if IDXMODF[i]])
    p_mod = np.array([model_flux_loc[i] for i in range(_len) if IDXMODF[i]])
    w_obs = np.array([wvl_loc[i] for i in range(_len) if IDXOBSF[i]])
    p_obs = np.array([obs_flux_loc[i] for i in range(_len) if IDXOBSF[i]])

    # Linear least squares
    matmod = np.empty((2, len(w_mod)), dtype=np.float64)
    matmod[0,:] = w_mod
    matmod[1,:] = 1.0
    matobs = np.empty((2, len(w_obs)), dtype=np.float64)
    matobs[0,:] = w_obs
    matobs[1,:] = 1.0

    if solidmodel:
        c1 = np.array([0, np.mean(model_flux_loc)])
    else:
        c1 = linsolve_nocov_c(matmod, p_mod)
    c2 = linsolve_nocov_c(matobs, p_obs)
    e1 = np.array([0.,0.])
    e2 = np.array([0.,0.])

    droite_model = c1[0]*wvl + c1[1]
    droite_obs = c2[0]*wvl + c2[1]

    return droite_model/droite_obs, [w_obs, p_obs, w_mod, p_mod], c1, e1, c2, e2
