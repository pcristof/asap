# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport NAN
import numpy as np
cimport numpy as np
cimport cython

from asap.c_tools import linsolve

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

# ---- Quickselect (in-place) ----
cdef double quickselect(double[:] arr, ITYPE_t k):
    cdef ITYPE_t left = 0, right = arr.shape[0] - 1
    cdef ITYPE_t i, j
    cdef double pivot, tmp
    while left < right:
        pivot = arr[right]
        i = left
        for j in range(left, right):
            if arr[j] <= pivot:
                tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
                i += 1
        tmp = arr[i]
        arr[i] = arr[right]
        arr[right] = tmp
        if k == i:
            return arr[i]
        elif k < i:
            right = i - 1
        else:
            left = i + 1
    return arr[left]

# ---- Percentile ignoring NaNs (in-place compaction) ----
cdef double percentile_nonan_inplace(double[:] arr, double p):
    cdef ITYPE_t n = arr.shape[0]
    cdef ITYPE_t count = 0
    cdef ITYPE_t i
    # compact non-NaN values in-place
    for i in range(n):
        if arr[i] == arr[i]:  # not NaN
            arr[count] = arr[i]
            count += 1
    if count == 0:
        return NAN
    cdef ITYPE_t k = <ITYPE_t>((p / 100.0) * (count - 1))
    return quickselect(arr[:count], k)

# ---- Fast mean ----
cdef double mean_fast(double[:] arr):
    cdef ITYPE_t i, n = arr.shape[0]
    cdef double s = 0.
    for i in range(n):
        s += arr[i]
    return s / n

# ---- Main function ----
@cython.boundscheck(False)
@cython.wraparound(False)
def adjust_continuum5_fast_inplace(
    double[:] wvl,
    double[:] obs_flux,
    double[:] model_flux,
    double p = 50.0,
    int nWindows = 6
    ):
    cdef ITYPE_t N = obs_flux.shape[0]
    cdef ITYPE_t window_size = N // nWindows
    cdef ITYPE_t i, j, idx

    cdef double[:] obs_view = np.empty(N, dtype=np.float64)
    cdef double[:] mod_view = np.empty(N, dtype=np.float64)
    cdef double[:] wvl_view = np.empty(N, dtype=np.float64)
    for i in range(N):
        obs_view[i] = obs_flux[i]  # simple fast copy
        mod_view[i] = model_flux[i]  # simple fast copy
        wvl_view[i] = wvl[i]  # simple fast copy

    # ---- Mask extreme obs_flux values once ----
    cdef double upper = percentile_nonan_inplace(obs_view, 98.0)
    cdef double lower = percentile_nonan_inplace(obs_view, 10.0)
    for i in range(N):
        if obs_view[i] > upper or obs_view[i] < lower:
            obs_view[i] = NAN

    # ---- Pre-allocate window arrays once ----
    cdef double[:] window_obs = np.empty(window_size, dtype=np.float64)
    cdef double[:] window_mod = np.empty(window_size, dtype=np.float64)
    cdef double[:] window_wvl = np.empty(window_size, dtype=np.float64)

    cdef np.ndarray[DTYPE_t, ndim=1] wave_points = np.zeros(nWindows, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] obs_points = np.zeros(nWindows, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mod_points = np.zeros(nWindows, dtype=np.float64)

    # ---- Main loop over windows ----
    for idx in range(nWindows):
        # Copy window into memoryviews
        for j in range(window_size):
            i = idx * window_size + j
            window_obs[j] = obs_view[i]
            window_mod[j] = mod_view[i]
            window_wvl[j] = wvl_view[i]

        # Compute percentiles and mean, in-place compaction for obs
        obs_points[idx] = percentile_nonan_inplace(window_obs, p)
        mod_points[idx] = percentile_nonan_inplace(window_mod, p)  # model assumed no NaNs
        wave_points[idx] = mean_fast(window_wvl)

    cdef np.ndarray[DTYPE_t, ndim=2] mat = np.empty((2, nWindows), dtype=np.float64)
    for j in range(nWindows):
        mat[0, j] = wave_points[j]
        mat[1, j] = 1.  

    cdef double[::1] c_obs
    cdef double[::1] c_mod
    c_mod = linsolve.hybrid_linsolve_nocov(mat, mod_points)
    c_obs = linsolve.hybrid_linsolve_nocov(mat, obs_points)

    # cdef double[:] droite_model = np.empty(N, dtype=np.float64)
    # cdef double[:] droite_obs = np.empty(N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(N, dtype=np.float64)
    for i in range(N):
        value1 = c_mod[0]*wvl[i] + c_mod[1]
        value2 = c_obs[0]*wvl[i] + c_obs[1]
        result[i] = value1/value2


    return result, wave_points, obs_points, mod_points
