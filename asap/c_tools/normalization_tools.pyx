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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                               NEXT METHOD
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# cython: boundscheck=False, wraparound=False, cdivision=True

# ctypedef np.float64_t DTYPE_t
# ctypedef Py_ssize_t ITYPE_t

# -----------------------------
# Normalization
# -----------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def normalize_axis_cy(
    double[:] a, 
    double[:] b
    ):
    cdef ITYPE_t n = b.shape[0]
    cdef DTYPE_t mean_b = 0.
    cdef DTYPE_t min_b = b[0]
    cdef DTYPE_t max_b = b[0]
    cdef ITYPE_t i
    cdef double[:] out = np.empty(a.shape[0], dtype=np.float64)
    
    # Compute mean, min, max of b
    for i in range(n):
        mean_b += b[i]
        if b[i] < min_b:
            min_b = b[i]
        elif b[i] > max_b:
            max_b = b[i]
    mean_b /= n

    for i in range(a.shape[0]):
        out[i] = (a[i] - mean_b) / (max_b - min_b)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def revert_normalize_axis_cy(
    np.ndarray[DTYPE_t, ndim=1] a, 
    np.ndarray[DTYPE_t, ndim=1] b):
    cdef ITYPE_t n = b.shape[0]
    cdef DTYPE_t mean_b = 0.
    cdef DTYPE_t min_b = b[0]
    cdef DTYPE_t max_b = b[0]
    cdef ITYPE_t i
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(a.shape[0], dtype=np.float64)

    for i in range(n):
        mean_b += b[i]
        if b[i] < min_b:
            min_b = b[i]
        elif b[i] > max_b:
            max_b = b[i]
    mean_b /= n

    for i in range(a.shape[0]):
        out[i] = a[i] * (max_b - min_b) + mean_b

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def fit_1d_polynomial_cy(double[:] x, double[:] y, int degree=3, bint normalize_axes=False):
    """
    Fit 1D polynomial using memoryviews only (no NumPy).
    Returns coefficients from highest degree to constant term.
    """
    cdef ITYPE_t n = x.shape[0]
    cdef ITYPE_t i, j, k
    cdef double[:] xx = x

    # Normalize if needed
    if normalize_axes:
        xx = normalize_axis_cy(x, x)  # should also return memoryview

    cdef ITYPE_t d = degree
    cdef double[:, ::1] X = cython.view.array(shape=(n, d+1), itemsize=8, format="d", mode="c")  # Vandermonde

    # Build Vandermonde matrix
    for i in range(n):
        X[i,0] = 1.0
        for j in range(1, d+1):
            X[i,j] = X[i,j-1] * xx[i]

    # Compute X^T X (size (d+1, d+1)) and X^T Y (size d+1)
    cdef double[:, ::1] XTX = cython.view.array(shape=(d+1, d+1), itemsize=8, format="d", mode="c")
    cdef double[:] XTY = cython.view.array(shape=(d+1,), itemsize=8, format="d", mode="c")
    cdef double sum_val

    for i in range(d+1):
        for j in range(d+1):
            sum_val = 0.0
            for k in range(n):
                sum_val += X[k,i] * X[k,j]
            XTX[i,j] = sum_val

    for i in range(d+1):
        sum_val = 0.0
        for k in range(n):
            sum_val += X[k,i] * y[k]
        XTY[i] = sum_val

    # Solve linear system XTX * A = XTY via Gauss elimination
    cdef double[:] A = cython.view.array(shape=(d+1,), itemsize=8, format="d", mode="c")
    cdef ITYPE_t row, col
    cdef double factor, temp

    # Copy XTX and XTY to avoid modifying original (optional)
    cdef double[:, ::1] M = cython.view.array(shape=(d+1, d+1), itemsize=8, format="d", mode="c")
    cdef double[:] B = cython.view.array(shape=(d+1,), itemsize=8, format="d", mode="c")
    for i in range(d+1):
        B[i] = XTY[i]
        for j in range(d+1):
            M[i,j] = XTX[i,j]

    # Gaussian elimination
    for i in range(d+1):
        # Pivoting
        temp = M[i,i]
        if temp == 0.0:
            raise ValueError("Singular matrix in polynomial fit")
        for j in range(i+1, d+1):
            factor = M[j,i] / M[i,i]
            for k in range(i, d+1):
                M[j,k] -= factor * M[i,k]
            B[j] -= factor * B[i]

    # Back substitution
    for i in range(d, -1, -1):
        sum_val = B[i]
        for j in range(i+1, d+1):
            sum_val -= M[i,j] * A[j]
        A[i] = sum_val / M[i,i]

    return A

# cython: boundscheck=False, wraparound=False, cdivision=True

@cython.boundscheck(False)
@cython.wraparound(False)
def poly1d_horner(double[:] x,
                  double[:] coeffs,
                  bint normalize_axes=False,
                  np.ndarray[DTYPE_t, ndim=1] normalize_range=None):
    """
    Evaluate a polynomial using Horner's method.
    - x: points to evaluate (1D array)
    - coeffs: polynomial coefficients (highest degree first)
    - normalize_axes: if True, normalize x
    - normalize_range: optional range for normalization
    """
    cdef ITYPE_t n = x.shape[0]
    cdef ITYPE_t degree = coeffs.shape[0] - 1
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef double[:] xnorm = x
    cdef ITYPE_t i, j
    cdef DTYPE_t val

    # Optional normalization
    if normalize_axes:
        if normalize_range is None:
            xnorm = (x - np.mean(x)) / (np.max(x) - np.min(x))
        else:
            xnorm = (x - np.mean(normalize_range)) / (np.max(normalize_range) - np.min(normalize_range))

    # Memoryviews for speed
    cdef DTYPE_t[:] x_mv = xnorm
    cdef DTYPE_t[:] c_mv = coeffs
    cdef DTYPE_t[:] out_mv = out

    # Horner's method
    for i in range(n):
        val = c_mv[0]
        for j in range(1, degree+1):
            val = val * x_mv[i] + c_mv[j]
        out_mv[i] = val

    return out

# ---- Main function ----
@cython.boundscheck(False)
@cython.wraparound(False)
def adjust_continuum6_fast_inplace(
    double[:] wvl,
    double[:] obs_flux,
    double[:] model_flux,
    int degree = 5,
    int nWindows = 6
    ):
    cdef ITYPE_t N = obs_flux.shape[0]
    cdef ITYPE_t window_size = N // nWindows
    cdef ITYPE_t i, j, idx, nb_nan, new_N

    cdef double[:] coeffs
    # for i in range(N):
    #     obs_view[i] = obs_flux[i]  # simple fast copy
    #     mod_view[i] = model_flux[i]  # simple fast copy
    #     wvl_view[i] = wvl[i]  # simple fast copy
    ## Count NaNs:
    nb_nan = 0
    for i in range(N):
        if obs_flux[i]!=obs_flux[i]: ## Is a NaN
            nb_nan+=1
    new_N = N-nb_nan    

    ## New arrays with only non-NaN elements:
    cdef double[:] wvl_view = np.empty(new_N, dtype=np.float64)
    cdef double[:] residuals = np.empty(new_N, dtype=np.float64)
    cdef double[:] wvl_norm = np.empty(new_N)
    cdef np.ndarray[DTYPE_t, ndim=1] continuum = np.empty(new_N)
    j = 0
    for i in range(N):
        if not obs_flux[i]!=obs_flux[i]:
            wvl_view[j] = wvl[i]
            # obs_view[j]=obs_flux[i]
            residuals[j]=model_flux[i]/obs_flux[i]
            j+=1
            
    wvl_norm = normalize_axis_cy(wvl_view, wvl_view)
    coeffs = fit_1d_polynomial_cy(wvl_norm, residuals, degree)

    continuum = poly1d_horner(wvl_norm, coeffs[::-1])

    return continuum



# ---- Main function ----
@cython.boundscheck(False)
@cython.wraparound(False)
def adjust_continuum7_fast_inplace(
    double[:] wvl,
    double[:] obs_flux,
    int nbins = 5,
    ):
    '''Now in this version I want to obtain a smooth continuum by rejecting
    the spectral lines'''

    cdef ITYPE_t N = obs_flux.shape[0]
    cdef ITYPE_t i, nb, idx, nb_nan, new_N

    cdef double sumx
    # for i in range(N):
    #     obs_view[i] = obs_flux[i]  # simple fast copy
    #     mod_view[i] = model_flux[i]  # simple fast copy
    #     wvl_view[i] = wvl[i]  # simple fast copy
    ## Count NaNs:
    nb_nan = 0
    for i in range(N):
        if obs_flux[i]!=obs_flux[i]: ## Is a NaN
            nb_nan+=1
    new_N = N-nb_nan    

    ## New arrays with only non-NaN elements:
    cdef double[:] residuals = np.empty(new_N, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] continuum = np.empty(new_N)
    nb = 0
    sumx = 0.
    for i in range(2*nbins):
        if obs_flux[i]==obs_flux[i]: ## is not a NaN
            sumx+=obs_flux[i]
            nb+=1
            residuals[i]=0.
        if i<nbins:
            continuum[i] = 1.
    for i in range(nbins, N-nbins):
        if obs_flux[i-nbins]==obs_flux[i-nbins]:
            sumx-=obs_flux[i-nbins] ## Remove the previous
            nb-=1
        if obs_flux[i+nbins]==obs_flux[i+nbins]:
            sumx+=obs_flux[i+nbins]
            nb+=1
        residuals[i] = sumx/nb
        continuum[i] = residuals[i]
    for i in range(N-nbins, N):
        continuum[i]=1.

    return continuum

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def adjust_continuum7_fast_inplace_opt(
    double[:] obs_flux,
    int nbins=5,
):
    cdef Py_ssize_t N = obs_flux.shape[0]
    cdef Py_ssize_t i, left, right
    cdef double val

    cdef double[:] continuum = np.ones(N, dtype=np.float64)
    cdef double[:] cumsum_flux = np.zeros(N + 1, dtype=np.float64)
    cdef int[:] cumsum_count = np.zeros(N + 1, dtype=np.int32)

    # ---- Build prefix sums ----
    for i in range(N):
        val = obs_flux[i]

        if val == val:  # not NaN
            cumsum_flux[i+1] = cumsum_flux[i] + val
            cumsum_count[i+1] = cumsum_count[i] + 1
        else:
            cumsum_flux[i+1] = cumsum_flux[i]
            cumsum_count[i+1] = cumsum_count[i]

    # ---- Compute window means ----
    for i in range(nbins, N - nbins):
        left  = i - nbins
        right = i + nbins + 1

        val = cumsum_count[right] - cumsum_count[left]

        if val > 0:
            continuum[i] = (
                cumsum_flux[right] - cumsum_flux[left]
            ) / val

    return np.asarray(continuum)
