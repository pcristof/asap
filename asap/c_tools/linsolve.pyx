# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs
from scipy.linalg import lapack

ctypedef np.float64_t DTYPE_t
ctypedef Py_ssize_t ITYPE_t

# ---- Gaussian elimination for small matrices ----
cdef double[::1] _gauss_elim(DTYPE_t[:, ::1] mat, DTYPE_t[::1] Y):
    cdef ITYPE_t nrows = mat.shape[0]
    cdef ITYPE_t ncols = mat.shape[1]
    cdef ITYPE_t i, j, k
    cdef double[:, ::1] subcov = cython.view.array(shape=(ncols, ncols), itemsize=8, format="d")
    cdef double[::1] rhs = cython.view.array(shape=(ncols,), itemsize=8, format="d")
    cdef double[::1] coeffs = cython.view.array(shape=(ncols,), itemsize=8, format="d")
    cdef double factor, tmp
    cdef ITYPE_t max_row
    cdef double max_val

    # mat.T @ mat
    for i in range(ncols):
        for j in range(ncols):
            subcov[i, j] = 0.0
            for k in range(nrows):
                subcov[i, j] += mat[k, i] * mat[k, j]

    # mat.T @ Y
    for i in range(ncols):
        rhs[i] = 0.0
        for k in range(nrows):
            rhs[i] += mat[k, i] * Y[k]

    # Gaussian elimination with partial pivoting
    for i in range(ncols):
        max_row = i
        max_val = fabs(subcov[i, i])
        for j in range(i + 1, ncols):
            if fabs(subcov[j, i]) > max_val:
                max_val = fabs(subcov[j, i])
                max_row = j
        if max_row != i:
            # Swap rows
            for k in range(ncols):
                tmp = subcov[i, k]
                subcov[i, k] = subcov[max_row, k]
                subcov[max_row, k] = tmp
            tmp = rhs[i]
            rhs[i] = rhs[max_row]
            rhs[max_row] = tmp

        # Eliminate below
        for j in range(i + 1, ncols):
            factor = subcov[j, i] / subcov[i, i]
            for k in range(i, ncols):
                subcov[j, k] -= factor * subcov[i, k]
            rhs[j] -= factor * rhs[i]

    # Back-substitution
    for i in range(ncols - 1, -1, -1):
        tmp = rhs[i]
        for j in range(i + 1, ncols):
            tmp -= subcov[i, j] * coeffs[j]
        coeffs[i] = tmp / subcov[i, i]

    return coeffs

# ---- Hybrid solver ----
def hybrid_linsolve_nocov(np.ndarray[DTYPE_t, ndim=2] mat,
                          np.ndarray[DTYPE_t, ndim=1] Y):
    """
    Solve coeffs = (mat.T @ mat)^(-1) @ mat.T @ Y
    Automatically picks:
    - Cython Gaussian elimination for ncols < 50
    - LAPACK dgesv for ncols >= 50
    """
    cdef ITYPE_t ncols = mat.shape[1]

    if ncols < 50:
        # Fast pure Cython memoryview path
        return _gauss_elim(mat, Y)
    else:
        # LAPACK path for larger matrices
        # Form normal equations: mat.T @ mat @ coeffs = mat.T @ Y
        M = mat.T @ mat
        b = mat.T @ Y
        # Solve using LAPACK dgesv
        x, info = lapack.dgesv(M, b)[:2]
        if info != 0:
            raise np.linalg.LinAlgError(f"LAPACK dgesv failed with info={info}")
        return x