# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

## Cython package for the interpolation. This is not faster than the numba
## function, but may help convert the rest of the core code to Cython.

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log, log10, exp, pow

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

## This is a simple interpolation function
cdef inline double interp(
    double x, double x0, double x1,
    double y0, double y1
) nogil:
    if x1 == x0:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

## Function to replace np.where.
## Assumes sorted grid
cdef inline ITYPE_t find_low(double x, double[:] array):
    cdef ITYPE_t i, n = array.shape[0]
    if n==1:
        return 0 ## Only one value in array
    elif array[0]>=x:
        return 0 ## First element matches request
    elif array[n-1]<=x:
        return n-1 ## Last element matches request
    else: ## Search for best match
        for i in range(n - 1):
            if array[i+1] > x:
                return i
    # return n - 2

## Find bounds
## Assumes sorted array
## The function returns the lower and upper indices
# cdef inline ITYPE_t find_low(double x, double[:] array):
#     cdef ITYPE_t i, n = array.shape[0]
cdef inline void find_bounds(
        double x,
        double[:] array,
        ITYPE_t* idxlow,
        ITYPE_t* idxhigh
    ) nogil:
    cdef ITYPE_t i, n = array.shape[0]
    if n==1:
        idxlow[0] = 0 ## Only one value in array
        idxhigh[0] = 0 ## Only one value in array
        return
    elif array[0]>x: ## Boundary request lower than bound
        idxlow[0] = 0 
        idxhigh[0] = 0
        return
    elif array[n-1]<x: ## Boundary request higher than bound
        idxlow[0] = n-1  
        idxhigh[0] = n-1
        return
    else: ## Search for best match
        for i in range(n - 1):
            if array[i]==x: ## Exact match
                idxlow[0] = i
                idxhigh[0] = i
                return
            elif array[i+1]==x: ## Exact match next iteration
                idxlow[0] = i+1
                idxhigh[0] = i+1
                return
            elif array[i+1] > x:
                idxlow[0] = i
                idxhigh[0] = i+1
                return
    # return n - 2

def interpolate_4d(
    double teff,
    double logg,
    double mh,
    double alpha,
    double[:] teffs,
    double[:] loggs,
    double[:] mhs,
    double[:] alphas,
    double[:, :, :, :, :] spectra,
    int mode
    ):
    """
    mode:
        0 = linear
        1 = log
        2 = log10
    """

    cdef ITYPE_t it0, it1, il0, il1, im0, im1, ia0, ia1
    cdef double tl, th, ll, lh, ml, mh_, al, ah
    cdef ITYPE_t r, nlam = spectra.shape[4]

    cdef double v0000, v0001, v0010, v0011
    cdef double v0100, v0101
    cdef double v1000, v1001
    cdef double v0110, v0111
    cdef double v1100, v1101
    cdef double v1010, v1011
    cdef double v1110, v1111

    cdef double a00, a01, a10, a11
    cdef double b00, b01, b10, b11
    cdef double c0, c1, c2, c3
    cdef double d0, d1
    cdef double s

    cdef np.ndarray[np.float64_t, ndim=1] out

    cdef double wt
    cdef double wl
    cdef double wm
    cdef double wa


    # --- indices
    find_bounds(teff, teffs, &it0, &it1)
    # it1 = it0 + 1
    find_bounds(logg, loggs, &il0, &il1)
    # il1 = il0 + 1
    find_bounds(mh, mhs, &im0, &im1)
    # im1 = im0 + 1
    find_bounds(alpha, alphas, &ia0, &ia1)
    # ia1 = ia0 + 1

    tl = teffs[it0]; th = teffs[it1]
    ll = loggs[il0]; lh = loggs[il1]
    ml = mhs[im0];  mh_ = mhs[im1]
    al = alphas[ia0]; ah = alphas[ia1]

    out = np.empty(nlam)
    # cdef double s

    ## Precompute the weights for interpolation; same for all pixels.
    if th==tl: wt=0.
    else: wt = (teff  - tl) / (th - tl)
    if (lh==ll): wl=0
    else: wl = (logg  - ll) / (lh - ll)
    if (mh_==ml): wm=0
    else: wm = (mh    - ml) / (mh_ - ml)
    if (ah==al): wa=0
    else: wa = (alpha - al) / (ah - al)
    
    for r in range(nlam): ## Loop over lambda first (no arrays handling after
                          ## this point.)

        # --- fetch values
        v0000 = spectra[it0, il0, im0, ia0, r]
        v0001 = spectra[it0, il0, im0, ia1, r]
        v0010 = spectra[it0, il0, im1, ia0, r]
        v0011 = spectra[it0, il0, im1, ia1, r]
        v0100 = spectra[it0, il1, im0, ia0, r]
        v0101 = spectra[it0, il1, im0, ia1, r]
        v1000 = spectra[it1, il0, im0, ia0, r]
        v1001 = spectra[it1, il0, im0, ia1, r]
        v0110 = spectra[it0, il1, im1, ia0, r]
        v0111 = spectra[it0, il1, im1, ia1, r]
        v1100 = spectra[it1, il1, im0, ia0, r]
        v1101 = spectra[it1, il1, im0, ia1, r]
        v1010 = spectra[it1, il0, im1, ia0, r]
        v1011 = spectra[it1, il0, im1, ia1, r]
        v1110 = spectra[it1, il1, im1, ia0, r]
        v1111 = spectra[it1, il1, im1, ia1, r]

        if mode == 1:
            v0000 = log(v0000); v0001 = log(v0001)
            v0010 = log(v0010); v0011 = log(v0011)
            v0100 = log(v0100); v0101 = log(v0101)
            v1000 = log(v1000); v1001 = log(v1001)
            v0110 = log(v0110); v0111 = log(v0111)
            v1100 = log(v1100); v1101 = log(v1101)
            v1010 = log(v1010); v1011 = log(v1011)
            v1110 = log(v1110); v1111 = log(v1111)
        elif mode == 2:
            v0000 = log10(v0000); v0001 = log10(v0001)
            v0010 = log10(v0010); v0011 = log10(v0011)
            v0100 = log10(v0100); v0101 = log10(v0101)
            v1000 = log10(v1000); v1001 = log10(v1001)
            v0110 = log10(v0110); v0111 = log10(v0111)
            v1100 = log10(v1100); v1101 = log10(v1101)
            v1010 = log10(v1010); v1011 = log10(v1011)
            v1110 = log10(v1110); v1111 = log10(v1111)

        # alpha
        # a00 = interp(alpha, al, ah, v0000, v0001)
        # a01 = interp(alpha, al, ah, v0010, v0011)
        # a10 = interp(alpha, al, ah, v0100, v0101)
        # a11 = interp(alpha, al, ah, v0110, v0111)
        # b00 = interp(alpha, al, ah, v1000, v1001)
        # b01 = interp(alpha, al, ah, v1010, v1011)
        # b10 = interp(alpha, al, ah, v1100, v1101)
        # b11 = interp(alpha, al, ah, v1110, v1111)

        a00 = v0000 + wa*(v0001-v0000)
        a01 = v0010 + wa*(v0011-v0010)
        a10 = v0100 + wa*(v0101-v0100)
        a11 = v0110 + wa*(v0111-v0110)
        b00 = v1000 + wa*(v1001-v1000)
        b01 = v1010 + wa*(v1011-v1010)
        b10 = v1100 + wa*(v1101-v1100)
        b11 = v1110 + wa*(v1111-v1110)

        # teff
        # c0 = interp(teff, tl, th, a00, b00)
        # c1 = interp(teff, tl, th, a01, b01)
        # c2 = interp(teff, tl, th, a10, b10)
        # c3 = interp(teff, tl, th, a11, b11)
        c0 = a00 + wt*(b00-a00)
        c1 = a01 + wt*(b01-a01)
        c2 = a10 + wt*(b10-a10)
        c3 = a11 + wt*(b11-a11)

        # logg
        # d0 = interp(logg, ll, lh, c0, c2)
        # d1 = interp(logg, ll, lh, c1, c3)
        d0 = c0 + wl*(c2-c0)
        d1 = c1 + wl*(c3-c1)

        # mh
        # s = interp(mh, ml, mh_, d0, d1)
        s = d0 + wm*(d1-d0)

        if mode == 1:
            out[r] = exp(s)
        elif mode == 2:
            out[r] = pow(10.0, s)
        else:
            out[r] = s

    return (teff, logg, mh), out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] interpolate_4d_opt(
    ITYPE_t it0,
    ITYPE_t it1,
    ITYPE_t il0,
    ITYPE_t il1,
    ITYPE_t im0,
    ITYPE_t im1,
    ITYPE_t ia0,
    ITYPE_t ia1,
    double wt,
    double wl,
    double wm,
    double wa,
    double[:, :, :, :, :] spectra,
    int mode
    ):
    """
    mode:
        0 = linear
        1 = log
        2 = log10
    """

    # cdef ITYPE_t it0, it1, il0, il1, im0, im1, ia0, ia1
    # cdef double tl, th, ll, lh, ml, mh_, al, ah
    cdef ITYPE_t r, nlam = spectra.shape[4]

    cdef double v0000, v0001, v0010, v0011
    cdef double v0100, v0101
    cdef double v1000, v1001
    cdef double v0110, v0111
    cdef double v1100, v1101
    cdef double v1010, v1011
    cdef double v1110, v1111

    cdef double a00, a01, a10, a11
    cdef double b00, b01, b10, b11
    cdef double c0, c1, c2, c3
    cdef double d0, d1
    cdef double s

    cdef np.ndarray[np.float64_t, ndim=1] out

    # cdef double wt
    # cdef double wl
    # cdef double wm
    # cdef double wa

    out = np.empty(nlam)

    for r in range(nlam): ## Loop over lambda first (no arrays handling after
                          ## this point.)

        # --- fetch values
        v0000 = spectra[it0, il0, im0, ia0, r]
        v0001 = spectra[it0, il0, im0, ia1, r]
        v0010 = spectra[it0, il0, im1, ia0, r]
        v0011 = spectra[it0, il0, im1, ia1, r]
        v0100 = spectra[it0, il1, im0, ia0, r]
        v0101 = spectra[it0, il1, im0, ia1, r]
        v1000 = spectra[it1, il0, im0, ia0, r]
        v1001 = spectra[it1, il0, im0, ia1, r]
        v0110 = spectra[it0, il1, im1, ia0, r]
        v0111 = spectra[it0, il1, im1, ia1, r]
        v1100 = spectra[it1, il1, im0, ia0, r]
        v1101 = spectra[it1, il1, im0, ia1, r]
        v1010 = spectra[it1, il0, im1, ia0, r]
        v1011 = spectra[it1, il0, im1, ia1, r]
        v1110 = spectra[it1, il1, im1, ia0, r]
        v1111 = spectra[it1, il1, im1, ia1, r]

        if mode == 1:
            v0000 = log(v0000); v0001 = log(v0001)
            v0010 = log(v0010); v0011 = log(v0011)
            v0100 = log(v0100); v0101 = log(v0101)
            v1000 = log(v1000); v1001 = log(v1001)
            v0110 = log(v0110); v0111 = log(v0111)
            v1100 = log(v1100); v1101 = log(v1101)
            v1010 = log(v1010); v1011 = log(v1011)
            v1110 = log(v1110); v1111 = log(v1111)
        elif mode == 2:
            v0000 = log10(v0000); v0001 = log10(v0001)
            v0010 = log10(v0010); v0011 = log10(v0011)
            v0100 = log10(v0100); v0101 = log10(v0101)
            v1000 = log10(v1000); v1001 = log10(v1001)
            v0110 = log10(v0110); v0111 = log10(v0111)
            v1100 = log10(v1100); v1101 = log10(v1101)
            v1010 = log10(v1010); v1011 = log10(v1011)
            v1110 = log10(v1110); v1111 = log10(v1111)

        # alpha
        a00 = v0000 + wa*(v0001-v0000)
        a01 = v0010 + wa*(v0011-v0010)
        a10 = v0100 + wa*(v0101-v0100)
        a11 = v0110 + wa*(v0111-v0110)
        b00 = v1000 + wa*(v1001-v1000)
        b01 = v1010 + wa*(v1011-v1010)
        b10 = v1100 + wa*(v1101-v1100)
        b11 = v1110 + wa*(v1111-v1110)

        # teff
        c0 = a00 + wt*(b00-a00)
        c1 = a01 + wt*(b01-a01)
        c2 = a10 + wt*(b10-a10)
        c3 = a11 + wt*(b11-a11)

        # logg
        d0 = c0 + wl*(c2-c0)
        d1 = c1 + wl*(c3-c1)

        # mh
        s = d0 + wm*(d1-d0)

        if mode == 1:
            out[r] = exp(s)
        elif mode == 2:
            out[r] = pow(10.0, s)
        else:
            out[r] = s

    return out


def wrap_interpolate_4d(
    double teff,
    double logg,
    double mh,
    double alpha,
    double[:] teffs,
    double[:] loggs,
    double[:] mhs,
    double[:] alphas,
    double[:, :, :, :, :, :] spectra_arr,
    int mode
    ):
    """
    Call wrap_function_fine_linear_4d for each spectra in a list.

    Parameters
    ----------
    teffs, loggs, mhs, alphas : 1D arrays
        Grid values
    spectra_list : list
        List of 5D arrays: (nteff, nlogg, nmh, nalpha, nlambda)
    teff, logg, mh, alpha : float
        Target parameters for interpolation
    mode : str
        'linear', 'log', 'log10'

    Returns
    -------
    results : list
        Each element is the interpolated spectrum
    """
    cdef ITYPE_t r, nreg = spectra_arr.shape[4] ## number of regions
    cdef ITYPE_t l, nlam = spectra_arr.shape[5] ## number of regions
    cdef np.ndarray[np.float64_t, ndim=2] out
    cdef np.ndarray[np.float64_t, ndim=1] interp_spec

    out = np.empty((nreg, nlam))

    for l in range(nreg):
        # Call the fast Cython function
        _, interp_spec = interpolate_4d(
            teff, logg, mh, alpha,
            teffs, loggs, mhs, alphas,
            spectra_arr[:,:,:,:,l,:],
            0
        )
        out[l] = interp_spec
    return (teff, logg, mh), out

def wrap_interpolate_4d_opt(
    double teff,
    double logg,
    double mh,
    double alpha,
    double[:] teffs,
    double[:] loggs,
    double[:] mhs,
    double[:] alphas,
    double[:, :, :, :, :, :] spectra_arr,
    int mode
    ):
    """
    Call wrap_function_fine_linear_4d for each spectra in a list.

    Parameters
    ----------
    teffs, loggs, mhs, alphas : 1D arrays
        Grid values
    spectra_list : list
        List of 5D arrays: (nteff, nlogg, nmh, nalpha, nlambda)
    teff, logg, mh, alpha : float
        Target parameters for interpolation
    mode : str
        'linear', 'log', 'log10'

    Returns
    -------
    results : list
        Each element is the interpolated spectrum
    """
    cdef ITYPE_t r, nreg = spectra_arr.shape[4] ## number of regions
    cdef ITYPE_t l, nlam = spectra_arr.shape[5] ## number of regions
    cdef np.ndarray[np.float64_t, ndim=2] out
    cdef np.ndarray[np.float64_t, ndim=1] interp_spec

    cdef double wt
    cdef double wl
    cdef double wm
    cdef double wa
    cdef ITYPE_t it0, it1, il0, il1, im0, im1, ia0, ia1
    cdef double tl, th, ll, lh, ml, mh_, al, ah

    # --- indices
    find_bounds(teff, teffs, &it0, &it1)
    # it1 = it0 + 1
    find_bounds(logg, loggs, &il0, &il1)
    # il1 = il0 + 1
    find_bounds(mh, mhs, &im0, &im1)
    # im1 = im0 + 1
    find_bounds(alpha, alphas, &ia0, &ia1)
    # ia1 = ia0 + 1

    tl = teffs[it0]; th = teffs[it1]
    ll = loggs[il0]; lh = loggs[il1]
    ml = mhs[im0];  mh_ = mhs[im1]
    al = alphas[ia0]; ah = alphas[ia1]

    ## Precompute the weights for interpolation; same for all pixels.
    if th==tl: wt=0.
    else: wt = (teff  - tl) / (th - tl)
    if (lh==ll): wl=0
    else: wl = (logg  - ll) / (lh - ll)
    if (mh_==ml): wm=0
    else: wm = (mh    - ml) / (mh_ - ml)
    if (ah==al): wa=0
    else: wa = (alpha - al) / (ah - al)

    out = np.empty((nreg, nlam))


    for l in range(nreg):
        # Call the fast Cython function
        # interp_spec = interpolate_4d_opt(
        #     it0, it1, il0, il1, im0, im1, ia0, ia1,
        #     wt, wl, wm, wa,
        #     spectra_arr[:,:,:,:,l,:],
        #     mode
        # )
        _, interp_spec = interpolate_4d(
            teff, logg, mh, alpha,
            teffs, loggs, mhs, alphas,
            spectra_arr[:,:,:,:,l,:],
            0
        )
        out[l] = interp_spec
    return (teff, logg, mh), out