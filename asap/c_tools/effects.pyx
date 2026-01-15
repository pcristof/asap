# fast_convolve_naive.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp, fabs, floor, pi
from libc.math cimport M_PI
from libc.math cimport pi, fabs, round, log


# Convert FWHM to sigma
cdef double FWHM_to_sigma(double fwhm):
    return fwhm / (2.0 * sqrt(2.0 * log(2.0)))

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_1d_full(double[:] signal, double[:] kernel):
    """
    Fast FULL convolution: len = n + m - 1
    """
    cdef:
        Py_ssize_t n = signal.shape[0]
        Py_ssize_t m = kernel.shape[0]
        Py_ssize_t size = n + m - 1
        Py_ssize_t i, j

    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros(size, dtype=np.float64)

    cdef double* s = &signal[0]
    cdef double* k = &kernel[0]
    cdef double* o = &out[0]

    # Loop over kernel (usually shorter → better cache)
    for j in range(m):
        for i in range(n):
            o[i + j] += s[i] * k[j]

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def convolve_1d_same(double[:] signal, double[:] kernel):
    """
    Fast SAME convolution
    Output length = len(signal)
    """
    cdef:
        Py_ssize_t n = signal.shape[0]
        Py_ssize_t m = kernel.shape[0]
        Py_ssize_t half = m // 2
        Py_ssize_t i, j, jj

    cdef np.ndarray[np.float64_t, ndim=1] out = np.zeros(n, dtype=np.float64)

    cdef double* s = &signal[0]
    cdef double* k = &kernel[0]
    cdef double* o = &out[0]

    for i in range(n):
        for j in range(m):
            jj = i + j - half
            if 0 <= jj < n:
                o[i] += s[jj] * k[j]

    return out

# lsf_rotate_cy.pyx
@cython.boundscheck(False)
@cython.wraparound(False)
def lsf_rotate_cy(double deltav, double vsini, double epsilon=0.6):
    """
    Fast pure-Cython rotational broadening kernel
    Returns: (velocity grid, kernel)
    """

    cdef:
        Py_ssize_t npts, i
        Py_ssize_t nwid
        double inv_vsini
        double xi, xnorm
        double e1, e2, e3
        double* xptr
        double* kptr

    # --- number of points (force odd) ---
    npts = <Py_ssize_t>floor(2.0 * vsini / deltav)
    if (npts & 1) == 0:
        npts += 1

    nwid = npts >> 1
    inv_vsini = 1.0 / vsini

    # --- allocate arrays ---
    cdef np.ndarray[np.float64_t, ndim=1] x = np.empty(npts, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] ker = np.empty(npts, dtype=np.float64)

    xptr = <double*>x.data
    kptr = <double*>ker.data

    # --- constants ---
    e1 = 2.0 * (1.0 - epsilon)
    e2 = M_PI * epsilon * 0.5
    e3 = M_PI * (1.0 - epsilon / 3.0) * vsini

    # --- main loop ---
    for i in range(npts):
        xi = <double>(i - nwid)
        xptr[i] = xi * deltav
        xnorm = xptr[i] * inv_vsini ## This is half of x1.

        if fabs(xnorm) >= 1.0: ## To ensure "absolute"
            kptr[i] = (e1 * sqrt(-(1.0 - xnorm * xnorm))
                       + e2 * (-(1.0 - xnorm * xnorm))) / e3
        else:
            kptr[i] = (e1 * sqrt(1.0 - xnorm * xnorm)
                       + e2 * (1.0 - xnorm * xnorm)) / e3

    return x, ker

# gaussian_cy.pyx

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float64_t, ndim=1] gaussian_cy(double sigma):
    """
    Ultra-fast Gaussian kernel builder (C-only).
    Unnormalized on purpose (normalize after).
    """

    cdef int n, i, mid
    cdef double inv_2sig2, x
    cdef np.ndarray[np.float64_t, ndim=1] g_arr
    cdef double[:] g

    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)

    # Kernel size (odd)
    n = <int>(17.0 * sigma)
    if not (n & 1):
        n -= 1

    mid = n >> 1
    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    g_arr = np.empty(n, dtype=np.float64)
    g = g_arr

    for i in range(n):
        x = i - mid
        g[i] = exp(-x * x * inv_2sig2)

    return g_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void pad_1d(double[:] arr_in, double[:] arr_out, int pad_left, int pad_right, double value=0.0):
    """
    arr_in: input array
    arr_out: preallocated output array of size arr_in.size + pad_left + pad_right
    pad_left, pad_right: number of elements to pad
    value: value to pad with
    """
    cdef int i, n = arr_in.shape[0]
    
    # Left padding
    for i in range(pad_left):
        arr_out[i] = value
    
    # Copy original array
    for i in range(n):
        arr_out[pad_left + i] = arr_in[i]
    
    # Right padding
    for i in range(pad_right):
        arr_out[pad_left + n + i] = value

# rt_macro_cy.pyx

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray _rt_macro_cy(double velocity_step,
                             double vmac,
                             double m):
    """
    Pure Cython radial-tangential macroturbulence kernel.
    Uses memoryviews internally for speed.
    Returns a standard NumPy array (callable from Python).
    """

    cdef double sigma, sigr, sigt
    cdef Py_ssize_t nmk, nk, i
    cdef double xi, sum_r, sum_t
    cdef double area_r = 0.5, area_t = 0.5

    # --- projected sigmas ---
    sigma = vmac / (sqrt(2.0) * velocity_step)
    sigr = sigma * m
    sigt = sigma * sqrt(1.0 - m*m)

    # --- number of points ---
    nmk = <Py_ssize_t>(sigma * 10.0 + 0.5)
    if nmk < 3:
        nmk = 3
    nk = 2 * nmk + 1

    # --- allocate arrays ---
    cdef np.ndarray[np.float64_t, ndim=1] mrkern = np.zeros(nk, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] mtkern = np.zeros(nk, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] mkern = np.empty(nk, dtype=np.float64)

    # get typed memoryviews
    cdef double[:] mr = mrkern
    cdef double[:] mt = mtkern
    cdef double[:] out = mkern

    # --- radial kernel ---
    sum_r = 0.0
    if sigr > 0.0:
        for i in range(nk):
            xi = (i - nmk) / sigr
            mr[i] = exp(-0.5 * xi * xi)
            sum_r += mr[i]
        for i in range(nk):
            mr[i] /= sum_r
    else:
        mr[nmk] = 1.0

    # --- tangential kernel ---
    sum_t = 0.0
    if sigt > 0.0:
        for i in range(nk):
            xi = (i - nmk) / sigt
            mt[i] = exp(-0.5 * xi * xi)
            sum_t += mt[i]
        for i in range(nk):
            mt[i] /= sum_t
    else:
        mt[nmk] = 1.0

    # --- combine radial and tangential contributions ---
    for i in range(nk):
        out[i] = area_r * mr[i] + area_t * mt[i]

    return mkern


@cython.boundscheck(False)
@cython.wraparound(False)
def rt_macro_cy(double velocity_step, double vmac):
    """
    Optimized pure Cython version of rt_macro.
    Computes disk-integrated radial-tangential macroturbulence kernel.
    Returns a NumPy array of type float64.
    """
    cdef int itot = 9
    cdef int dski, ju, count
    cdef double dskri, ritot = itot, dskr, area, scale
    cdef int nk
    cdef np.ndarray[np.float64_t, ndim=1] kernel
    cdef double[:] subkern
    cdef double[:] kview, skview
    cdef int i

    # --- initialize kernel with disk center (mu=0) ---
    kernel = _rt_macro_cy(velocity_step, vmac, 0.0)
    kview = kernel
    nk = kernel.shape[0]
    count = 0

    # --- loop over annuli ---
    for dski in range(1, 10):
        ju = 6 * dski
        dskri = dski
        dskr = (dskri - 0.5) / ritot
        area = 3.141592653589793 * (2.0 * dskri - 1.0) / (6.0 * dskri * (ritot * ritot))
        scale = ju / area

        # compute subkernel for this annulus
        subkern = _rt_macro_cy(velocity_step, vmac, fabs(dskr))
        skview = subkern

        # --- add scaled subkernel elementwise using memoryviews ---
        for i in range(nk):
            kview[i] += skview[i] * scale

        count += ju

    # --- normalize final kernel ---
    for i in range(nk):
        kview[i] /= count

    return kernel


@cython.boundscheck(False)
@cython.wraparound(False)
def broaden_spectrum_2_cy(np.ndarray[np.float64_t, ndim=1] wvl,
                           np.ndarray[np.float64_t, ndim=1] flux,
                           double vinstru=0.0,
                           double vsini=0.0,
                           double epsilon=0.6,
                           double vmac=0.0,
                           vmac_mode='g'):
    """
    Fast Cython version of broaden_spectrum_2
    Returns np.ndarray same length as flux
    """

    cdef int n = flux.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] output = flux.copy()
    cdef double[:] oflux = output  # memoryview for speed

    cdef int pad_left
    cdef int pad_right
    cdef np.ndarray[np.float64_t, ndim=1] gauss_kernel
    cdef np.ndarray[np.float64_t, ndim=1] rot_kernel
    cdef np.ndarray[np.float64_t, ndim=1] rtkernel
    cdef np.ndarray[np.float64_t, ndim=1] masterkernel
    cdef int mid
    cdef double vel_step
    cdef double sigma_gauss
    cdef double vmacGauss, vmacRT
    cdef int len_master

    # --- Determine vmac mode ---
    vmacMode = vmac_mode.lower()
    if vmacMode in ('gaussian', 'g'):
        vmacGauss = vmac
        vmacRT = 0.0
    elif vmacMode in ('radial-tangential', 'radial tangeantial', 'rt'):
        vmacGauss = 0.0
        vmacRT = vmac
    else:
        vmacGauss = vmac
        vmacRT = 0.0

    # --- Velocity step (km/s) ---
    mid = n // 2
    vel_step = (wvl[mid] - wvl[mid-1]) / ((wvl[mid] + wvl[mid-1])/2.0) * 2.99792458e5

    # --- Convert FWHM to sigma ---
    vmacGauss = FWHM_to_sigma(vmacGauss)
    vinstru = FWHM_to_sigma(vinstru)

    # --- Gaussian sigma in pixels ---
    sigma_gauss = sqrt(vmacGauss**2 + vinstru**2) / vel_step

    # --- Gaussian kernel ---
    gauss_kernel = None
    if sigma_gauss > 0.0:
        gauss_kernel = gaussian_cy(sigma_gauss)
        gauss_kernel /= gauss_kernel.sum()

    # --- Rotation kernel ---
    rot_kernel = None
    if vsini != 0.0:
        if vsini < 0.:
            vsini = -vsini
        _, rot_kernel = lsf_rotate_cy(vel_step, vsini, epsilon=epsilon)
        rot_kernel /= rot_kernel.sum()

    # --- RT kernel ---
    rtkernel = None
    if vmacRT > 0:
        rtkernel = rt_macro_cy(vel_step, vmacRT)
        rtkernel /= rtkernel.sum()

    # --- Build master kernel (full) ---
    masterkernel = np.array([1.0], dtype=np.float64)
    if gauss_kernel is not None:
        masterkernel = np.convolve(masterkernel, gauss_kernel, mode='full')
    if rot_kernel is not None:
        masterkernel = np.convolve(masterkernel, rot_kernel, mode='full')
    if rtkernel is not None:
        masterkernel = np.convolve(masterkernel, rtkernel, mode='full')

    # --- Convolve spectrum in 'same' mode ---
    len_master = masterkernel.shape[0]
    if n < len_master:
        pad_left = (len_master - 1) // 2
        pad_right = len_master - 1 - pad_left
        padded = np.empty(n + pad_left + pad_right, dtype=np.float64)
        pad_1d(oflux, padded, pad_left, pad_right, 1.0)
        padded = np.convolve(padded, masterkernel, mode='same')
        output[:] = padded#[pad_left:pad_left+n]
    else:
        output[:] = np.convolve(oflux, masterkernel, mode='same')#[(len_master-1)//2:(len_master-1)//2+n]

    return output



# ######## ------_____________________
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def broaden_spectrum_2_cy(np.ndarray[np.float64_t, ndim=1] wvl,
#                            np.ndarray[np.float64_t, ndim=1] flux,
#                            double vinstru=0.0,
#                            double vsini=0.0,
#                            double epsilon=0.6,
#                            double vmac=0.0,
#                            vmac_mode='g'):
#     """
#     Cython version of broaden_spectrum_2
#     """

#     cdef int n = flux.shape[0]
#     cdef np.ndarray[np.float64_t, ndim=1] output = flux.copy()  # real ndarray
#     cdef double[:] oflux = output
#     cdef double[:] s = oflux
#     cdef double[:] oflux_tmp
#     cdef double[:] oflux_tmp2 = np.empty(n)
#     # Determine vmac mode
#     cdef double vmacGauss, vmacRT

#     vmacMode = vmac_mode.lower()
#     if vmacMode == 'gaussian' or vmacMode == 'g':
#         vmacRT = 0.0
#         vmacGauss = vmac
#     elif vmacMode == 'radial-tangential' or vmacMode == 'radial tangeantial' \
#         or vmacMode == 'rt':
#         vmacRT = vmac
#         vmacGauss = 0.0
#     else:
#         vmacRT = 0.0
#         vmacGauss = vmac

#     # Velocity step in km/s
#     cdef double vel_step
#     cdef int mid = n // 2
#     vel_step = (wvl[mid] - wvl[mid-1]) / ((wvl[mid] + wvl[mid-1]) / 2.0) * 2.99792458e5

#     # Convert FWHM to sigma
#     vmacGauss = FWHM_to_sigma(vmacGauss)
#     vinstru = FWHM_to_sigma(vinstru)

#     # Quadratic sum
#     cdef double vgauss = sqrt(vmacGauss**2 + vinstru**2)
#     cdef double sigma_gauss = vgauss / vel_step

#     # Gaussian kernel
#     cdef np.ndarray[np.float64_t, ndim=1] gauss_kernel = None
#     cdef int len_gauss_kernel = 0
#     if sigma_gauss > 0.0:
#         gauss_kernel = gaussian_cy(sigma_gauss)   # must return np.ndarray[float64_t]
#         gauss_kernel /= gauss_kernel.sum()
#         len_gauss_kernel = gauss_kernel.shape[0]

#     # Rotation kernel
#     cdef np.ndarray[np.float64_t, ndim=1] rot_kernel = None
#     cdef int len_rot_kernel = 0
#     if vsini > 0:
#         if vsini < 0: vsini = -vsini
#         _, rot_kernel = lsf_rotate_cy(vel_step, vsini, epsilon=epsilon)
#         rot_kernel /= rot_kernel.sum()
#         len_rot_kernel = rot_kernel.shape[0]

#     # RT kernel
#     cdef np.ndarray[np.float64_t, ndim=1] rtkernel = None
#     cdef int len_rtkernel = 0
#     if vmacRT > 0:
#         rtkernel = rt_macro_cy(vel_step, vmacRT)
#         rtkernel /= rtkernel.sum()
#         len_rtkernel = rtkernel.shape[0]

#     # Master kernel convolution
#     cdef np.ndarray[np.float64_t, ndim=1] masterkernel = np.array([1.0], dtype=np.float64)

#     if gauss_kernel is not None:
#         masterkernel = convolve_1d_cy(masterkernel, gauss_kernel)
#     if rot_kernel is not None:
#         masterkernel = convolve_1d_cy(masterkernel, rot_kernel)
#     if rtkernel is not None:
#         masterkernel = convolve_1d_cy(masterkernel, rtkernel)

#     # Convolve spectrum
#     cdef int len_master = masterkernel.shape[0]
#     if n < len_master:
#         diff = len_master - n
#         edge = 2 * diff
#         oflux_tmp = np.empty(n+2*edge)
#         pad_1d(oflux, oflux_tmp, edge, edge, 1.0)
#         oflux_tmp = convolve_1d_cy(oflux_tmp, masterkernel, mode='same')
#         cdef int start = (kernel.shape[0] - 1) // 2
#         out_same = out[start:start + signal.shape[0]]
#         output[:] = oflux_tmp[edge:-edge]
#     else:
#         output = np.convolve(oflux, masterkernel, mode='same')

#     return output