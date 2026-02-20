# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, pi, asin
import cython 
from libc.math cimport sqrt, exp, log
from libc.math cimport ceil

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


# Convert FWHM to sigma
cdef double FWHM_to_sigma(double fwhm):
    return fwhm / (2.0 * sqrt(2.0 * log(2.0)))

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

# Replace this with your real doppler function
cdef double doppler(double vrad) nogil:
    return 1.0 + vrad / 3e5

# cdef void precompute_interp(
#     double[:] x,
#     double dop,
#     Py_ssize_t[:] idx,
#     double[:] w
# ) noexcept nogil:
#     cdef Py_ssize_t i, j = 0, n = x.shape[0]
#     cdef double xp_j, xp_j1

#     for i in range(n):
#         while j < n-2 and x[j+1] * dop < x[i]:
#             j += 1
#         xp_j  = x[j]   * dop
#         xp_j1 = x[j+1] * dop
#         idx[i] = j
#         w[i] = (x[i] - xp_j) / (xp_j1 - xp_j)

# cdef inline void interp_apply(
#     Py_ssize_t[:] idx,
#     double[:] w,
#     double[:] fp,
#     double[:] out
# ) noexcept nogil:
#     cdef Py_ssize_t i, j
#     for i in range(out.shape[0]):
#         j = idx[i]
#         out[i] = fp[j] * (1.0 - w[i]) + fp[j+1] * w[i]

cdef inline void interp_linear(
    double[:] x,
    double[:] xp,
    double[:] fp,
    double[:] out
) noexcept nogil:
    cdef Py_ssize_t i, j = 0
    cdef double t
    for i in range(x.shape[0]):
        while j < xp.shape[0] - 2 and xp[j+1] < x[i]:
            j += 1
        t = (x[i] - xp[j]) / (xp[j+1] - xp[j])
        out[i] = fp[j] * (1.0 - t) + fp[j+1] * t

cdef inline void interp_linear_shift(
    double[:] x,
    double dop,
    double[:] fp,
    double[:] out
) noexcept nogil:
    cdef Py_ssize_t i, j = 0
    cdef double t
    for i in range(x.shape[0]):
        while j < x.shape[0] - 2 and x[j+1]*dop < x[i]:
            j += 1
        t = (x[i] - x[j]*dop) / (x[j+1]*dop - x[j]*dop)
        out[i] = fp[j] * (1.0 - t) + fp[j+1] * t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void interp_linear_precompute(
    double[:] xp,
    double[:] fp,
    double[:] slopes,
    double[:] offsets
) noexcept nogil:
    cdef Py_ssize_t i, n = xp.shape[0] - 1
    cdef double dx
    for i in range(n):
        dx = xp[i+1] - xp[i]
        slopes[i]  = (fp[i+1] - fp[i]) / dx
        offsets[i] = fp[i] - slopes[i] * xp[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void interp_linear_apply(
    double[:] x,
    double[:] xp,
    double[:] slopes,
    double[:] offsets,
    double[:] out
) noexcept nogil:
    cdef Py_ssize_t i, j = 0
    cdef Py_ssize_t nx = x.shape[0]
    cdef Py_ssize_t np = xp.shape[0] - 1
    cdef double xi

    for i in range(nx):
        xi = x[i]
        if xi <= xp[0] or xi >= xp[np]:
            out[i] = 0.0
            continue

        while xi > xp[j+1]:
            j += 1

        out[i] = slopes[j] * xi + offsets[j]

def integrate_sphere(
           np.ndarray[double, ndim=1] wvl,
           np.ndarray[double, ndim=2] stokesi,
           np.ndarray[double, ndim=2] cont,
           np.ndarray[double, ndim=1] mus,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef int itot = stokesi.shape[1]
    cdef int i, j, k, ncells, idx, idx_best, nwl
    cdef double ri, r, area, rj, th, x, y, z
    cdef double sai, mu_angle, vrad, dop
    cdef double best, diff

    nwl = wvl.shape[0]

    # --- Use numpy arrays here, not memoryviews ---
    cdef np.ndarray[np.float64_t, ndim=1] num = np.zeros(nwl, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] den = np.zeros(nwl, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] s, c, wvl_dop #, s_int, c_int
    cdef double[:] s_int, c_int

    cdef int ncellsFac = 6

    s_int = np.empty(nwl)
    c_int = np.empty(nwl)

    sai = sin(2.0 * pi * rotAngle / 360.0)

    for i in range(1, itot + 1):
        ri = float(i)
        ncells = ncellsFac * i
        r = (ri - 0.5) / itot
        area = pi * (2.0 * i - 1.0) / (ncellsFac * i * (itot**2))

        for j in range(ncells):
            rj = float(j)
            th = pi * (rj) / (3.0 * ri)
            x = r * cos(th)
            y = r * sin(th)
            z = sqrt(1.0 - x*x - y*y)

            mu_angle = asin(z)

            # Find nearest mu
            idx_best = 0
            best = 1e9

            for idx in range(mus.shape[0]):
                diff = (mu_angle - mus[idx])**2
                if diff < best:
                    best = diff
                    idx_best = idx

            # get spectra
            s = stokesi[:, idx_best]
            c = cont[:, idx_best]

            vrad = r * veq * cos(th)
            vrad *= sai
            dop = doppler(vrad)

            # --- create doppler-shifted wavelength array ---
            wvl_dop = np.empty(nwl, dtype=np.float64)
            for k in range(nwl):
                wvl_dop[k] = wvl[k] * dop

            # --- interpolation using numpy arrays ---
            interp_linear(wvl, wvl_dop, s, s_int)
            interp_linear(wvl, wvl_dop, c, c_int)

            for k in range(nwl):
                num[k] += s_int[k] * area
                den[k] += c_int[k] * area

    return num / den


def integrate_sphere_fast(
        #    np.ndarray[double, ndim=1] wvl,
        #    np.ndarray[double, ndim=2] stokesi,
        #    np.ndarray[double, ndim=2] cont,
            double[:] wvl,
            double[:, :] stokesi,
            double[:, :] cont,
        #    Cell[:] cells,
           np.ndarray cells_np,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef Cell[:] cells = cells_np
    
    cdef int itot = stokesi.shape[1]
    cdef int i, j, k, idx, idx_best, nwl
    cdef double ri, r, area, rj, th, x, y, z
    cdef double sai, mu_angle, vrad, dop
    cdef double best, diff
    
    cdef Py_ssize_t ncells = cells.shape[0]

    nwl = wvl.shape[0]

    # --- Use numpy arrays here, not memoryviews ---
    # cdef np.ndarray[np.float64_t, ndim=1] num = np.zeros(nwl, dtype=np.float64)
    # cdef np.ndarray[np.float64_t, ndim=1] den = np.zeros(nwl, dtype=np.float64)
    # cdef np.ndarray[np.float64_t, ndim=1] s, c, wvl_dop #, s_int, c_int
    # cdef double[:] s_int, c_int
    cdef double[:] num = np.zeros(nwl, dtype=np.float64)
    cdef double[:] den = np.zeros(nwl, dtype=np.float64)

    cdef double[:] s_int = np.empty(nwl, dtype=np.float64)
    cdef double[:] c_int = np.empty(nwl, dtype=np.float64)
    cdef double[:] wvl_dop = np.empty(nwl, dtype=np.float64)


    # s_int = np.empty(nwl)
    # c_int = np.empty(nwl)
    # wvl_dop = np.empty(nwl, dtype=np.float64)

    sai = sin(2.0 * pi * rotAngle / 360.0)

    with nogil:
        for icell in range(ncells):
            mu_idx = cells[icell].mu_idx
            area   = cells[icell].area
            vrad   = cells[icell].vrad_norm * veq * sai
            dop    = doppler(vrad)

            for k in range(nwl):
                wvl_dop[k] = wvl[k] * dop

            interp_linear(wvl, wvl_dop, stokesi[:, mu_idx], s_int)
            interp_linear(wvl, wvl_dop, cont[:, mu_idx],    c_int)

            for k in range(nwl):
                num[k] += s_int[k] * area
                den[k] += c_int[k] * area

    return np.asarray(num) / np.asarray(den)

def integrate_sphere_fast_regions(
        #    np.ndarray[double, ndim=1] wvl,
        #    np.ndarray[double, ndim=2] stokesi,
        #    np.ndarray[double, ndim=2] cont,
            double[:, :] wvl,
            np.ndarray[double, ndim=3] stokesi,
            double[:, :, :] cont,
        #    Cell[:] cells,
           np.ndarray cells_np,
           np.ndarray mus,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef Cell[:] cells = cells_np
    
    cdef int k, nwl, i, edge
    cdef double area
    cdef double sai, vrad, dop, vel_step
    cdef int r
    
    cdef Py_ssize_t ncells = cells.shape[0]
    cdef Py_ssize_t nreg = wvl.shape[0]
    cdef Py_ssize_t nmus = stokesi.shape[1]

    nwl = wvl.shape[1]

    # --- Use numpy arrays here, not memoryviews ---
    # cdef np.ndarray[np.float64_t, ndim=1] num = np.zeros(nwl, dtype=np.float64)
    # cdef np.ndarray[np.float64_t, ndim=1] den = np.zeros(nwl, dtype=np.float64)
    # cdef np.ndarray[np.float64_t, ndim=1] s, c, wvl_dop #, s_int, c_int
    # cdef double[:] s_int, c_int
    cdef double[:] num = np.empty(nwl, dtype=np.float64)
    cdef double[:] den = np.empty(nwl, dtype=np.float64)

    cdef double[:] s_int = np.empty(nwl, dtype=np.float64)
    cdef double[:] c_int = np.empty(nwl, dtype=np.float64)
    cdef double[:] wvl_dop = np.empty(nwl, dtype=np.float64)
    cdef double[:] wvl_r = np.empty(nwl, dtype=np.float64)
    # cdef double[:] weights = np.empty(nwl, dtype=np.float64)
    cdef Py_ssize_t[:] lastbins = np.empty(nreg, dtype=np.intp)
    # cdef double[:, :, :] 
    cdef np.ndarray[np.float64_t, ndim=3] stokesi_b 
    stokesi_b = np.empty((nreg, nmus, nwl), dtype=np.float64)
    cdef double[:,:,:] stokesi_b_mem = stokesi_b
    cdef np.ndarray[np.float64_t, ndim=1] rtkernel

    cdef double[:,:] num_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef double[:,:] den_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef Py_ssize_t[:] idx = np.empty(nwl, dtype=np.intp)

    # s_int = np.empty(nwl)
    # c_int = np.empty(nwl)
    # wvl_dop = np.empty(nwl, dtype=np.float64)

    ## Broaden spectra to account for macroturbulence
    for r in range(nreg):
        lastbins[r] = nwl
        mid = nwl // 2
        vel_step = (wvl[r, mid] - wvl[r, mid-1]) / ((wvl[r, mid] + wvl[r, mid-1])/2.0) * 2.99792458e5
        for k in range(nmus):
            if vmac>0.:
                if vmac_mode=='rt':
                    rtkernel = _rt_macro_cy(vel_step, vmac, mus[k])
                elif vmac_mode=='g':
                    sigma_gauss = FWHM_to_sigma(vmac) / vel_step
                    rtkernel = gaussian_cy(sigma_gauss)
                rtkernel /= rtkernel.sum()
                for i in range(nwl-1, 0, -1):
                    if stokesi[r,k,i]!=0.: 
                        lastbins[r]=i
                        break
                stokesi_b[r, k, :] = np.convolve(stokesi[r,k,:], rtkernel, mode='same')#[(len_master-1)//2:(len_master-1)//2+n]
                # np.convolve(stokesi[r,k], rtkernel, mode='same')#[(len_master-1)//2:(len_master-1)//2+n]
            else:
                stokesi_b[r, k] = stokesi[r,k]
        
    sai = sin(2.0 * pi * rotAngle / 360.0)

    edge = <int>ceil(vmac/vel_step)

    for icell in range(ncells):
        vrad   = cells[icell].vrad_norm * veq * sai
        dop    = doppler(vrad)
        cells[icell].dop = dop

    with nogil:
        for r in range(nreg):
            wvl_r = wvl[r]
            # num = np.zeros(nwl, dtype=np.float64)
            # den = np.zeros(nwl, dtype=np.float64)
            for k in range(nwl):
                num[k] = 0.
                den[k] = 0.
            ## Precompute interpolation weights for this region:
            for icell in range(ncells):
                mu_idx = cells[icell].mu_idx
                area   = cells[icell].area
                # vrad   = cells[icell].vrad_norm * veq * sai
                dop   = cells[icell].dop

                for k in range(nwl):
                    wvl_dop[k] = wvl_r[k] * dop

                if dop!=1.0:
                    # precompute_interp(wvl_r, dop, idx, weights)
                    interp_linear(wvl_r, wvl_dop, stokesi_b_mem[r, mu_idx, :], s_int)
                    interp_linear(wvl_r, wvl_dop, cont[r, mu_idx, :],    c_int)
                    # interp_apply(idx, weights, stokesi_b[r, mu_idx, :], s_int)
                    # interp_apply(idx, weights, cont[r, mu_idx, :], s_int)
                else:
                    for k in range(nwl):
                        s_int[k] = stokesi_b[r, mu_idx, k]
                        c_int[k] = cont[r, mu_idx, k]

                for k in range(nwl):
                    num[k] += s_int[k] * area
                    den[k] += c_int[k] * area
            for i in range(nwl, lastbins[r]-edge, -1):
                num[i]=0.
            for i in range(0, edge,1):
                num[i]=0.
            num_2d[r] = num
            den_2d[r] = den
    return np.asarray(num_2d) / np.asarray(den_2d)

def integrate_sphere_fast_regions_2(
    ## CLEVER TRICKS; 40% faster than version 1 !
    ## Linear interpoaltion is embeded in the code
    ## -> allows faster computation by avoiding memory asignments. 
    ## 
        #    np.ndarray[double, ndim=1] wvl,
        #    np.ndarray[double, ndim=2] stokesi,
        #    np.ndarray[double, ndim=2] cont,
            double[:, :] wvl,
            np.ndarray[double, ndim=3] stokesi,
            double[:, :, :] cont,
        #    Cell[:] cells,
           np.ndarray cells_np,
           np.ndarray mus,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef Cell[:] cells = cells_np
    
    cdef int k, nwl, i, edge, j
    cdef double area
    cdef double sai, vrad, dop, vel_step
    cdef int r
    
    cdef Py_ssize_t ncells = cells.shape[0]
    cdef Py_ssize_t nreg = wvl.shape[0]
    cdef Py_ssize_t nmus = stokesi.shape[1]

    nwl = wvl.shape[1]

    # --- Use numpy arrays here, not memoryviews ---
    cdef double[:] num = np.empty(nwl, dtype=np.float64)
    cdef double[:] den = np.empty(nwl, dtype=np.float64)

    cdef double[:] wvl_r = np.empty(nwl, dtype=np.float64)
    cdef Py_ssize_t[:] lastbins = np.empty(nreg, dtype=np.intp)
    cdef np.ndarray[np.float64_t, ndim=3] stokesi_b 
    stokesi_b = np.empty((nreg, nmus, nwl), dtype=np.float64)
    cdef double[:,:,:] stokesi_b_mem = stokesi_b
    cdef np.ndarray[np.float64_t, ndim=1] rtkernel

    cdef double[:,:] num_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef double[:,:] den_2d = np.empty((nreg,nwl), dtype=np.float64)

    ## Broaden spectra to account for macroturbulence
    for r in range(nreg):
        lastbins[r] = nwl
        mid = nwl // 2
        vel_step = (wvl[r, mid] - wvl[r, mid-1]) / ((wvl[r, mid] + wvl[r, mid-1])/2.0) * 2.99792458e5
        for k in range(nmus):
            if vmac>0.:
                if vmac_mode=='rt':
                    rtkernel = _rt_macro_cy(vel_step, vmac, mus[k])
                elif vmac_mode=='g':
                    sigma_gauss = FWHM_to_sigma(vmac) / vel_step
                    rtkernel = gaussian_cy(sigma_gauss)
                rtkernel /= rtkernel.sum()
                for i in range(nwl-1, 0, -1):
                    if stokesi[r,k,i]!=0.: 
                        lastbins[r]=i
                        break
                stokesi_b[r, k, :] = np.convolve(stokesi[r,k,:], rtkernel, mode='same')#[(len_master-1)//2:(len_master-1)//2+n]
            else:
                stokesi_b[r, k] = stokesi[r,k]
        
    sai = sin(2.0 * pi * rotAngle / 360.0)

    edge = <int>ceil(vmac/vel_step)

    for icell in range(ncells):
        vrad   = cells[icell].vrad_norm * veq * sai
        dop    = doppler(vrad)
        cells[icell].dop = dop

    with nogil:
        for r in range(nreg):
            wvl_r = wvl[r]
            for k in range(nwl):
                num[k] = 0.
                den[k] = 0.
            ## Precompute interpolation weights for this region:
            for icell in range(ncells):
                mu_idx = cells[icell].mu_idx
                area   = cells[icell].area
                # vrad   = cells[icell].vrad_norm * veq * sai
                dop   = cells[icell].dop

                j=0
                for k in range(nwl):
                    if dop!=1.0:
                        while j < wvl_r.shape[0] - 2 and wvl_r[j+1]*dop < wvl_r[k]:
                            j += 1
                        t = (wvl_r[k] - wvl_r[j]*dop) / (wvl_r[j+1]*dop - wvl_r[j]*dop)
                        num[k] += stokesi_b_mem[r, mu_idx, j] * (1.0 - t) + stokesi_b_mem[r, mu_idx, j+1] * t * area
                        den[k] += cont[r, mu_idx, j] * (1.0 - t) + stokesi_b_mem[r, mu_idx, j+1] * t * area
                    else:
                        num[k] += stokesi_b_mem[r, mu_idx, k]*area
                        den[k] += cont[r, mu_idx, k]*area
            for i in range(nwl, lastbins[r]-edge, -1):
                num[i]=0.
            for i in range(0, edge,1):
                num[i]=0.
            num_2d[r] = num
            den_2d[r] = den
    return np.asarray(num_2d) / np.asarray(den_2d)


def integrate_sphere_fast_regions_2_opt(
    ## CLEVER TRICKS; 40% faster than version 1 !
    ## Linear interpoaltion is embeded in the code
    ## -> allows faster computation by avoiding memory asignments. 
    ## We loop through the mu angles to create an interpolator.
    ## 
            double[:, :] wvl,
            np.ndarray[double, ndim=3] stokesi,
            double[:, :, :] cont,
        #    Cell[:] cells,
           np.ndarray cells_np,
           np.ndarray mus,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef Cell[:] cells = cells_np
    
    cdef int k, nwl, i, edge, j, var
    cdef double area
    cdef double sai, vrad, dop, vel_step, dx
    cdef int r
    
    cdef Py_ssize_t ncells = cells.shape[0]
    cdef Py_ssize_t nreg = wvl.shape[0]
    cdef Py_ssize_t nmus = stokesi.shape[1]

    nwl = wvl.shape[1]

    # --- Use numpy arrays here, not memoryviews ---
    cdef double[:] stokesi_b_mem_1d = np.empty(nwl, dtype=np.float64)
    cdef double[:] cont_b_mem_1d = np.empty(nwl, dtype=np.float64)
    cdef double[:] num = np.empty(nwl, dtype=np.float64)
    cdef double[:] den = np.empty(nwl, dtype=np.float64)
    cdef double[:] slopes = np.empty(nwl-1, dtype=np.float64)
    cdef double[:] offsets = np.empty(nwl-1, dtype=np.float64)
    cdef double[:] slopes_cont = np.empty(nwl-1, dtype=np.float64)
    cdef double[:] offsets_cont = np.empty(nwl-1, dtype=np.float64)

    cdef double[:] wvl_r = np.empty(nwl, dtype=np.float64)
    cdef Py_ssize_t[:] lastbins = np.empty(nreg, dtype=np.intp)
    cdef np.ndarray[np.float64_t, ndim=3] stokesi_b 
    stokesi_b = np.empty((nreg, nmus, nwl), dtype=np.float64)
    cdef double[:,:,:] stokesi_b_mem = stokesi_b
    cdef np.ndarray[np.float64_t, ndim=1] rtkernel

    cdef double[:,:] num_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef double[:,:] den_2d = np.empty((nreg,nwl), dtype=np.float64)

    ## Broaden spectra to account for macroturbulence
    for r in range(nreg):
        lastbins[r] = nwl
        mid = nwl // 2
        vel_step = (wvl[r, mid] - wvl[r, mid-1]) / ((wvl[r, mid] + wvl[r, mid-1])/2.0) * 2.99792458e5
        for k in range(nmus):
            if vmac>0.:
                if vmac_mode=='rt':
                    rtkernel = _rt_macro_cy(vel_step, vmac, mus[k])
                elif vmac_mode=='g':
                    sigma_gauss = FWHM_to_sigma(vmac) / vel_step
                    rtkernel = gaussian_cy(sigma_gauss)
                rtkernel /= rtkernel.sum()
                for i in range(nwl-1, 0, -1):
                    if stokesi[r,k,i]!=0.: 
                        lastbins[r]=i
                        break
                stokesi_b[r, k, :] = np.convolve(stokesi[r,k,:], rtkernel, mode='same')#[(len_master-1)//2:(len_master-1)//2+n]
            else:
                stokesi_b[r, k] = stokesi[r,k]
        
    sai = sin(2.0 * pi * rotAngle / 360.0)

    edge = <int>ceil(vmac/vel_step)

    for icell in range(ncells):
        vrad   = cells[icell].vrad_norm * veq * sai
        dop    = doppler(vrad)
        cells[icell].dop = dop

    with nogil:
        for r in range(nreg):
            wvl_r = wvl[r]
            ## Initialize the numerator and denominator
            for k in range(nwl):
                num[k] = 0.
                den[k] = 0.
            var = 0
            for icell in range(ncells):
                ## For this cell:
                mu_idx = cells[icell].mu_idx
                area   = cells[icell].area
                # vrad   = cells[icell].vrad_norm * veq * sai
                dop   = cells[icell].dop
                ## Only if we changed mu angle do we recompute the interp
                ## This is the spectrum we keep
                if mu_idx!=var:
                #     print(mu_idx)
                    stokesi_b_mem_1d = stokesi_b_mem[r, mu_idx]
                    cont_b_mem_1d = cont[r, mu_idx]
                    var = mu_idx
                    ## Compute the coefficients
                    for k in range(nwl-1):
                        slopes[k] = (stokesi_b_mem_1d[k+1] - stokesi_b_mem_1d[k]) / (wvl_r[k+1] - wvl_r[k])
                        offsets[k] = stokesi_b_mem_1d[k] - slopes[k] * wvl_r[k]
                        slopes_cont[k] = ((cont_b_mem_1d[k+1] - cont_b_mem_1d[k]) / (wvl_r[k+1] - wvl_r[k]))
                        offsets_cont[k] = cont_b_mem_1d[k] - slopes_cont[k] * wvl_r[k]
                j = 0
                k = 0
                while True:
                    if wvl_r[j]*dop>wvl_r[nwl-1]:
                        break ## Reached the end
                    elif wvl_r[j]*dop<wvl_r[0]:
                        ## While we are out of bounds on the left,
                        ## We skip the bin and continue
                        j+=1
                    elif wvl_r[j]*dop<wvl_r[k]:
                        ## We are not out of bounds on the left, but we are
                        ## misplaced with respect to k.
                        k-=1
                    elif wvl_r[j]*dop>wvl_r[k+1]:
                        ## We are not out or bounds on the left, but we are not
                        ## correctly placed
                        k+=1
                    ## Now the arrays are aligned for that j,
                    ## So we register and move to the next j and k
                    else:
                        if wvl_r[j]==wvl_r[k]:
                            num[j]+=stokesi_b_mem_1d[k]*area
                            den[j]+=cont_b_mem_1d[k]*area
                        elif (wvl_r[j]*dop>wvl_r[k]) & (wvl_r[j]*dop<wvl_r[k+1]):
                            num[j]+=(slopes[k]*wvl_r[j]*dop+offsets[k])*area
                            den[j]+=(slopes_cont[k]*wvl_r[j]*dop+offsets_cont[k])*area
                        k+=1
                        j+=1
                        ## Break the loop if eithe j or k reached the en of the array !
                        if k+1>nwl-1: 
                            break
                        if j+1>nwl-1: 
                            break
            num_2d[r] = num
            den_2d[r] = den
    return np.asarray(num_2d) / np.asarray(den_2d)

import cython

cdef struct Cell:
    int    mu_idx       # nearest mu index
    double area         # projected area weight
    double vrad_norm    # r * cos(theta), NO veq or sin(i)
    double dop    # r * cos(theta), NO veq or sin(i)

@cython.boundscheck(False)
@cython.wraparound(False)
def build_sphere_grid(
    double[:] mus,
    int itot,
    int ncellsFac=6
):
    """
    Precompute surface integration cells for a rotating sphere.

    Parameters
    ----------
    mus : 1D array
        Mu angles used by the spectra (asin(mu))
    itot : int
        Number of radial rings
    ncellsFac : int
        Controls azimuthal sampling density

    Returns
    -------
    cells : numpy structured array viewable as Cell[:]
    """

    cdef int i, j, idx, idx_best, ncells
    cdef double ri, r, area
    cdef double rj, th, x, y, z, mu_angle
    cdef double diff, best

    # --- Count total cells ---
    cdef int ncell_tot = 0
    for i in range(1, itot + 1):
        ncell_tot += ncellsFac * i

    # --- Allocate cell array ---
    cdef np.ndarray cells_np = np.empty(
        ncell_tot,
        dtype=np.dtype([
            ("mu_idx",    np.int32),
            ("area",      np.float64),
            ("vrad_norm", np.float64),
            ("dop", np.float64),
        ], align=True)
    )

    cdef Cell[:] cells = cells_np
    cdef int icell = 0

    # --- Build grid ---
    for i in range(1, itot + 1):
        ri = <double>i
        ncells = ncellsFac * i

        r = (ri - 0.5) / itot
        area = pi * (2.0 * ri - 1.0) / (ncellsFac * ri * itot * itot)

        for j in range(ncells):
            rj = <double>j
            th = pi * rj / (3.0 * ri)

            x = r * cos(th)
            y = r * sin(th)
            z = sqrt(1.0 - x*x - y*y)

            mu_angle = asin(z)

            # --- nearest mu ---
            idx_best = 0
            best = 1e300
            for idx in range(mus.shape[0]):
                diff = (mu_angle - mus[idx])
                diff = diff * diff
                if diff < best:
                    best = diff
                    idx_best = idx

            cells[icell].mu_idx    = idx_best
            cells[icell].area      = area
            cells[icell].vrad_norm = r * cos(th)
            cells[icell].dop = 0.

            icell += 1

    return cells_np