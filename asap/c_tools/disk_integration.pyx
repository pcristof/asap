# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, sqrt, pi, asin

# Replace this with your real doppler function
cdef double doppler(double vrad) nogil:
    return 1.0 + vrad / 3e5

cdef void precompute_interp(
    double[:] x,
    double dop,
    Py_ssize_t[:] idx,
    double[:] w
) noexcept nogil:
    cdef Py_ssize_t i, j = 0, n = x.shape[0]
    cdef double xp_j, xp_j1

    for i in range(n):
        while j < n-2 and x[j+1] * dop < x[i]:
            j += 1
        xp_j  = x[j]   * dop
        xp_j1 = x[j+1] * dop
        idx[i] = j
        w[i] = (x[i] - xp_j) / (xp_j1 - xp_j)

cdef inline void interp_apply(
    Py_ssize_t[:] idx,
    double[:] w,
    double[:] fp,
    double[:] out
) noexcept nogil:
    cdef Py_ssize_t i, j
    for i in range(out.shape[0]):
        j = idx[i]
        out[i] = fp[j] * (1.0 - w[i]) + fp[j+1] * w[i]

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
            double[:, :, :] stokesi,
            double[:, :, :] cont,
        #    Cell[:] cells,
           np.ndarray cells_np,
           double veq=10.0,
           double rotAngle=90.0,
           double vsini=0.,
           double vmac=0.,
           vmac_mode='g'
           ):

    cdef Cell[:] cells = cells_np
    
    cdef int k, nwl
    cdef double area
    cdef double sai, vrad, dop
    cdef int r
    
    cdef Py_ssize_t ncells = cells.shape[0]
    cdef Py_ssize_t nreg = wvl.shape[0]

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
    cdef double[:] weights = np.empty(nwl, dtype=np.float64)
    

    cdef double[:,:] num_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef double[:,:] den_2d = np.empty((nreg,nwl), dtype=np.float64)
    cdef Py_ssize_t[:] idx = np.empty(nwl, dtype=np.intp)

    # s_int = np.empty(nwl)
    # c_int = np.empty(nwl)
    # wvl_dop = np.empty(nwl, dtype=np.float64)

    sai = sin(2.0 * pi * rotAngle / 360.0)

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
                    precompute_interp(wvl_r, dop, idx, weights)
                    # interp_linear(wvl_r, wvl_dop, stokesi[r, mu_idx, :], s_int)
                    # interp_linear(wvl_r, wvl_dop, cont[r, mu_idx, :],    c_int)
                    interp_apply(idx, weights, stokesi[r, mu_idx, :], s_int)
                    interp_apply(idx, weights, cont[r, mu_idx, :], s_int)
                else:
                    for k in range(nwl):
                        s_int[k] = stokesi[r, mu_idx, k]
                        c_int[k] = cont[r, mu_idx, k]

                for k in range(nwl):
                    num[k] += s_int[k] * area
                    den[k] += c_int[k] * area
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