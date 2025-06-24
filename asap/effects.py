import numpy as np
from numba import jit
from scipy import constants as cst

@jit(nopython=True)
def gaussian(sigma):
    x = np.arange(16*sigma+1)-8*sigma
    x = np.arange(17*sigma)#-8*sigma
    if x.size%2==0:
        x = x[:-1]
    x = x - len(x)//2
    #     x = np.arange(20*sigma+sigma)-11*sigma
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x)**2/(2*sigma**2))

def __lsf_rotate(deltav,vsini,epsilon=0.6):
    # Based on lsf_rotate.pro:
    #  http://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
    #
    # Adapted from rotin3.f in the SYNSPEC software of Hubeny & Lanz
    # http://nova.astro.umd.edu/index.html    Also see Eq. 17.12 in
    # "The Observation and Analysis of Stellar Photospheres" by D. Gray (1992)
    ## The equation is rather 18.14
    ## Correct

    ## The number of points that we need
    npts = np.floor(2*vsini/deltav)
    if npts % 2 == 0:
        npts += 1
    nwid = np.floor(npts/2)
    # nwid = npts/2
    x = np.arange(npts) - nwid
    ## Transform this axis in units of deltav (here km/s)
    x = x*deltav
    ## The maximum velocity is vsini
    maxvel = vsini
    ## Compute the different part fo the equation
    e1 = 2.0*(1.0 - epsilon)
    ## Correct
    e2 = np.pi*epsilon/2.0
    ## Not correct? Must add maximum shift !
    e3 = np.pi*(1.0 - epsilon/3.0)*maxvel
    # ## Should be?
    # e3 = 1.0/np.pi/deltav/(1.0-epsilon)/3.
    ##
    x1 = np.abs(1.0 - (x/vsini)**2)

    velgrid = x#*maxvel
    ## wrong?
    ker = (e1*np.sqrt(x1) + e2*x1)/e3
    ## Should be?
    # ker = (e1*np.sqrt(x1) + e2*x1)*e3

    
    # # padding array using CONSTANT mode 
    # ker = np.pad(ker, (3, 3), 'constant',  
    #                 constant_values=(0, 0)) 

    return velgrid, ker

@jit(nopython=True, cache=True)
def _rt_macro(velocity_step, vmac, m):
    """
        velocity_step: fluxes should correspond to a spectrum homogeneously sampled in velocity space
                    with a fixed velocity step [km/s]
        vmac   : macroturbulence velocity [km/s]

        Based on SME's rtint
        It uses radial-tangential instead of isotropic Gaussian macroturbulence.
    """
    # if vmac is not None and vmac > 0:
    # mu represent angles that divide the star into equal area annuli,
    # ordered from disk center (mu=1) to the limb (mu=0).
    # But since we don't have intensity profiles at various viewing (mu) angles
    # at this point, we just take a middle point:
#         m = 0.5
    # Calc projected simga for radial and tangential velocity distributions.
    sigma = vmac/np.sqrt(2.0) / velocity_step
    sigr = sigma * m
    sigt = sigma * np.sqrt(1.0 - m**2.)
    # Figure out how many points to use in macroturbulence kernel
    # nmk = max(min(round(sigma*10), (len(flux)-3)/2), 3)
    nmk = max(round(sigma*10), 3)
    # nmk = round(sigma*10)
    # Construct radial macroturbulence kernel w/ sigma of mu*vmac/sqrt(2)
    if sigr > 0:
        xarg = (np.arange(2*nmk+1)-nmk) / sigr   # exponential arg
        #mrkern = np.exp(max((-0.5*(xarg**2)),-20.0))
        mrkern = np.exp(-0.5*(xarg**2))
        mrkern = mrkern/mrkern.sum()
    else:
        mrkern = np.zeros(2*nmk+1)
        mrkern[nmk] = 1.0    #delta function

    # Construct tangential kernel w/ sigma of sqrt(1-mu**2)*vmac/sqrt(2.)
    if sigt > 0:
        xarg = (np.arange(2*nmk+1)-nmk) /sigt
        mtkern = np.exp(-0.5*(xarg**2))
        mtkern = mtkern/mtkern.sum()
    else:
        mtkern = np.zeros(2*nmk+1)
        mtkern[nmk] = 1.0

    ## Sum the radial and tangential components, weighted by surface area
    area_r = 0.5
    area_t = 0.5
    mkern = area_r*mrkern + area_t*mtkern

    # Convolve the flux with the kernel
    # flux_conv = 1 - fftconvolve(1-flux, mkern, mode='same') # Fastest
    #import scipy
    #flux_conv = scipy.convolve(flux, mkern, mode='same') # Equivalent but slower

    return mkern
    # else:
    #     return None

@jit(nopython=True, cache=True)
def rt_macro(velocity_step, vmac):
    '''Function to compute a disk-integrated macroturbulence kernel.
    CAUTION: This one DOES not account for limb darkening.'''
    itot = 9
    dskdj = 0
    count = 0
    kernel = _rt_macro(velocity_step, vmac, 0)
    for dski in range(1,10):
        # print("Computation for anulus ", dski)
        ju = 6*dski
        dskri = float(dski)
        ritot = float(itot)
        dskr = (dskri - 0.5)/ritot
        area = np.pi*(2.0*dskri - 1.0)/(6.0*dskri*(ritot**2))

        ## We suppose a uniform disk, no need to loop
        # for dskj in range(0, ju):
            # dskaj = float(dskj)
            # dskth = np.pi*(dskaj - dskdj)/(3.0*dskri)
            # dskx = dskr*np.cos(dskth)
            # dsky = dskr*np.sin(dskth)
            # dskz = np.sqrt(1.0 - dskx**2 - dsky**2)
        subkern = _rt_macro(velocity_step, vmac, abs(dskr))
        # print(subkern)
        ## And here instead of just looping, we put a factor
        # subkern = subkern.copy()/area
        subkern = ju * subkern/area
        kernel = kernel + subkern
        count = count + ju
    kernel = kernel / count
    return kernel

def broaden_spectrum_2(wvl, flux, vinstru=0, vsini=0, epsilon=0.6, 
                     vmac=0, vmac_mode='g'):
    '''This is the clean function to use to broaden spectra.
    Last update Sep. 15, 2022'''

    oflux = flux.copy()

    vmacMode = vmac_mode.lower()
    if vmacMode == 'gaussian' or vmacMode == 'g':
        vmacRT = 0
        vmacGauss = vmac
    elif vmacMode=='radial-tangential' or vmacMode=='radial tangeantial' \
        or vmacMode=='rt':
        vmacRT = vmac
        vmacGauss = 0

    ## Define the wavelength (array assumed constant in velocity)
    vel_step = (wvl[len(wvl)//2]-wvl[len(wvl)//2-1]) / \
        ((wvl[len(wvl)//2]+wvl[len(wvl)//2-1])/2) * cst.c*1e-3 ## In Km/s

    ## Convert vmac and vsini to standard deviation
    vmacGauss = vmacGauss/(2*np.sqrt(2*np.log(2))) # FWHM to standard deviation
    vinstru = vinstru/(2*np.sqrt(2*np.log(2))) # FWHM to standard deviation
    
    ## Quadratic sum of the gaussian values
    vgauss = np.sqrt(vmacGauss**2 + vinstru**2)
    sigma_gauss = vgauss/vel_step

    ## Compute the gaussian kernel
    if sigma_gauss <= 0:
        gauss_kernel = None
        len_gauss_kernel = 0
    elif sigma_gauss > 0:
        gauss_kernel = gaussian(sigma_gauss)
        len_gauss_kernel = len(gauss_kernel)
        gauss_kernel /= gauss_kernel.sum()

    ## Compute the roation kernel
    # if vsini>0:
    #     rot_kernel = None
    if vsini==0:
        rot_kernel = None
        len_rot_kernel = 0
    else:
        if vsini<0: vsini = -vsini
        _, rot_kernel = __lsf_rotate(vel_step, vsini, epsilon=epsilon)
        len_rot_kernel = len(rot_kernel)

        # from scipy.integrate import simpson
        # from numpy import trapz
        # print(vel_step, vsini, len(rot_kernel))
        # area = simpson(rot_kernel, dx=vel_step)
        # print("area =", area)

        # from IPython import embed
        # embed()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # # for vsini in np.arange(0, 2, 0.2):
        # #     _, rot_kernel = __lsf_rotate(vel_step, vsini, epsilon=epsilon)
        # #     plt.plot(_, rot_kernel, label=str(vsini))
        # plt.plot(_, rot_kernel/rot_kernel.sum(), label=str(vsini))
        # plt.legend()
        # plt.show()
        # exit()
        # from IPython import embed
        # embed()
        # import os
        # os._exit(1)
        area = rot_kernel.sum()
        rot_kernel /= area

    ## Compute the RT kernel
    if vmacRT>0:
        rtkernel = rt_macro(vel_step, vmacRT)
        len_rtkernel = len(rtkernel)
        rtkernel /= rtkernel.sum()
    else:
        rtkernel = None
        len_rtkernel = 0
    ## Perform convolutions
    #
    # if gauss_kernel is not None:
    #     oflux = np.convolve(oflux, gauss_kernel, mode='same')
    # if rot_kernel is not None:
    #     oflux = np.convolve(oflux, rot_kernel, mode='same')
    # if rtkernel is not None:
    #     oflux = np.convolve(oflux, rtkernel, mode='same')

    masterkernel = np.array([1]) ## We start with a DIRAC
    if gauss_kernel is not None:
        masterkernel = np.convolve(masterkernel, gauss_kernel, mode='full')
    if rot_kernel is not None:
        masterkernel = np.convolve(masterkernel, rot_kernel, mode='full')
    if rtkernel is not None:
        masterkernel = np.convolve(masterkernel, rtkernel, mode='full')
    # masterkernel = masterkernel/np.sum(masterkernel)
    oflux = np.convolve(oflux, masterkernel, mode='same')

    # if gauss_kernel is not None:
    #     oflux = scipy.ndimage.convolve1d(oflux, gauss_kernel, mode='constant')
    # if rot_kernel is not None:
    #     oflux = scipy.ndimage.convolve1d(oflux, rot_kernel, mode='constant')
    # if rtkernel is not None:
    #     oflux = scipy.ndimage.convolve1d(oflux, rtkernel, mode='constant')
    
    #
    # if gauss_kernel is not None:
    #     oflux = scipy.signal.fftconvolve(oflux, gauss_kernel, 
    #                                         mode='same', axes=-1)
    # if rot_kernel is not None:
    #     oflux = scipy.signal.fftconvolve(oflux, rot_kernel, 
    #                                         mode='same', axes=-1)
    # if rtkernel is not None:
    #     oflux = scipy.signal.fftconvolve(oflux, rtkernel, 
    #                                         mode='same', axes=-1)


    return oflux

def _sampling_uniform_in_velocity(wave_base, wave_top, velocity_step):
    """
    -- Based on iSpec source code --
    Create a uniformly spaced grid in terms of velocity:

    Input parameters:
        - wave_base     :   [float] Lowest value of the array
        - wave_top      :   [float] Highest limit of the array
        - velocity_step :   [float] Wavelength step
    Output parameters:
        - _             :   [ndarray] Wavelength array evenly spaced in velocity

    iSpec docstring:
    - An increment in position (i => i+1) supposes a constant velocity increment
      (velocity_step).
    - An increment in position (i => i+1) does not implies a constant wavelength
      increment.
    - It is uniform in log(wave) since:
          Wobs = Wrest * (1 + Vr/c)^[1,2,3..]
          log10(Wobs) = log10(Wrest) + [1,2,3..] * log10(1 + Vr/c)
      The last term is constant when dealing with wavelenght in log10.
    - Useful for building the cross correlate function used for determining the 
      radial velocity of a star.
    """
    # Speed of light
    c = 299792.4580 # km/s
    #c = 299792458.0 # m/s

    ### Numpy optimized:
    # number of elements to go from wave_base to wave_top in increments of 
    # velocity_step
    i = int(np.ceil( (c * (wave_top - wave_base)) / (wave_base*velocity_step)))
    grid = wave_base * np.power((1 + (velocity_step / c)), np.arange(i)+1)

    # Ensure wavelength limits since the "number of elements i" tends to be 
    # overestimated
    wfilter = grid <= wave_top
    grid = grid[wfilter]

    return np.asarray(grid)