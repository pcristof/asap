from numba import jit
from asap import params
from scipy import constants as cst
import numpy as np
import warnings
from scipy.interpolate import interp1d 
from asap import effects as effects

@jit(nopython=params.JIT)
def doppler(v):
    '''Returns relativistic doppler factor in km/s.
    Caution: conventions assume that the v is negative if the source and the 
    object are moving towards each other, positive if they are moving away 
    from each other.'''
    _v = v#*1e3
    if _v >= (cst.c*1e-3):
        print('Caution, star going faster than light !')
        return np.inf
    if _v < -(cst.c*1e-3):
        print('Caution, star going faster than light !')
        return 0
    else:
        factor = np.sqrt( (1 + _v/(cst.c*1e-3)) / (1 - _v/(cst.c*1e-3)))
    return factor

@jit(nopython=params.JIT, cache=True)
def jitlinsolve_nocov(mat, Y, Yerr):
    '''Same exact thing as jitlinsolve, but does not return the covariance
    matrix. It turns out that computing hte covariance matrix is FREAKISHLY
    slow.'''
    subcov = np.linalg.inv(np.dot(mat.T, mat))
    coeffs = np.dot(np.dot(subcov, mat.T), Y) ## these are the coeffs
    return coeffs

def convert_lambda_in_vacuum(lambdas, method=1):
    '''Method to convert the air wavelengths in vacuum. 
    Inpired from equation on VALD website: 
    https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    CAUTION: input wavelength must be in ANGSTROMS
    Input parameters:
    - vacuum_lambdas        :   Wavelength solution to convert. Must be 
                                expressed in Angstroms.
    - method                :   [int] 1 or 2.
                                if 1 : equation updated on 09/25/2021
                                -- extracted from the VALD webpage.
                                if 2 : equation used before 09/25/2021 
                                -- inverted equation for vacuum to air 
                                conversion     '''

    ## For SPIRou we can check that the wavelength make sense:
    if np.any(lambdas<8000) or np.any(lambdas>29000) :
        warnings.warn("WARNING: Are you sure you inputted ANGSTROM values for the wavelengths?")
    if method==1:
        s = 1e4/lambdas
        n = 1 + 0.00008336624212083 \
            + 0.02408926869968 / (130.1065924522 - s**2) \
            + 0.0001599740894897 / (38.92568793293 - s**2)
    elif method==2:
        s = 1e4/lambdas
        n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    else:
        raise Exception('Method unknown')
    vacuum_lambdas = n * lambdas
    return vacuum_lambdas

def convert_lambda_in_air(vacuum_lambdas):
    '''Method to convert the air wavelengths in vacuum. Inpired from equation on VALD website: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    CAUTION: input wavelength must be in ANGSTROMS
    Input parameters:
    - vacuum_lambdas        :   Wavelength solution to convert. Must be 
                                expressed in Angstroms.
    - method                :   [int] 1 or 2.
                                if 1 : equation updated on 09/25/2021
                                if 2 : equation used before 09/25/2021    '''

    ## For SPIRou we can check that the wavelength make sense:
    if np.any(vacuum_lambdas<8000) or np.any(vacuum_lambdas>29000) :
        warnings.warn("WARNING: Are you sure you inputted ANGSTROM values for the wavelengths?")

    s = 10*4/vacuum_lambdas
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    lambdas = vacuum_lambdas / n
    return lambdas

def resample_vel_interp(wvl, flux, vel=None, kind='linear'):
    '''Same as resample_vel but does not use fourier'''
    # Estimate the step in wavelength:
    wvlstep = wvl[1] - wvl[0]
    # If no velocity step requested, take that from the center of the array
    if vel is None:
        vel = wvlstep/wvl[len(wvl)//2]*cst.c*1e-3 # km/s
    # We now define the new grid of wavelength that is regular in speed
    outwvl = effects._sampling_uniform_in_velocity(wvl[0], wvl[-1], vel)
    # And we retrieve the associated flux from simple interpolation
    if kind=='linear':
        outflux = np.interp(outwvl, wvl, flux)
    else:
        outflux = interp1d(wvl, flux, kind=kind)(outwvl)
    return outwvl, outflux