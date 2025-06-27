# ╔═══╗  ╔═══╗  ╔═══╗  ╔═══╗
# ║╔═╗║  ║╔═╗║  ║╔═╗║  ║╔═╗║
# ║║ ║║  ║╚══╗  ║║ ║║  ║╚═╝║
# ║╚═╝║  ╚══╗║  ║╚═╝║  ║╔══╝
# ║╔═╗║╔╗║╚═╝║╔╗║╔═╗║╔╗║║   ╔╗
# ╚╝ ╚╝╚╝╚═══╝╚╝╚╝ ╚╝╚╝╚╝   ╚╝


# ╔═══╗    ╔═══╗             ╔╗         ╔╗     ╔═══╗         ╔╗
# ║╔═╗║    ║╔═╗║            ╔╝╚╗        ║║     ║╔═╗║         ║║
# ║║ ║║    ║╚══╗╔══╗╔══╗╔══╗╚╗╔╝╔═╗╔══╗ ║║     ║║ ║║╔═╗ ╔══╗ ║║ ╔╗ ╔╗╔══╗╔╗╔══╗
# ║╚═╝║    ╚══╗║║╔╗║║╔╗║║╔═╝ ║║ ║╔╝╚ ╗║ ║║     ║╚═╝║║╔╗╗╚ ╗║ ║║ ║║ ║║║══╣╠╣║══╣
# ║╔═╗║    ║╚═╝║║╚╝║║║═╣║╚═╗ ║╚╗║║ ║╚╝╚╗║╚╗    ║╔═╗║║║║║║╚╝╚╗║╚╗║╚═╝║╠══║║║╠══║
# ╚╝ ╚╝    ╚═══╝║╔═╝╚══╝╚══╝ ╚═╝╚╝ ╚═══╝╚═╝    ╚╝ ╚╝╚╝╚╝╚═══╝╚═╝╚═╗╔╝╚══╝╚╝╚══╝
#               ║║                                              ╔═╝║
#               ╚╝                                              ╚══╝
#                                ╔═══╗          ╔╗
#                                ║╔═╗║          ║║
#                                ║╚═╝║╔╗╔══╗╔══╗║║ ╔╗╔═╗ ╔══╗
#                                ║╔══╝╠╣║╔╗║║╔╗║║║ ╠╣║╔╗╗║╔╗║
#                                ║║   ║║║╚╝║║║═╣║╚╗║║║║║║║║═╣
#                                ╚╝   ╚╝║╔═╝╚══╝╚═╝╚╝╚╝╚╝╚══╝
#                                       ║║
#                                       ╚╝


'''Object for a spectral analysis'''

################################
#### ---- DEPENDENCIES ---- ####
################################
from astropy.io import fits
import numpy as np
from asap.guess_vrad import guess_vrad_5 as guess_vrad
from asap import params
from asap import analysis_tools as tls
from asap import line_selection_tools as line_tools
from asap.spectral_analysis_pack import wrap_function_fine_linear_4d
from asap.spectral_analysis_pack import broaden_spectra
from asap.spectral_analysis_pack import veiling_function
from asap import effects as effects
import time
from multiprocessing import Pool
import emcee
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser, ExtendedInterpolation
from IPython import embed
from astroquery.simbad import Simbad
from asap import logg_tools as ltls
from asap import normalization_tools as norm_tools
from numba import jit
import h5py
from importlib.resources import files

def read_res(filename):
    '''This function is designed to parse the results in a typical raw output
    file.'''
    f = open(filename, 'r')
    for i, line in enumerate(f.readlines()):
        sl = line.split()
        if i==0:
            coeffs = [float(sl[i]) for i in range(len(sl))]
        elif i==1:
            ecoeffs = [float(sl[i]) for i in range(len(sl))]
        elif i==2:
            T = float(sl[0]); L = float(sl[1]); M = float(sl[2]); A = float(sl[3])
        elif i==3:
            eT = float(sl[0]); eL = float(sl[1]); eM = float(sl[2]); eA = float(sl[3])
        elif i==4:
            substr = line.split(':')[-1]
            bf = float(substr.split()[0])
            dbf = float(substr.split()[1])
        elif i==5:
            substr = line.split(':')[-1]
            avb = float(substr.split()[0])
            davb = float(substr.split()[1])
        elif i==6:
            if 'vb' not in line: ## sanity check
                print('read_res issue; vb not found in line {}'.format(i))
            substr = line.split(':')[-1]
            vb = float(substr.split()[0])
            dvb = float(substr.split()[1])
        elif i==7:
            if 'RV' not in line: ## sanity check
                print('read_res issue; RV not found in line {}'.format(i))
            substr = line.split(':')[-1]
            guessrv = float(substr.split()[0].replace('[', '').replace(']', ''))
        elif i==8:
            if 'RV' not in line: ## sanity check
                print('read_res issue; RV not found in line {}'.format(i))
            substr = line.split(':')[-1]
            rv = float(substr.split()[0])
            drv = float(substr.split()[1])
        elif i==9:
            if 'vsini' not in line: ## sanity check
                print('read_res issue; vsini not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            vsini = float(substr.split()[0])
            dvsini = float(substr.split()[1])
        elif i==10:
            if 'vmac' not in line: ## sanity check
                print('read_res issue; vmac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            vmac = float(substr.split()[0])
            dvmac = float(substr.split()[1])
            vmacMode = line.split(':')[0].split('[')[-1].replace(']', '')
        elif i==13:
            if 'nb. of points' not in line.lower(): ## sanity check
                print('read_res issue; nb. of points not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            nbOfPoints = round(float(substr.strip()))
        elif i==14:
            if 'normfactor' not in line.lower(): ## sanity check
                print('read_res issue; normfactor not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            normfactor = float(substr.strip())
        elif i==15:
            if 'veilingfac' not in line.lower(): ## sanity check
                print('read_res issue; veilingfac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            strvalues = substr.split()
            veilingfac = []
            for value in strvalues:
                veilingfac.append(float(value))
            veilingfac = np.array(veilingfac)
        elif i==16:
            if 'veilingfac' not in line.lower(): ## sanity check
                print('read_res issue; veilingfac not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            strvalues = substr.split()
            eveilingfac = []
            for value in strvalues:
                eveilingfac.append(float(value))
            eveilingfac = np.array(eveilingfac)
        elif i==17:
            if 'lum' not in line.lower(): ## sanity check
                print('read_res issue; lum not found in line {}'.format(i))
                print(line)
            substr = line.split(':')[-1]
            lum = float(substr.split()[0])
            dlum = float(substr.split()[1])
        elif i==18:
            if 'Mk' not in line: ## sanity check
                print('read_res issue; Mk not found in line {}'.format(i))
            substr = line.split(':')[-1]
            absmk = float(substr.split()[0])
            dabsmk = float(substr.split()[1])
        elif i==19:
            if 'dist' not in line: ## sanity check
                print('read_res issue; dist not found in line {}'.format(i))
            substr = line.split(':')[-1]
            dist = float(substr.split()[0])
            ddist = float(substr.split()[1])
    f.close()

    output = {"coeffs":coeffs, "ecoeffs":ecoeffs,
              "teff":T, "logg":L, "mh":M, "alpha":A,
              "dteff":eT, "dlogg":eL, "dmh":eM, "dalpha":eA,
              "vb":vb, "guessrv":guessrv, "rv":rv, "vsini":vsini, "vmac":vmac,
              "evb":dvb, "erv":drv, "evsini":dvsini, "evmac":dvmac,
              "lum":lum, "elum":dlum, "absmk":absmk, "eabsmk":dabsmk,
              "dist":dist, "edist":ddist,
              "bf":bf, "dbf":dbf,
              "avb":avb, "davb":davb,
              "nbpoints": nbOfPoints,
              "normfactor": normfactor,
              "veilingfac": veilingfac,
              "eveilingfac": eveilingfac,
              }
    return output

def readstr(instr):
    '''Small helper function to check that the strings are correctly formatted
    in the config files.
    The function checks that these are not ' of " in the passed string.'''
    
    if ("'" in instr) | ("\"" in instr):
        print("Warning, removed quoting marks in config strings.")
        outstr = instr.replace("'", "").replace("\"", "")
    return outstr

def configcleanup(config):
    '''looks at all the strings in the config file and removes quotation marks'''
    for key in config.keys():
        for subkey in config[key].keys():
            instr = config[key][subkey]
            if ("'" in instr) | ("\"" in instr):
                print("Warning, removed quoting marks in config strings.")
                outstr = instr.replace("'", "").replace("\"", "")
                config[key][subkey] = outstr
    return config

@jit(nopython=params.JIT, cache=True)
def pre_norm_grid_numba(d1, d2, d3, d4, d5, d6, d7,
                  teffs, loggs, mhs, alphas, bs,
                  nwvls, grid_n):
    '''Run a moving median through the observations'''
    rollmed_window = 500
    rollmed_btd = 0
    rollmed_p = 90
    # ncont = np.ones((d6, d7)) ## TODO: change this ugly hardcode
    ngrid_n = np.ones(np.shape(grid_n))
    # ngrid_n = np.array(np.shape(grid_n), dtype=float)
    ntot = d1*d2*d3*d4*d5
    n = 0
    for it in range(d1):#, teff in enumerate(teffs):
        for il in range(d2):#, logg in enumerate(loggs):
            for im in range(d3):#, mh in enumerate(mhs):
                for ia in range(d4):#, alpha in enumerate(alphas):
                    for ib in range(d5):#, B in enumerate(bs):
                        for r in range(d6):
                        # for r in range(d6):
                            _wvl = nwvls[r]
                            _flux = grid_n[ib, it, il, im, ia, r]
                            _diff = np.diff(_wvl)
                            wvlstep = _diff[len(_diff)//2]
                            dspeed = wvlstep / _wvl[len(_diff)//2] * 3*1e5
                            _nbins = rollmed_window/dspeed
                            _window = int(_nbins)
                            _, _ncont = norm_tools.moving_median(_flux, _window,
                                                                btd=rollmed_btd, p=rollmed_p)
                            grid_n[ib, it, il, im, ia, r] = _flux / _ncont
                        # ncont = np.array(ncont)
                        n += 1
                        # print('Flattening... {:.2f} %'.format(n/ntot*100), end='\r')
                        # print('Flattening... ' +  str(n/ntot*100) + ' %')
                        print(n/ntot*100)
            # print('coucou')
    return grid_n

def filter_lines(spectrum, models):
    '''First attempt to rejecting spurious pixels.
    Given a two dimentional grid of model spectra, well adjusted on the observed spectrum,
    for each observed spectrum we search for the min and max value of the pixel in the models
    and flag the model if it is out of bounds.'''

    modelsT = models.T

    mins = np.empty(spectrum.shape)
    maxs = np.empty(spectrum.shape)
    stds = np.empty(spectrum.shape)
    mask = np.ones(spectrum.shape, dtype=bool)
    for pix in range(len(spectrum)):
        _min = np.min(modelsT[pix])
        _max = np.max(modelsT[pix])
        _std = np.std(modelsT[pix])
        if (spectrum[pix]>_min) and (spectrum[pix]<_max):
            _mask = True
        else:
            _mask = False
        mins[pix] = _min
        maxs[pix] = _max
        stds[pix] = _std
        mask[pix] = _mask


    ## Now we can try to refine that a bit by saying:
    ## we should not have a very high logg and very high temperature
    ## We could try to vary one parameter around the expected optimal value.
    
    
    return mask

def adjust_errors_from_file(filename):
    '''Function returning the adjusted errors given an input fit-data.fits file.'''
    hdu = fits.open(filename)
    wvl = hdu['WVL'].data
    flx = hdu['FLUX'].data
    err = hdu['ERROR'].data
    flx_fit = hdu['FLUXFIT'].data
    fit = hdu['FIT'].data
    idxtofit = tuple(hdu['IDXTOFIT'].data)
    hdu.close()

    newerrors = np.copy(err[idxtofit])
    res = flx_fit[idxtofit] - fit[idxtofit]

    chi2 = np.sum((res/err[idxtofit])**2) ## Not normalized in any way, so very high chi2

    ## Weight the errors
    newerrors = err[idxtofit] + np.sqrt(1+(res/err[idxtofit])**2)
    chi2 = np.sum((res/newerrors)**2)
    ## Ensure the reduced chi2 is one (scale the weighted error bars)
    factor = chi2/len(newerrors)
    newerrors = newerrors  * np.sqrt(factor)

    chi2 = np.sum((res/newerrors)**2)

    # std = np.std(flx_fit[idxtofit] - fit[idxtofit])
    # med = np.median(flx_fit[idxtofit] - fit[idxtofit])
    # idxout = (res > (med+2*std)) | (res < (med-2*std))
    # newerrors[idxtofit][idxout] = np.nan

    return newerrors

class SpectralAnalysis:
    '''SpectralAnalysis class to perform MCMC analysis of a SPIRou spectrum.
    The class allows to check for the process self consistence (e.g. check
    that the shape of the grids loaded are in agreement with the shape of the 
    axis arrays).
    Values for the Teff, log(g), [M/H] and [a/Fe] arrays are defaulted. They
    can be changed using the update functions.'''

    def __init__(self, **kargs):
        self.dynesty = False
        self.teffs = np.arange(2700, 4000, 100); 
        self.loggs = np.arange(4.0, 6., .5)
        self.mhs = np.arange(-1.0, 1.0, .5); 
        self.alphas = np.arange(-0.25, 0.75, .25); 
        self.regions = np.array([[0,0]]) ## Initialization
        self.vinstru = 4.3 ## Initial guess instrumental width (gaussian)
        self.vb = 0 ## Broad. applied to spectrum (quadrat. added to 4.3km/s)
        self.fitmac = False ## Whether vmac is fitted
        self.vmac = 0 ## Macroturbulence velcotiy (km/s)
        self.vmacMode = 'rt' ## Macroturbulence model ('rt' or 'g')
        self.rv = 0 ## Radial velocity adjustment [km/s]. Default is 0.
        self.fitbroad = True ## Should we fit vb? Default is true
        self.fitrot   = False ## Should we fit rot? Default is true
        self.vsini    = 0 ## Default rotation value
        self.fitrv    = True ## Should we fit RV? Default is true
        self.d7 = 0 ## Initialization
        self.fit_factors = True ## Should we fit the filling factors?
        self.logCoeffs = False ## Should we search for coeffs in log? 
        self.fitDeriv = False ## By default, we fit the spectrum, not it's derivative
        if 'fit_factors' in kargs.keys(): 
            self.fit_factors = kargs['fit_factors']
        if self.fit_factors:
            self.bs = np.arange(0, 12, 2)
        else:
            self.bs = np.array([0])
        self.bfVals = np.array([]) ## initialize the bypass of magnetic values
        self.get_grid_dims()
        ## Default number of walkers and steps and ncores for MCMC
        self.nwalkers = 200
        self.nsteps = 2000
        self.ncores = 60
        self.star = 'nonamestar'
        self.adjcont = True # Adjust continuum. Default is True
        self.guessRV = True # Default is True
        self.guessed_rv = 0 # Default is 0
        self.resampleVel = True
        ## DEFAULT PATHS
        self.pathtodata = "./"
        self.pathtogrid = "./"
        self.linelist = "dummy.txt"
        self.normfacfile = "dummy.txt"
        self.opath = 'output_{}/'.format(self.star)
        ## MCMC options
        self.parallel = True
        ## Advance options
        self.debugMode = False
        self.savebackend = False
        self.minLineDepthFit = 1.0
        self.errType = 'std'
        ## Normalization factor used to in lnlike
        self.renorm = False
        self.normFactor = None
        self.coeffs = np.zeros(len(self.bs)); self.coeffs[0] = 1
        ## Atmospheric parameters
        self.fitTeff    = True
        self.fitLogg    = True
        self.fitMh      = True
        self.fitAlpha   = True
        self._T         = None
        self._T2        = None # second temperature for 2 Teff model
        self._L         = None
        self._M         = None
        self._A         = None
        self.fillTeffs  = np.array([1, 0])
        self.smoothSpectraVel = 0.0 ## initialize the velocity used to smooth the spectra
        self.smoothSpectra = False
        ## Define some variables to store chi2 and ln part of the likelihood
        self._res       = 0 ## That will store the chi2
        self._lncorr    = 0 ## This will store the ln(2.pi.err**2)
        #
        self.interpFunc = 'linear'
        ## Now we add a veiling factor as an option
        self.fitVeiling = False
        self.fitBands = 'YJHK' ## default if fitVeiling is true...
        self.nbFitVeil = len(self.fitBands)
        self.veilingBands = 'IYJHKL'
        self.veilingFac = np.array([0,0,0,0,0,0]) ## Default to zero for all bands
        self.veilingFacToFit = np.array([0,0,0,0]) ## Default to zero for all bands
        ## Declare magnitudes and distances attributes
        self.Mk = np.nan
        self.dMk = np.nan
        self.Mg = np.nan
        self.dMg = np.nan
        self.Mj = np.nan
        self.dMj = np.nan
        self.mk = np.nan
        self.dmk = np.nan
        self.mg = np.nan
        self.dmg = np.nan
        self.mj = np.nan
        self.dmj = np.nan
        self.dist = np.nan
        self.ddist = np.nan
        self.rL = np.nan
        self.drL = np.nan
        self.plx = np.nan
        self.dplx = np.nan
        self.sptype = 'None'
        ## Initialize trigger for automatic logg computation
        self.autoLogg = False
        self.errorsAdj = False ## Informational variable (only used to keep track)
        self.file_struc = '{}g{:0.1f}z{:0.2f}a{:.2f}b{:04.0f}p{:0.1f}rot{:0.2f}beta{:0.2f}.hdf5'
        self.instrument = 'spirou'
        self.AVAILABLE_PARAMS = ['a0', 'a2', 'a4', 'a6', 'a8', 'a10',
                                 'teff', 'logg', 'mh', 'alpha', 
                                 'vb', 'rv', 'vsini', 'vmac', 
                                 'rI', 'rY', 'rJ', 'rH', 'rK', 'rL',
                                 'teff2', 'fillteff2'] ## list of all paramters that could be fitted
        self.PARAMS_FIT = []
        self.PARAMS = {} ## list of parameters that are actually fitted

    def simbad_grep(self, _star=None):
        '''Function to grab and parse information on the star from SIMBAD
        for the star.'''
        ## Format input string for SIMBAD query
        if _star is None:
            _star = self.star
        _star = _star.lower()
        if 'pm' in _star:
            ## if your star is a PM* star, we may have a problem with the pluses, minuses etc.
            ## So we'll have to parse the name.
            ## Step one: get rid of underscore (separators)
            _star = _star.replace('_', '')
            ## Create a unique seperator between "PM" and the identifier
            _star = _star.replace('pm', 'pm_')
            ## Parse the name
            _halfstar = _star.split('_')
            _halfstar[1] = _halfstar[1].replace('p', '+') # This one will not work if pm is in the name.
            _halfstar[1] = _halfstar[1].replace('m', '-')
            _star = _halfstar[0]+'_'+_halfstar[1]
        if _star=='gl169_1a': _star = 'gl169.1a'
        _star = _star.replace('_', ' ') # remove the "_" for a space
        _star = _star.replace('gl', 'gl ')
        _star = _star.replace('gj', 'gj ') # Add space to gliese for query
        ## Add a known exception:
        ## Initialize SIMBAD object for query
        simbad = Simbad()
        simbad.add_votable_fields('pmra', 'pmdec', 'distance', 'rv_value', 
                                'plx', 'plx_error', 
                                'ra', 'dec',
                                'flux(G)', 'flux_error(G)'
                                'flux(K)', 'flux_error(K)', 
                                'flux(V)', 'flux_error(V)',
                                'flux(r)', 'flux_error(r)',
                                'flux(J)', 'flux_error(J)',
                                'flux_bibcode(J)', 'sptype'
                                )
        ## Query magnitudes from SIMBAD
        try:
            result_ids = simbad.query_object(_star)
        except:
            return None
        if result_ids is None: 
            print('Warning: could not find star on Simbad')
            return None
        p = result_ids['PLX_VALUE'][0]; dp = result_ids['PLX_ERROR'][0]
        mg = float(result_ids['FLUX_G'][0]); dmg = float(result_ids['FLUX_ERROR_G'][0])
        mj = float(result_ids['FLUX_J'][0]); dmj = float(result_ids['FLUX_ERROR_J'][0])        
        mk = float(result_ids['FLUX_K'][0]); dmk = float(result_ids['FLUX_ERROR_K'][0])
        sptype = result_ids['SP_TYPE'][0]   
        ## Convert to aboslute magnitude and distance
        d, dd = ltls.dist_parsec(p, dp)
        Mg, dMg = ltls.convert_mag(mg, d, dmg, dd)
        Mj, dMj = ltls.convert_mag(mj, d, dmj, dd)
        Mk, dMk = ltls.convert_mag(mk, d, dmk, dd)
        ## Known exceptions?
        if _star=='gj 205':
            print('SpectraAnalysis')
            print('CAUTION VALUES HARDCODED FOR GL205')
            print('CAUTION VALUES HARDCODED FOR GL205')
            print('CAUTION VALUES HARDCODED FOR GL205')
            print('CAUTION VALUES HARDCODED FOR GL205')
            print(dMg, dMj, dMk)
            dMk = dMj = 0.06
        ## Bolometric correction & luminosity
        bcg, dbcg = ltls.BCg_cifuentes(Mg, Mj, dMg, dMj)
        bcj, dbcj = ltls.BCj_cifuentes(Mg, Mj, dMg, dMj)
        bMsun = 4.74 ## Sun bolometric lum
        ## Compute the bolometric magnitude: either from G of J
        ## J appears to give lower errors in the end
        bM = Mg + bcg
        dbM = np.sqrt(dMg**2 + dbcg**2)
        bM = Mj + bcj
        dbM = np.sqrt(dMj**2 + dbcj**2)
        ## Compute the luminosity
        rL = 10**((bM - bMsun)/(-2.5)) # Bolometric lum rel. to Sun.
        drL = abs(np.log(10)/(-2.5) * 10**((bM - bMsun)/(-2.5)) * dbM)
        drL = abs(1/np.log(10)/(2.5) * 10**((bM - bMsun)/(-2.5)) * dbM)
        ## Now in log
        self.rL_log = np.log10(rL)
        self.drL_log = 1/(rL*np.log(10))*drL
        ## Set attributes for global variables
        self.set_dist(d, dd) # distance
        self.set_Mk(Mk, dMk) # absolute K band magnitude
        self.set_Mg(Mg, dMg) # absolute K band magnitude
        self.set_Mj(Mj, dMj) # absolute K band magnitude
        self.set_mk(mk, dmk) # absolute K band magnitude
        self.set_mg(mg, dmg) # absolute K band magnitude
        self.set_mj(mj, dmj) # absolute K band magnitude
        self.set_rL(rL, drL) # bolometric luminosity relative to Sun
        self.set_plx(p, dp) # bolometric luminosity relative to Sun
        self.sptype = sptype

    def compute_logg(self, teff, mh, dteff=0):
        '''Function computes logg from empirical relation (Mann et al., 2019) and Boltzmann law'''
        if np.isnan(self.Mk) | np.isnan(self.rL):
            ostr = "NaN in magnitudes and/or luminosity." \
                   + " Did you run simbad_grep() ?"
            raise Exception(ostr)
        ## Compute radius from Boltzmann law and effective temperature + lum
        _rad, _drad = ltls.boltzmannRadius(teff, self.rL, dteff, self.drL)
        ## Compute mass from mass-Mk magnitude relation Mann et al. (2019)
        _mass, _dmass = ltls.mass_lum_mann19(self.Mk, self.dMk, mh)
        ## Compute log(g)
        _logg, _dlogg = ltls.compute_logg(_mass, _rad, _dmass, _drad)
        # self._L = _logg
        return _logg

    def read_config(self, config_file=None):
        '''This function reads the config.ini file and sets the attributes
        based on the user inputs. The config.ini file MUST be in the same
        format as that generated by gen_config.py.'''
        
        if config_file is None:
            locpath = os.getcwd()
            config_file = locpath + '/config.ini'
        if not os.path.isfile(config_file):
            errstring = "Config file missing.\n" \
                        + "You can use the config file generator to create" \
                        + " an example file. Run gen_config.py to do so."       
            raise Exception(errstring)
        #
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_file)

        ## ----------------------------------------------------
        ## READ THE OPTIONS
        #
        ## MAIN TRIGGERS
        fitBroad    = config.getboolean('MAIN', 'fitBroad')
        fitRV       = config.getboolean('MAIN', 'fitRV')
        rv          = float(config['MAIN']['rv'])
        fitFields   = config.getboolean('MAIN', 'fitFields')
        fillFactors = config['MAIN']['fillFactors']
        try:
            fieldsArr = config['MAIN']['fieldsArr'].split()
            bs = np.arange(float(fieldsArr[0])/1000, float(fieldsArr[1])/1000, float(fieldsArr[2])/1000)
            self.update_bs(bs)
            print(f'self.bs updated to: {self.bs}')
        except:
            pass
        try:
            logCoeffs   = config.getboolean('MAIN', 'logCoeffs')
        except:
            logCoeffs = self.logCoeffs
        bfVals = self.bfVals
        if fillFactors.lower().strip()=='none':
            fillFactors = np.array([1.])
            fitFields = False ## Prevent the situation with no magnetic coeff and a fitFields
        else:
            fillFactors = fillFactors.replace('[', '').replace(']', '')
            if ',' in fillFactors: fillFactors = fillFactors.split(',')
            else: fillFactors = fillFactors.split()
            nbFields    = len(fillFactors)
            fillFactors = [float(fillFactors[i]) for i in range(nbFields)]
            fillFactors = np.array(fillFactors)
        try:
            bfVals = config['MAIN']['bfVals']
            bfVals = bfVals.replace('[', '').replace(']', '')
            bfVals = bfVals.replace(',', '')
            bfVals = bfVals.split()
            nbFields    = len(bfVals)
            bfVals = [float(bfVals[i]) for i in range(nbFields)]
            bfVals = np.array(bfVals)
        except:
            pass
        fitMac      = config.getboolean('MAIN', 'fitMac')
        vmac        = float(config['MAIN']['vmac'])
        vmacMode    = config['MAIN']['vmacMode']
        fitRot      = config.getboolean('MAIN', 'fitRot')
        vsini       = float(config['MAIN']['vsini'])
        try:
            vinstru   = float(config['MAIN']['vinstru'])
        except:
            vinstru  = 4.3
        try:
            instrument = config['MAIN']['instrument']
        except:
            instrument = self.instrument
        renorm      = config.getboolean('MAIN', 'reNorm')
        normFactor  = config['MAIN']['normFactor']
        if "none" in normFactor.lower():
            normFactor = 1
        else:
            normFactor = float(normFactor)
        if config.has_option('MAIN', 'veilingFac'):
            strveiling = config['MAIN']['veilingFac'].split()
            veilingFac = np.array([float(strveiling[i]) for i in range(len(strveiling))])
            if len(veilingFac)==1: veilingFac = veilingFac[0] ## Backward compatibility
            ## If we have that option, we MUST have the veilingBands passed as well
            veilingBands = config['MAIN']['veilingBands'].replace("'", "")
            veilingBands = veilingBands.replace(",", "").replace(" ", "")
            veilingBands = veilingBands.strip() ## This is a chain of characters
        else:
            veilingFac = self.veilingFac
            veilingBands = self.veilingBands
            print('Caution - veilingFac missing from config file')
        if config.has_option('MAIN', 'fitVeiling'):
            fitVeiling = config.getboolean('MAIN', 'fitVeiling')
            ## If we have that option, we MUST have the fitBands passed as well
            fitBands = config['MAIN']['fitBands'].replace("'", "")
            fitBands = fitBands.replace(",", "").replace(" ", "")
            fitBands = fitBands.strip() ## This is a chain of characters
        else:
            fitVeiling = self.fitVeiling
            fitBands = self.fitBands
            print('Caution - fitVeiling missing from config file')
        ## ATMOSPHERIC PARAMETERS
        fitTeff     = config.getboolean('ATMO', 'fitTeff')
        try:
            fitTeff2 = config.getboolean('ATMO', 'fitTeff2') ## For the two temperature model
        except:
            fitTeff2 = False
        fitLogg     = config.getboolean('ATMO', 'fitLogg')
        try:
            autoLogg = config.getboolean('ATMO', 'autoLogg')
        except:
            autoLogg = self.autoLogg
        fitMh       = config.getboolean('ATMO', 'fitMh')
        fitAlpha    = config.getboolean('ATMO', 'fitAlpha')
        _T          = config['ATMO']['teff']
        try:
            _T2     = config['ATMO']['teff2']
        except:
            _T2     = "none"
        _L          = config['ATMO']['logg']
        _M          = config['ATMO']['mh']
        _A          = config['ATMO']['alpha']
        if "none" in _T.lower(): _T = None
        else: _T = float(_T) 
        if "none" in _T2.lower(): _T2 = None
        else: _T2 = float(_T2)
        _L          = config['ATMO']['logg']
        if "none" in _L.lower(): _L = None
        else: _L = float(_L) 
        _M          = config['ATMO']['mh']
        if "none" in _M.lower(): _M = None
        else: _M = float(_M) 
        _A          = config['ATMO']['alpha']
        if "none" in _A.lower(): _A = None
        else: _A = float(_A)
        try:
            fillTeffsString = config['ATMO']['fillTeffs']
        except:
            fillTeffsString = "none"
        if "none" in fillTeffsString.lower(): fillTeffs = self.fillTeffs
        else:
            fillTeffsArray = fillTeffsString.split()
            if len(fillTeffsArray)!=2: 
                raise Exception('fillTeffs should provide two values')
            else:
                fillTeffs = np.array([float(fillTeffsArray[0]), float(fillTeffsArray[1])])
                if round(np.sum(fillTeffs),2)!=1.00:
                    raise Exception('The sum of fillTeffs must be 1')

        try:
            _Tarray     = config['ATMO']['teffArray'].replace(',', '')
            _Larray     = config['ATMO']['loggArray'].replace(',', '')
            _Marray     = config['ATMO']['mhArray'].replace(',', '')
            _Aarray     = config['ATMO']['alphaArray'].replace(',', '')
        except:
            _Tarray     = "None"
            _Larray     = "None"
            _Marray     = "None"
            _Aarray     = "None"
        if "none" in _Tarray.lower(): _Tarray = None
        elif len(_Tarray.split())!=3: 
            raise Exception("config: teffArray specifications not understood")
        else:
            _array = [int(float(_Tarray.split()[i])) for i in range(3)]
            teffarray = np.arange(_array[0], _array[1]+_array[2], _array[2])
        if "none" in _Larray.lower(): _Larray = None
        elif len(_Larray.split())!=3: 
            raise Exception("config: teffArray specifications not understood")
        else:
            _array = [float(_Larray.split()[i]) for i in range(3)]
            loggarray = np.arange(_array[0], _array[1]+_array[2], _array[2])
        if "none" in _Marray.lower(): _Marray = None
        elif len(_Marray.split())!=3: 
            raise Exception("config: teffArray specifications not understood")
        else:
            _array = [float(_Marray.split()[i]) for i in range(3)]
            mharray = np.arange(_array[0], _array[1]+_array[2], _array[2])
        if "none" in _Aarray.lower(): _Aarray = None
        elif len(_Aarray.split())!=3: 
            raise Exception("config: teffArray specifications not understood")
        else:
            _array = [float(_Aarray.split()[i]) for i in range(3)]
            alphaarray = np.arange(_array[0], _array[1]+_array[2], _array[2])
        ## ADVANCED OPTIONS
        adjCont     = config.getboolean('ADVANCED', 'adjCont')
        guessRV     = config.getboolean('ADVANCED', 'guessRV')
        resampleVel = config.getboolean('ADVANCED', 'resampleVel')
        debugMode   = config.getboolean('ADVANCED', 'debugMode')
        if config.has_option('ADVANCED', 'interpFunc'):
            interpFunc = config['ADVANCED']['interpFunc']
        else:
            interpFunc = 'linear'
        try:
            fitDeriv    = config.getboolean('ADVANCED', 'fitDeriv')
        except:
            fitDeriv    = False ## Default value
        if config.has_option('ADVANCED', 'minLineDepthFit'):
            minLineDepthFit = float(config['ADVANCED']['minLineDepthFit'])
        else:
            minLineDepthFit = 1.0
        if config.has_option('ADVANCED', 'smoothSpectraVel'):
            smoothSpectraVel = float(config['ADVANCED']['smoothSpectraVel'])
        else:
            smoothSpectraVel = 0.0
        if config.has_option('ADVANCED', 'errType'):
            errType = config['ADVANCED']['errType']
        else:
            errType = 'propag'
        ## PATHS SETUP
        pathtogrid  = config['PATHS']['pathToGrid']
        pathtodata  = config['PATHS']['pathToData']
        linelist    = config['PATHS']['lineListFile']
        try: ## Check that the attribute exists for backwards compatibility
            normfacfile = config['PATHS']['normFactorFile']
            if (normfacfile.strip()=="") | ("none" in normfacfile.lower()):
                # normfacfile = paths.support_data \
                #             + 'spectral_analysis/normFactors.txt'
                normfacfile = files("asap.support_data").joinpath("normFactors.txt")
        except:
            # normfacfile = paths.support_data \
            #             + 'spectral_analysis/normFactors.txt'
            normfacfile = files("asap.support_data").joinpath("normFactors.txt")

        ## MCMC OPTIONS
        nbWalkers   = int(config['MCMC']['nbWalkers'])
        nbSteps     = int(config['MCMC']['nbSteps'])
        parallel    = config.getboolean('MCMC', 'parallel')
        nbCores     = int(config['MCMC']['nbCores'])
        saveBackend = config.getboolean('MCMC', 'saveBackend')
        ## ----------------------------------------------------
        ## SET UP ENVIRONMENT VARIABLES
        #
        # MAIN TRIGGERS
        self.set_fitbroad(fitBroad) ## Whether or not to fit vb
        self.set_fitrv(fitRV) ## Whether or not to fit vb
        self.set_rv(rv)
        self.set_fitFields(fitFields)
        if not fitFields:
            ## We may want to provide values and not fit them
            # bs = np.arange(len(fillFactors))*2
            bs = self.bs
            nfields = len(fillFactors)
            if len(bs)>nfields:
                bs = bs[:nfields]
            print('These are the fields we consider:{}'.format(bs))
            self.update_bs(bs) ## 
            # self.update_bs(0) ## Only one value of magnetic field == non-magnetic case
            self.update_fillFactors(fillFactors)
        else:
            bs = self.bs
            nfields = len(fillFactors)
            if len(bs)>nfields:
                bs = bs[:nfields]
            self.update_bs(bs) ## Only one value of magnetic field == non-magnetic case
            self.update_fillFactors(fillFactors)
        if len(bfVals)==0:
            pass
        else:
            bs = bfVals
            self.update_bs(bs)                

        self.set_fitrot(fitRot)
        self.set_vsini(vsini)
        self.set_vinstru(vinstru)
        self.set_fitmac(fitMac)
        self.set_vmac(vmac)
        self.set_vmacMode(vmacMode)
        self.set_renorm(renorm)
        self.set_normFactor(normFactor)
        self.set_logCoeffs(logCoeffs)
        ## ATMOSPHERIC PARAMETERS
        self.set_fitTeff(fitTeff)
        self.set_fitTeff2(fitTeff2) ## for 2 teff model
        self.set_fitLogg(fitLogg)
        self.set_autoLogg(autoLogg)
        self.set_fitMh(fitMh)
        self.set_fitAlpha(fitAlpha)
        self.set_teff(_T)
        self.set_teff2(_T2) ## for 2 teff model
        self.set_logg(_L)
        self.set_mh(_M)
        self.set_alpha(_A)
        self.update_fillTeffs(fillTeffs)
        if _Tarray is not None:
            self.update_teffs(teffarray)
        if _Larray is not None:
            self.update_loggs(loggarray)
        if _Marray is not None:
            self.update_mhs(mharray)
        if _Aarray is not None:
            self.update_alphas(alphaarray)
        ## ADVANCED OPTIONS
        self.set_adjcont(adjCont)
        self.set_guessRV(guessRV)
        self.set_resampleVel(resampleVel)
        self.set_debugMode(debugMode)
        self.set_interpFunc(interpFunc)
        self.set_fitDeriv(fitDeriv)
        self.set_minLineDepthFit(minLineDepthFit)
        self.set_smoothSpectraVel(smoothSpectraVel)
        self.set_errType(errType)
        ## PATHS SETUP
        self.set_pathtogrid(pathtogrid)
        self.set_pathtodata(pathtodata)
        self.set_linelist(linelist)
        self.set_normfacfile(normfacfile)
        #
        self.set_nwalkers(nbWalkers)
        self.set_nsteps(nbSteps)
        self.set_parallel(parallel)
        self.set_ncores(nbCores) ## Number of cores to use for the MCMC
        self.set_savebackend(saveBackend) ## Number of cores to use for the MCMC
        #
        self.set_veilingBands(veilingBands) ## veiling factor
        self.set_veilingFac(veilingFac) ## veiling factor
        self.set_fitVeiling(fitVeiling) ## veiling factor
        ## Here is a check: the veilingFactors cannot be different from the number of fitted bands.
        if not self.fitVeiling:
            fitBands = ""
        self.set_fitBands(fitBands) ## veiling factor
        if len(self.veilingBands)!=len(self.veilingFac):
            print('Fatal error: shape of provided veiling factors and number of bands mistmach.')
            print('Requested bands: {}'.format(self.veilingBands))
            print('With initial factors: {}'.format(self.veilingFac))
            exit()
        #
        self.set_instrument(instrument)
        #
        self.init_PARAMS() ## Creates a list of keys and default values

    def init_PARAMS(self):
        '''Updates the list of AVAILABLE PARAMS (to match the number of coefficents and veiling factors)
        Creates a directory  with the default values and or the values from a '''
        ## Here again the number of fields must be the nunber that we fit +1

        PARAMS_FIT = []
        AVAILABLE_PARAMS = [] ## parameters for which we MUST have a value (otherwise code breaks)
        ## For the fields, if we fit, we fit all of them
        nbOfFields = len(self.bs)
        for i in range(nbOfFields):
            AVAILABLE_PARAMS.append('a{}'.format(i))
            if self.fitFields:
                PARAMS_FIT.append('a{}'.format(i))
        #
        AVAILABLE_PARAMS.append('teff')
        if (self.fitTeff):
            PARAMS_FIT.append('teff')
        AVAILABLE_PARAMS.append('logg')
        if (self.fitLogg):
            PARAMS_FIT.append('logg')
        AVAILABLE_PARAMS.append('mh')
        if (self.fitMh):
            PARAMS_FIT.append('mh')
        AVAILABLE_PARAMS.append('alpha')
        if (self.fitAlpha):
            PARAMS_FIT.append('alpha')
        AVAILABLE_PARAMS.append('vb')
        if (self.fitbroad):
            PARAMS_FIT.append('vb')
        AVAILABLE_PARAMS.append('rv')
        if (self.fitrv):
            PARAMS_FIT.append('rv')
        AVAILABLE_PARAMS.append('vsini')
        if self.fitrot:
            PARAMS_FIT.append('vsini')
        AVAILABLE_PARAMS.append('vmac')
        if self.fitmac:
            PARAMS_FIT.append('vmac')
        ## For the veiling, we can choose WHICH band we fit so we must use the two seperate variables
        for band in self.veilingBands: ## All the bands even if they are not fitted
            AVAILABLE_PARAMS.append('r{}'.format(band))
        if self.fitVeiling:
            for band in self.fitBands: ## All the bands even if they are not fitted
                PARAMS_FIT.append('r{}'.format(band))
        AVAILABLE_PARAMS.append('teff2')
        AVAILABLE_PARAMS.append('fillteff_0')
        AVAILABLE_PARAMS.append('fillteff_1')
        if self.fitTeff2:
            PARAMS_FIT.append('teff2')
            PARAMS_FIT.append('fillteff_0')
            PARAMS_FIT.append('fillteff_1')
        self.AVAILABLE_PARAMS = AVAILABLE_PARAMS
        self.PARAMS_FIT = PARAMS_FIT

        ## Now intialize the dictionaries of values
        ## That really should replace any attributes in the future...
        
        list_of_params_raw=[self.coeffs, [self._T], [self._L], [self._M], [self._A], 
                        [self.vb], [self.rv], [self.vsini], [self.vmac],
                        self.veilingFac, [self._T2], self.fillTeffs]
        list_of_params = []
        for element in list_of_params_raw:
            for subelement in element:
                list_of_params.append(subelement)

        # list_of_params = np.concatenate(list_of_params, -1)
        
        if len(self.AVAILABLE_PARAMS)!=len(list_of_params):
            raise Exception('SpectraAnalysis.get_PARAMS: issue creating the PARAMS dict.')

        for ik, key in enumerate(self.AVAILABLE_PARAMS):
            self.PARAMS[key] = list_of_params[ik]
            self.PARAMS['e_'+key] = 0

    def get_PARAMS(self, mcmcs=None, emcmcs=None):
        '''Updates the list of AVAILABLE PARAMS (to match the number of coefficents and veiling factors)
        Creates a directory  with the default values and or the values from a '''
        ## Here again the number of fields must be the nunber that we fit +1
    
        PARAMS = self.PARAMS.copy()

        for ip, param in enumerate(self.PARAMS_FIT):
            PARAMS[param] = mcmcs[ip]
            PARAMS['e_'+param] = emcmcs[ip]
        return PARAMS

    def set_PARAMS(self, mcmcs, emcmcs):
        PARAMS =  self.get_PARAMS(mcmcs, emcmcs)
        self.PARAMS = PARAMS

    def get_grid_dims(self):
        '''Compute the dimensions of the grid used to store the data.
        Dimensions labeled from d1 to d7 are respectively the number of Teff,
        number of log(g), number of [M/H], number of [a/Fe], number of B,
        number of regions (estimated from input region file, initialized to 0),
        and the number of data points in a region.'''
        self.d1 = len(self.teffs); self.d2 = len(self.loggs); 
        self.d3 = len(self.mhs); self.d4 = len(self.alphas); 
        self.d5 = len(self.bs); self.d6 = len(self.regions)
        # self.d7 = 0 ## Initializization
        self.griDims = (self.d1, self.d2, self.d3, self.d4, 
                        self.d5, self.d6, self.d7)

    ###########################
    #### ---- SETTERS ---- ####
    ###########################
    ## Helper funcions to set the teffs, loggs, mhs, alphas and bs arrays.
    ## Please use these. 
    def update_teffs(self, teffs):
        ## The input must be an array to avoid issues with len() later.
        if isinstance(teffs, int) | isinstance(teffs, float):
            teffs = np.array([teffs])
        self.teffs = teffs
        self.get_grid_dims()
    def update_loggs(self, loggs):
        ## The input must be an array to avoid issues with len() later.
        if isinstance(loggs, int) | isinstance(loggs, float):
            loggs = np.array([loggs])
        self.loggs = loggs
        self.get_grid_dims()
    def update_mhs(self, mhs):
        ## The input must be an array to avoid issues with len() later.
        if isinstance(mhs, int) | isinstance(mhs, float):
            mhs = np.array([mhs])
        self.mhs = mhs
        self.get_grid_dims()
    def update_alphas(self, alphas):
        ## The input must be an array to avoid issues with len() later.
        if isinstance(alphas, int) | isinstance(alphas, float):
            alphas = np.array([alphas])
        self.alphas = alphas
        self.get_grid_dims()
    def update_bs(self, bs):
        ## The input must be an array to avoid issues with len() later.
        if isinstance(bs, int) | isinstance(bs, float):
            bs = np.array([bs])
        self.bs = bs
        self.get_grid_dims()
    def update_fillFactors(self, fillFactors):
        '''WARNING, this constructor sets all filling factors including the fits zero component.'''
        if len(fillFactors)!=(len(self.bs)):
            raise Exception('Provided fillFactors inconsitent with requested number of fields. '
                            + 'Did you run self.update_bs?')
        ## The input must be an array to avoid issues with len() later.
        if isinstance(fillFactors, int) | isinstance(fillFactors, float):
            fillFactors = None
        if abs(np.sum(fillFactors)-1.)>0.01:
            # print('Provided fillFactors do not sum up to one. Defaulting to 0.1 for magnetic fields')
            print('Provided fillFactors do not sum up to one. Deriving non-mag from mag')
            if len(self.bs)==1:
                fillFactors = None
            else:
                # fillFactors = np.ones(len(self.bs)) * 0.01 ## Only magnetic components
                fillFactors[0] = 1. - np.sum(np.sum(fillFactors[1:]))
        self.coeffs = fillFactors
    def logmode(self, trig):
        if type(trig)!=bool:
            print("logmode take one bool argument.")
            print("logcoeffs is currently set to {}.".format(self.logCoeffs))
        else:
            self.logCoeffs = trig
    def set_nwalkers(self, nwalkers):
        self.nwalkers = nwalkers
        return self.nwalkers
    def set_nsteps(self, nsteps):
        self.nsteps = nsteps
        return self.nsteps
    def set_star(self, star):
        self.star = star.lower()
        # self.opath = 'output_{}/'.format(self.star)
    def set_ncores(self, ncores):
        self.ncores = int(ncores)
        return ncores
    def set_opath(self, opath):
        self.opath = opath
    def set_fitbroad(self, fitbroad):
        self.fitbroad = fitbroad ## Must be a boolean
    def set_vb(self, vb):
        self.vb = vb
    def set_fitrv(self, fitrv):
        self.fitrv = fitrv ## Must be a boolean
    def set_adjcont(self, adjcont):
        self.adjcont = adjcont ## Must be a boolean
    def set_guessRV(self, guessRV):
        self.guessRV = guessRV ## Must be a boolean
    def set_rv(self, rv):
        self.rv = rv ## Must be a boolean
    def set_fitFields(self, fitFields):
        self.fitFields = fitFields ## Must be a boolean
    def set_fitrot(self, fitrot):
        self.fitrot = fitrot ## Must be a boolean
    def set_vsini(self, vsini):
        self.vsini = vsini ## Must be a float
    def set_fitmac(self, fitmac):
        self.fitmac = fitmac ## Must be a boolean
    def set_vmac(self, vmac):
        self.vmac = vmac ## Must be a float
    def set_vinstru(self, vinstru):
        self.vinstru = vinstru
    def set_vmacMode(self, vmacMode): 
        if (vmacMode!='rt') & ((vmacMode!='g')):
            raise Exception('vmacMode not understood')
        self.vmacMode = vmacMode ## Must be a string
    def set_resampleVel(self, resampleVel):
        self.resampleVel = resampleVel ## Must be a boolean
    def set_pathtodata(self, pathtodata):
        self.pathtodata = pathtodata ## Must be a string
    def set_pathtogrid(self, pathtogrid):
        self.pathtogrid = pathtogrid ## Must be a string
    def set_linelist(self, linelist):
        self.linelist = linelist ## Must be a string
    def set_normfacfile(self, normfacfile):
        if not os.path.isfile(normfacfile): 
            print('Warning: requested normfacfile does not exists. Creating it.')
            open(normfacfile, 'w').close()
        self.normfacfile = normfacfile ## Must be a string
    def set_opath(self, opath):
        self.opath = opath ## Must be a string
    def set_debugMode(self, debug):
        self.debugMode = debug
    def set_parallel(self, parallel):
        self.parallel = parallel
    def set_savebackend(self, savebackend):
        self.savebackend = savebackend
    def set_normFactor(self, normFactor):
        self.normFactor = normFactor
    def set_renorm(self, renorm):
        self.renorm = renorm
    def set_fitTeff(self, fitTeff):
        self.fitTeff = fitTeff
    def set_fitTeff2(self, fitTeff2):
        self.fitTeff2 = fitTeff2
    def set_fitLogg(self, fitLogg):
        self.fitLogg = fitLogg
    def set_autoLogg(self, autoLogg):
        self.autoLogg = autoLogg
        ## To use autoLogg we must set fitLogg to False
        if self.autoLogg: self.set_fitLogg(False)
    def set_fitMh(self, fitMh):
        self.fitMh = fitMh
    def set_fitAlpha(self, fitAlpha):
        self.fitAlpha = fitAlpha
    def set_teff(self, _T):
        self._T = _T
    def set_teff2(self, _T2):
        self._T2 = _T2
    def set_logg(self, _L):
        self._L = _L
    def set_mh(self, _M):
        self._M = _M
    def set_alpha(self, _A):
        self._A = _A
    def update_fillTeffs(self, fillTeffs):
        self.fillTeffs = fillTeffs
    def set_interpFunc(self, interpFunc):
        self.interpFunc = interpFunc
    def set_veilingFac(self, veilingFac):
        if isinstance(veilingFac, float) | isinstance(veilingFac, int):
            n_veilingFac = np.ones(len(self.veilingBands))*veilingFac
            veilingFac = n_veilingFac
        self.veilingFac = veilingFac
    def set_veilingBands(self, veilingBands):
        self.veilingBands = veilingBands
    def set_fitBands(self, fitBands):
        self.fitBands = fitBands
        ## This is also where we must update the nbVeilFit and the veilingFacToFit
        self.nbFitVeil = len(fitBands)
        ## What are the coefficients for those bands?
        positions = np.empty(self.nbFitVeil, dtype=int)
        for ib, band in enumerate(self.fitBands):
            positions[ib] = self.veilingBands.find(band)
        self.veilingFacToFit = self.veilingFac[positions]

    def set_fitVeiling(self, fitVeiling):
        self.fitVeiling = fitVeiling
    def set_dist(self, dist, ddist=0):
        self.dist = dist
        if ddist>0:
            self.ddist = ddist
    def set_Mk(self, Mk, dMk=0):
        self.Mk = Mk
        if dMk>0:
            self.dMk = dMk
    def set_Mg(self, Mg, dMg=0):
        self.Mg = Mg
        if dMg>0:
            self.dMg = dMg
    def set_Mj(self, Mj, dMj=0):
        self.Mj = Mj
        if dMj>0:
            self.dMj = dMj
    def set_mk(self, mk, dmk=0):
        self.mk = mk
        if dmk>0:
            self.dmk = dmk
    def set_mg(self, mg, dmg=0):
        self.mg = mg
        if dmg>0:
            self.dmg = dmg
    def set_mj(self, mj, dmj=0):
        self.mj = mj
        if dmj>0:
            self.dmj = dmj
    def set_rL(self, rL, drL=0):
        self.rL = rL
        if drL>0:
            self.drL = drL
    def set_plx(self, plx, dplx=0):
        self.plx = plx
        if dplx>0:
            self.dplx = dplx
    def set_logCoeffs(self, logCoeffs):
        self.logCoeffs = logCoeffs
    def set_fitDeriv(self, fitDeriv):
        '''Sets the self.figDeriv variable. 
        If self.fitDeriv is True, then we fit the derivative of the spectra, 
        not the sepctra directly'''
        self.fitDeriv = fitDeriv
    def set_minLineDepthFit(self, minLineDepthFit):
        self.minLineDepthFit = minLineDepthFit
    def set_smoothSpectraVel(self, smoothSpectraVel):
        if smoothSpectraVel>0.0:
            self.smoothSpectra = True
        else:
            self.smoothSpectra = False
        self.smoothSpectraVel = smoothSpectraVel
    def set_errType(self, errType):
        '''errType; defines how the error is computed.
        options:
        - errType='propag'  : error propagated from the spectra
        - errType='std'     : error computed as the STD of the spectra used to bulid the template
        - errType='sqrt'    : error computed as 1/sqrt(normalized flux) '''
        self.errType = errType
    def set_dynesty(self, dynesty):
        self.dynesty = dynesty
    def set_instrument(self, instrument):
        self.instrument = instrument

    ## We want a constructor capable of setting the attributes of the
    ## object from a results file.
    def set_from_file(self, filename):
        attributes = read_res(filename)
        for key in attributes.keys():
            if key=='coeffs':
                self.update_fillFactors(attributes[key])
            # elif key=='teffs':
            #     self.update_teffs(attributes[key])
            # elif key=='loggs':
            #     self.update_loggs(attributes[key])
            # elif key=='mhs':
            #     self.update_mhs(attributes[key])
            # elif key=='alphas':
            #     self.update_alphas(attributes[key])
            elif key=='teff':
                self.set_teff(attributes[key])
            elif key=='logg':
                self.set_logg(attributes[key])
            elif key=='mh':
                self.set_mh(attributes[key])
            elif key=='alpha':
                self.set_alpha(attributes[key])
            elif key=='vb':
                self.set_vb(attributes[key])
            elif key=='guessrv':
                self.set_guessRV(attributes[key])
            elif key=='rv':
                self.set_rv(attributes[key])
            elif key=='vsini':
                self.set_vsini(attributes[key])
            elif key=='vmac':
                self.set_vmac(attributes[key])
            elif key=='lum':
                self.set_rL(attributes[key])
            elif key=='veilingFac':
                self.set_veilingFac(attributes[key])

    #####################################
    #### ---- LOAD OBSERVATIONS ---- ####
    #####################################
    def load_obs(self, filename):
        '''Function to load a template observation (version built by P. I. 
        Cristofari) from fits file.
        Input:
        - filename      :   [string] absolute path of the file to load.'''

        ## Try to resolve the filename
        if not os.path.isfile(filename):
            filename = filename.replace('gj', 'gl')
            self.star = self.star.replace('gj', 'gl')
        if not os.path.isfile(filename):
            filename = filename.replace('gl', 'gj')
            self.star = self.star.replace('gl', 'gj')
        ## Open fits file
        hdu = fits.open(filename)
        wvl = hdu[1].data; template = hdu[2].data; 
        ## Resolving the type of file. Originally, I had big fits with more cards than needed.
        ## For distribution I try to simplify things and set the input to 3 cards (wvl, flx, err).
        ## I need to check whether I am in the old format or not.
        ## Check if I have exactly the old cards in that order:
        if (len(hdu)>4):
            if  ((hdu[1].name=='WVL') and (hdu[2].name=='TEMPLATE') and (hdu[3].name=='ERR')) \
                and (hdu[4].name=='ERR_PROPAG') and (hdu[5].name=='CONTINUUM'):
                ## This is the old file format
                if self.errType=="propag":
                    template_err = hdu[4].data
                elif self.errType=='std':
                    template_err = hdu[3].data
                elif self.errType=='sqrt':
                    # template_err = 1/np.sqrt(template)
                    snrc = 2000 ## SNR in the continuum (assumed high)
                    template_err = np.sqrt(template*snrc**2+100) / snrc**2
            else:
                template_err = 0.01*template
        else:
            ## New simplified file format:
            template_err = hdu[3].data
        hdu.close()
        # dop = tls.doppler(110) ## RV shift
        if self.guessRV:
            radvel = guess_vrad(wvl, template)
        else:
            radvel = self.guessed_rv
        self.guessed_rv = radvel
        dop = tls.doppler(radvel) ## RV shift
        # med_wvl = tls.convert_lambda_in_air(wvl*10) * dop
        med_wvl = wvl*10 * dop
        med_spectrum = template
        med_err = template_err
        return med_wvl, med_spectrum, med_err, radvel

    ##################################
    #### ---- EXCLUDE PIXELS ---- ####
    ##################################
    def exclude_pixels(self):
        '''This function exclude pixels that obviously contain systematics
        Issue with this is we would need a first guess on the magnetic field.'''
        ## Consider the initial parameters:
        _Tl = self._T - np.diff(self.teffs)[0]//2
        _Th = self._T + np.diff(self.teffs)[0]//2
        _Ll = self._L - np.diff(self.loggs)[0]//2
        _Lh = self._L + np.diff(self.loggs)[0]//2
        _Ml = self._M - np.diff(self.mhs)[0]//2
        _Mh = self._M + np.diff(self.mhs)[0]//2
        _Al = self._A - np.diff(self.alphas)[0]//2
        _Ah = self._A + np.diff(self.alphas)[0]//2
        ## Consider the magnetic field strength that we have
        # fields = self.bs

        ## Here we compute a small grid of spectra around the expected parameters
        theshape = self.grid_n.shape
        mybgrid = np.empty((theshape[1], theshape[2], theshape[3], theshape[4], theshape[5], self.obs_flux.shape[-1]))
        for iT, _T in enumerate([_Tl, _Th]):
            for iL, _L in enumerate([_Ll, _Lh]):
                for iM, _M in enumerate([_Ml, _Mh]):
                    for iA, _A in enumerate([_Al, _Ah]):
                        # from IPython import embed
                        # embed()
                        if len(self.bs)<2:
                            _coeffs = np.array([1])
                        _fit = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, self.nan_mask, self.nwvls, self.grid_n, 
                                _coeffs, _T, _L, _M, _A,
                                self.teffs, self.loggs, self.mhs, self.alphas, self.vb,
                                self.rv, self.vsini, self.vmac, self.veilingFac, self._T2, self.fillTeffs)
                        mybgrid[iT, iL, iM, iA] = _fit
        
        mask_exclude = np.empty(self.obs_flux.shape)
        for o in range(mybgrid.shape[-2]):
            _mymask = filter_lines(self.obs_flux[o], mybgrid[:,:,:,:,o,:])
            mask_exclude[o] = _mymask

        # from IPython import embed
        # embed()
        # mask_exclude_bit = np.where(mask_exclude)
        # self.IDXTOFIT = (self.IDXTOFIT & mask_exclude_bit)
        self.IDXTOFIT = np.where((~np.isnan(self.obs_flux_tofit)) & (mask_exclude==1))


    def replace_observation_from_file(self, filename=None):
        '''Function replacing the flux by the best model obtained from previous run.
        This is a test for now.'''
        if filename is None:
            filename = self.opath+'fit-data.fits' ## That is the file that we save the fits to

        hdu = fits.open(filename)
        wvl = np.array(hdu['WVL'].data, dtype=float)
        flx = np.array(hdu['FLUX'].data, dtype=float)
        err = np.array(hdu['ERROR'].data, dtype=float)
        flx_fit = np.array(hdu['FLUXFIT'].data, dtype=float)
        fit = np.array(hdu['FIT'].data, dtype=float)
        idxtofit = tuple(hdu['IDXTOFIT'].data)
        hdu.close()

        ## Enlarge the error bars:
        _chi2 = np.sum((flx_fit[idxtofit]-fit[idxtofit])**2/err[idxtofit]**2)
        _nbpoints = len(flx_fit[idxtofit])
        factor = np.sqrt(_chi2/_nbpoints)

        err = err*factor

        flx_fit[~np.isnan(flx_fit)] = 1

        self.obs_flux = fit
        self.obs_flux_tofit = flx_fit*fit
        self.obs_err = err
        self.normFactor = 1
        self.renorm = False

        # std = np.std(flx_fit[idxtofit] - fit[idxtofit])
        # med = np.median(flx_fit[idxtofit] - fit[idxtofit])
        # idxout = (res > (med+2*std)) | (res < (med-2*std))
        # newerrors[idxtofit][idxout] = np.nan

        return 0


    def adjust_errors(self):
        '''This method is meant to adjust the weight of the error bars
        based on the previous optimal fit. Version 0.0'''
        ## Check that the required fits file is found
        thefile = self.opath+'fit-data.fits' ## That is the file that we save the fits to
        if not os.path.isfile(thefile):
            print('---- adjust_errors ----')
            print('File fit-data.fits not found.')
            print('Continuing with no error adjustment')
            print('-----------------------')
        else:
            self.obs_err[self.IDXTOFIT] = adjust_errors_from_file(thefile)
            self.errorsAdj = True
            self.normFactor = 1 ## Set the normalization factor to 1
            self.renorm = False ## Do not renormalize the error bars because already done.
    ######################################
    #### ---- CREATE THE REGIONS ---- ####
    ######################################
    def create_regions(self, region_file, med_wvl, med_spectrum, med_err, 
                       radvel):
        '''Function creates the regions for the observations.
        Input:
        - region_file       :   Input file to create regions. Format must be
                                compatible with line_tools.read_lines().'''
        # region_file = paths.irap_tools_data_path \
        #             + 'line_lists/newlist_01042022_noCa.txt'
        bounds, orders = line_tools.read_lines(region_file)
        bounds = bounds*10 ## Conversion in angstroms

        ## make a check
        differences = np.diff(bounds, axis=-1)
        differences = differences.T[0].T
        idx = np.where(differences<0)[0]
        if len(idx)>0:
            print('Error creating regions.')
            print('Limits upper bound lower than lower bound on line(s) # {}'.format(np.array(idx)+1))
            exit()

        ## I add implementation for broadening the regions that were given in the file
        ## using the given broadening of the spectral line (vsini and vmac)
        ## The goal is to increase the regions for stars with lager broadening
        new_bounds = []
        broad_value = np.sqrt(self.vsini**2+self.vmac**2) ## The additional overal broadening
        for bound in bounds:
            # width_wave = bound[1]-bound[0]
            # middle_wave = bound[0] + width_wave/2
            # width_vel = width_wave / middle_wave * 3*1e5 ## km/s
            ## Assuming this width was obtained for a ~5 km/s instrumental width
            ini_vel = 5 ## Assuming this width was obtained for a ~5 km/s instrumental width
            percent_increase = broad_value / ini_vel ## How much broader the region should be
            _new_bound = np.copy(bound)
            if percent_increase>1: ## For now I only enlarge the regions, never shriken them
                ini_width = bound[1]-bound[0]
                new_width = ini_width*percent_increase ## The new width of the region
                pad = (new_width - ini_width) / 2
                _new_bound[1] = bound[1]+pad
                _new_bound[0] = bound[0]-pad
            new_bounds.append(_new_bound)

        # for bound in bounds:
        #    wvllen = bound[-1]-bound[0]
        #    edge = 0.10 * wvllen
        #    bound[0] = bound[0]-edge
        #    bound[-1] = bound[-1]+edge
        
        ## This is a trick to adapt the size of the windows in the case we would have a spectrum with a higher sampling than SPIRou
        diffarr = np.diff(med_wvl[-1])
        refpoint = diffarr[-1] ## This is the reference we typically use to adjust the window length
        windowlength = round(400 * 0.165 / refpoint) ## We found that 400 windows work well for SPIRou data (sampled at about 0.165 nm at the end of the domain)

        ## If we have a large shift we should review the orders selection to 
        ## identify the best region.
        if self.instrument=='spirou':
            # blaze_file = paths.support_data + 'blaze_data/blaze_half_flux.txt'
            blaze_file = files("asap.support_data.blaze_data").joinpath("blaze_half_flux.txt")
        elif self.instrument=='winered':
            # blaze_file = paths.support_data + 'blaze_data/blaze_half_flux_{}.txt'.format(self.instrument)
            blaze_file = files("asap.support_data.blaze_data").joinpath(f"blaze_half_flux_{self.instrument}.txt")
        elif self.instrument=='igrins':
            # blaze_file = paths.support_data + 'blaze_data/blaze_{}.txt'.format(self.instrument)
            blaze_file = files("asap.support_data.blaze_data").joinpath(f"blaze_half_flux_{self.instrument}.txt")
        else:
            raise Exception('You have not defined an instrument.')
        # lims = np.loadtxt(blaze_file,  skiprows=1)
        lims = []
        f = open(blaze_file, 'r')
        for il,line in enumerate(f.readlines()):
            if line.strip()=="": continue ## empty line
            if line.strip()[0]=='#': continue ## comment
            ## Backward compatibility (first line was a comment)
            if il==0:
                try:
                    val = float(line.strip())
                except:
                    pass
            ##
            val = float(line.strip())
            lims.append(val)
        lims = np.array(lims)
        lims = lims * 10 # Angstrom
        dopplerFactor = tls.doppler(radvel)

        clims = lims * dopplerFactor
        clims = np.concatenate([[0], clims], axis=-1)
        norders = np.zeros(len(bounds), dtype=int)
        j = 0
        for i, bound in enumerate(bounds):
            pos = (bound[-1] > clims)
            norders[i] = np.arange(len(clims))[(pos)][-1]

        orders = norders

        f = open(region_file)
        elwvls = []
        els = []
        for line in f.readlines():
            if line.strip()[0]=="#": continue
            elwvls.append(float(line.split()[0]))
            els.append(line.split()[3])
        f.close()
        regions = bounds.copy()
        ## make a check
        differences = np.diff(regions, axis=-1)
        differences = differences.T[0].T
        idx = np.where(differences<0)[0]
        if len(idx)>0:
            print('Error creating regions.')
            print('Regions upper bound lower than lower bound on line(s) # {}'.format(np.array(idx)+1))
            exit()

        ## Create regions for observed data and uncertainties    
        obs_wvl, obs_flux, masks= line_tools.make_regions_2d_orders(
            med_wvl, med_spectrum, bounds, bounds, orders, 
            windowlength
            )
        obs_wvl, obs_err, _= line_tools.make_regions_2d_orders(
            med_wvl, med_err, bounds, bounds, orders, windowlength
            )
        nan_mask = masks.copy()
        nan_mask[nan_mask==0] = np.nan
        nan_mask[~np.isnan(nan_mask)] = 1

        ## We grab the new bounds of the regions that we selected.
        regions = np.array([[obs_wvl[r][0], obs_wvl[r][-1]] 
                            for r in range(len(obs_wvl))])
        self.regions = regions
        self.get_grid_dims() ## Get the grid dimensions with these regions

        self.bounds = bounds

        ## Caution ! The following block of code was just to test the program on some stars that 
        ## still contained some NaN in the selected regions. I was trying to interpolate the spectra.
        ## It's uggly, don't do it.
        # for r in range(len(obs_flux)):
        #     idx = np.where(~np.isnan(obs_flux[r]))
        #     t = np.interp(obs_wvl[r], obs_wvl[r][idx], obs_flux[r][idx])
        #     idx = np.where(~np.isnan(obs_err[r]))
        #     t2 = np.interp(obs_wvl[r], obs_wvl[r][idx], obs_err[r][idx])
        #     obs_flux[r] = t
        #     obs_err[r] = t2
        # # from IPython import embed
        # # embed()

        ## If we add the option to smooth the spectrum, it will go here:
        if self.smoothSpectra:
            for r in range(len(obs_flux)):
                obs_flux[r] = effects.broaden_spectrum_2(obs_wvl[r], obs_flux[r], 
                                                    vinstru=self.smoothSpectraVel, 
                                                    vsini=0, epsilon=0.6, 
                                                    vmac=0, vmac_mode='rt')
        ## Prepage data needed for the fitting process
        self.obs_flux_tofit = np.copy(obs_flux)
        self.obs_flux_tofit = self.obs_flux_tofit * nan_mask
        ## adapt the nan_mask so that we ignore the points of the continuum:
        if self.minLineDepthFit<1.0:
            for r in range(len(self.obs_flux_tofit)):
                ## Here I roughly estimate the continuum
                ## TODO: This is a very rough estimation of the continuum. We can fit it and do better !
                c = np.nanpercentile(obs_flux[r], 95) ## that's my continuum
                ## Then I ask how deep the deepest line is
                mc = np.nanpercentile(self.obs_flux_tofit[r], 1) ## that's my line depth
                depth = c - mc
                ## I remove everything that is above self.minLineDepthFit percent above the minimum I reject
                self.obs_flux_tofit[self.obs_flux_tofit>(c-(1-self.minLineDepthFit)*depth)] = np.nan
        ## If we use the derivative of the spectrum instead, this is where we compute it.
        if self.fitDeriv:
            _obs_flux_tofit = np.copy(self.obs_flux_tofit)
            for r in range(len(self.obs_flux_tofit)):
                _obs_flux_tofit[r][:-1] = np.diff(self.obs_flux_tofit[r])
            self.IDXTOFIT = np.where(~np.isnan(_obs_flux_tofit))
        else:
            self.IDXTOFIT = np.where(~np.isnan(self.obs_flux_tofit))
        
        ## Store in global attributes:
        self.obs_wvl = obs_wvl; self.obs_flux = obs_flux; 
        self.obs_err = obs_err
        self.nan_mask = nan_mask        

        return obs_wvl, obs_flux, obs_err, nan_mask, regions

    ###################################
    #### ---- LOAD MODEL GRID ---- ####
    ###################################
    def load_grid(self, pathtogrid, regions):
        '''Load a grid of models for all mag field strengths'''
        print('Loading grid')
        wgrid = np.zeros((self.d1, self.d2, self.d3, 
                          self.d4, self.d5, self.d6)).tolist()
        wvls = np.zeros((self.d6)).tolist()
        grid = np.zeros([self.d1, self.d2, self.d3, 
                          self.d4, self.d5, self.d6]).tolist()
        # teffs = np.arange(3300, 3500, 100); loggs = np.arange(5.0, 6., .5)
        # mhs = np.arange(-0.5, 0.5, .5); alphas = np.arange(0.00, 0.50, .25); 
        # bs = np.arange(0, 10, 2)
        # control_wgrid = np.zeros([2, 2, 2, 2, 9, 1, 632001])
        # control_grid = np.zeros([2, 2, 2, 2, 9, 1, 632001])
        ntot = self.d1*self.d2*self.d3*self.d4*self.d5
        n = 0
        for it, teff in enumerate(self.teffs):
            for il, logg in enumerate(self.loggs):
                for im, mh in enumerate(self.mhs):
                    for ia, alpha in enumerate(self.alphas):
                        for ib, B in enumerate(self.bs):
                            try:
                                ## Try to read the hdf5 file:
                                ## HARCODE phase and rot and beta are hardcoded
                                ## TODO: REMOVE HARDCODE
                                ztfile_struct = self.file_struc
                                phase = 0.0; rot=90.; beta=0.
                                ztfile = ztfile_struct.format(teff, logg, mh, alpha, B*1000, phase, rot, beta)
                                filename = pathtogrid + ztfile
                                with h5py.File(filename, 'r') as h5f:
                                    if 'wavelink' in h5f.keys():
                                        w = h5f['wavelink']['wave'][()]
                                    else:
                                        w = h5f['wave'][()]
                                    w = tls.convert_lambda_in_vacuum(w)
                                    s = h5f['norm_flux'][()]
                                if len(w)!=len(s):
                                    print('You are in SpectralAnalysis.load_grid.')
                                    print('Reading failed because len(w)!=len(s)')
                                    from IPython import embed
                                    embed()
                                    raise Exception('Issue with file {} -- dimension mismatch wave and norm_flux'.format(ztfile))
                                # grid_handle = h5py.File(pathtogrid + "all-spectra.hdf5", 'r')
                                # self.grid_wvl = grid_handle['wave'][()]
                                # self.grid_handle = grid_handle
                            except:
                                print('could not find {}'.format(ztfile))
                                ztfile = '{}g{:0.1f}z{:+0.2f}a{:+0.2f}b{:0.1f}.noconvol'.format(teff, logg, mh, alpha, B)
                                # if not os.path.isfile(ztfile):
                                #     ztfile += ".gz"
                                try:
                                    hdu = fits.open(pathtogrid + ztfile, memmap=False)  
                                    w = np.copy(hdu['WVL'].data)
                                    w = tls.convert_lambda_in_vacuum(w)
                                    s = np.copy(hdu['NFLUX'].data)
                                    hdu.close()
                                except:
                                    self.vinstru = np.sqrt(4.3**2 - 4.**2)
                                    ztfile = '{}g{:0.1f}z{:0.1f}a{:0.2f}.int_9200-25000.convol.fits'.format(teff, logg, mh, alpha)
                                    hdu = fits.open(pathtogrid + ztfile, memmap=False)  
                                    w = np.copy(hdu['WVL'].data)
                                    w = tls.convert_lambda_in_vacuum(w)
                                    s = np.copy(hdu['NFLUX'].data)
                                    hdu.close()
                                # w, s, _ = np.loadtxt(pathtogrid + ztfile, unpack=True)
                            #
                            _wvl, _spectrum = line_tools.make_contrained_regions(w, 
                                                            s, 
                                                            regions)
                            
                            interupt = False
                            for _r in range(len(_wvl)):
                                if len(_wvl)==0:
                                    interupt = True
                                    print('Problem: line list requested in the region {} but no model found there.'.format(regions[_r]))
                            if interupt:
                                exit(1)
                            # We add an option to resample the grid on a 
                            # wavelength solution constant in speed.
                            if self.resampleVel:
                                for ir in range(len(regions)):
                                    _wvl[ir], _spectrum[ir] = \
                                    tls.resample_vel_interp(_wvl[ir], _spectrum[ir], kind='cubic')
                            #
                            for ir in range(len(regions)):
                                wgrid[it][il][im][ia][ib][ir] = _wvl[ir]
                                grid[it][il][im][ia][ib][ir] = _spectrum[ir]
                                wvls[ir] = _wvl[ir]
                            n += 1
                            stat = n / ntot * 100
                            print("Reading... {:0.2f} %".format(stat), 
                                                            end='\r')
        ## Get the length of the arrays need to store the regions
        maxlen = 0
        wvls = wgrid[0][0][0][0][0]
        for r in range(len(wvls)):
            _len = len(wvls[r])
            if _len > maxlen:
                maxlen = _len
        self.d7 = maxlen
        ## Prepare numpy array of correct size
        nwvls = np.zeros((self.d6, self.d7))
        nspectra = np.zeros((self.d1, self.d2, self.d3, self.d4, 
                             self.d5, self.d6, self.d7))
        for it, teff in enumerate(self.teffs):
            for il, logg in enumerate(self.loggs):
                for im, mh in enumerate(self.mhs):
                    for ia, alpha in enumerate(self.alphas):
                        for ib, B in enumerate(self.bs):
                            for r in range(len(regions)):
                                _maxlen = len(grid[it][il][im][ia][ib][r])
                                _dwvls = wvls[r][-1] - wvls[r][-2]
                                _wvls = np.arange(maxlen) * _dwvls + wvls[r][0]
                                try:
                                    _wvls[:_maxlen] = wvls[r]
                                    nwvls[r] = _wvls
                                except:
                                    raise Exception("Pb with {} {} {} {} {}".format(teff, logg, mh, alpha, B))
                                nspectra[it, il, im, ia, ib, r][:_maxlen] = grid[it][il][im][ia][ib][r]

        grid = np.array(nspectra, dtype=float)
        self.grid_n = np.moveaxis(grid, -3, 0) ## Place magnetic field as first index
        self.nwvls = nwvls
        print('Done grid')
        return nwvls, self.grid_n, self.teffs, self.loggs, self.mhs, self.alphas

    def pre_norm_grid(self):
        grid_n = pre_norm_grid_numba(self.d1, self.d2, self.d3,
                                     self.d4, self.d5, self.d6, 
                                     self.d7,
                                     self.teffs, self.loggs, 
                                     self.mhs, self.alphas, 
                                     self.bs,
                                     self.nwvls, self.grid_n)
        self.grid_n = grid_n
    
    # @jit(nopython=params.JIT)
    def pre_norm_grid_slow(self):
        '''Run a moving median through the observations'''
        rollmed_window = 500
        rollmed_btd = 0
        rollmed_p = 90
        ncont = np.ones((self.d6, self.d7)) ## TODO: change this ugly hardcode

        ntot = self.d1*self.d2*self.d3*self.d4*self.d5
        n = 0
        for it, teff in enumerate(self.teffs):
            for il, logg in enumerate(self.loggs):
                for im, mh in enumerate(self.mhs):
                    for ia, alpha in enumerate(self.alphas):
                        for ib, B in enumerate(self.bs):
                            for r in range(self.d6):
                            # for r in range(self.d6):
                                _wvl = self.nwvls[r]
                                _flux = self.grid_n[ib, it, il, im, ia, r]
                                _diff = np.diff(_wvl)
                                wvlstep = _diff[len(_diff)//2]
                                dspeed = wvlstep / _wvl[len(_diff)//2] * 3*1e5
                                _nbins = rollmed_window/dspeed
                                _window = int(_nbins)
                                _, _ncont = norm_tools.moving_median(_flux, _window,
                                                                    btd=rollmed_btd, p=rollmed_p)
                                ncont[r] = _ncont
                                self.grid_n[ib, it, il, im, ia, r] = self.grid_n[ib, it, il, im, ia, r] / np.array(_ncont)
                            ncont = np.array(ncont)
                            n += 1
                            # print('Flattening... {:.2f} %'.format(n/ntot*100), end='\r')
                            print('Flattening... {:.2f} %'.format(n/ntot*100))
    
    def pre_norm_obs(self):
        '''Run a moving median through the models'''
        rollmed_window = 500
        rollmed_btd = 0
        rollmed_p = 90
        ncont = np.ones((self.d6, 400)) ## TODO: change this ugly hardcode
        for r in range(self.d6):
            _wvl = self.obs_wvl[r]
            _flux = self.obs_flux[r]
            _diff = np.diff(_wvl)
            wvlstep = _diff[len(_diff)//2]
            dspeed = wvlstep / _wvl[len(_diff)//2] * 3*1e5
            _nbins = rollmed_window/dspeed
            _window = int(_nbins)
            _, _ncont = norm_tools.moving_median_vel(_wvl, _flux, rollmed_window,
                                                 btd=rollmed_btd, p=rollmed_p)
            ncont[r] = _ncont
            print('Flattening... {:.2f} %'.format(r/self.d6*100), end='\r')
        ncont = np.array(ncont)
        self.obs_flux = self.obs_flux / ncont
        self.obs_err = self.obs_err / ncont

    def mask_small_lines(self, ths=0.05):
        '''This function sets to NaN the regions in which we find that the lines are
        not deep enough.'''

        fullnanregions = []
        for r in range(len(self.obs_wvl)):
            print('ok')
            _c, _pss, X, Xerr, X2, X2err = norm_tools.adjust_continuum5(wvl=self.obs_wvl[r],
                                                obs_flux=self.obs_flux[r],
                                                model_flux=np.ones(len(self.obs_flux[r])),
                                                window_size=100,
                                                p=90,
                                                degree=1, m=0.05)        

            _obscorr = self.obs_flux[r] / _c
            _nanmask = self.nan_mask[r]
            minval = np.nanmin(_obscorr*_nanmask)
            if (1 - minval) < ths: ## This line is too small !
                self.nan_mask[r] *= np.nan
            
            ## Is the whole region full of NaNs now?
            nanobs = self.obs_flux[r] * self.nan_mask[r]
            if len(nanobs[~np.isnan(nanobs)])<5:
                fullnanregions.append(r)

            # ## How many lines in the region?
            # _wvl = np.copy(self.obs_wvl[r])
            # _wvlnonan = _wvl[~np.isnan(self.nan_mask[r])]
            # _diff = np.diff(_wvlnonan)
            # idx = np.where(_diff>2)[0]
            # if len(idx)>1: ## We have multiple regions

            #     for subr in range(len(idx)+1):
            #         _wvlnonan
        ## Delete the regions that are all full of NaNs
        fullnanregions = fullnanregions[::-1] ## Reverse the order
        for rtodel in fullnanregions:
            print(rtodel)
            if len(self.obs_wvl.shape)>1:
                self.bounds = np.delete(self.bounds, rtodel, axis=0)
                self.regions = np.delete(self.regions, rtodel, axis=0)
                self.obs_wvl = np.delete(self.obs_wvl, rtodel, axis=0)
                self.obs_flux = np.delete(self.obs_flux, rtodel, axis=0)
                self.obs_err = np.delete(self.obs_err, rtodel, axis=0)
                self.nan_mask = np.delete(self.nan_mask, rtodel, axis=0)
            elif len(self.obs_wvl.shape)==1:
                if rtodel==0:
                    self.bounds = np.array([])
                    self.regions = np.array([])
                    self.obs_wvl = np.array([])
                    self.obs_flux = np.array([])
                    self.obs_err = np.array([])
                    self.nan_mask = np.array([])


        if len(self.obs_wvl)==0:
            raise Exception('mask_small_lines : '\
                            + 'All regions removed!')
        elif len(self.obs_wvl)==1:
            self.bounds = np.array([self.bounds])
            self.regions = np.array([self.regions])
            self.obs_wvl = np.array([self.obs_wvl])
            self.obs_flux = np.array([self.obs_flux])
            self.obs_err = np.array([self.obs_err])
            self.nan_mask = np.array([self.nan_mask])
        
        # from IPython import embed
        # embed()

        ## Here it is very important that we udpate the bins on which we perform the fit
        ## This was intially done in the create regions function.
        self.obs_flux_tofit = np.copy(self.obs_flux)
        self.obs_flux_tofit = self.obs_flux_tofit * self.nan_mask
        self.IDXTOFIT = np.where(~np.isnan(self.obs_flux_tofit))
        self.get_grid_dims() ## Get the grid dimensions with these regions

        return self.regions

    ################################
    #### ---- GEN SPECTRUM ---- ####
    ################################
    def gen_spec(self, obs_wvl, obs_flux, obs_err, nan_mask, nwvls, grid_n, 
             coeffs, T, L, M, A,
             teffs, loggs, mhs, alphas, vb=None, rv=None,  
             vsini=None, vmac=None, veilingFacToFit=None,
               T2=None, fillTeffs=np.array([1, 0])):
        '''Genertare interpolated, broadened and adjusted magnetic model.'''

        if vb is None: vb = self.vb
        if rv is None: rv = self.rv
        if vsini is None: vsini = self.vsini
        if vmac is None: vmac = self.vmac
        if veilingFacToFit is None: veilingFacToFit = self.veilingFacToFit
        ## Determine the radial velocity shift
        dopshift = tls.doppler(rv)
        _Bspec = np.zeros((self.d5, self.d6, self.d7))

        ## Ok, interesting test, but to be faster we would need to work with Numba dictionaries.
        ## For now this is a bit of a pain and it does not work with arrays as key types.
        ## So we would have to find a fast way to encode the parameters into a unique key.
        ## String conversion may be slow.
        ##
        ## An idea for the encoding could be:
        # T = 3400; L = 3.50; M = 0.25; A = 0.25; B = 5000
        # O = (int(T*100000000000000000) ## 13 zeros
        #     + int(L * 1000 * 10000000000000) ## Multiply by 1000, then 9 place holders
        #     + int(M * 1000 * 1000000000) ## Multiply by 1000, then 5 place holders
        #     + int(A * 1000 * 100000)
        #     + int(B)
        #     )
        # print(O)
        ## This make a unique int encoding a specific set of numbers.
        ##
        # ## This is a test for speed.
        # from IPython import embed
        # embed()

        # from numba import types
        # from numba.typed import Dict

        # # First create a dictionary using Dict.empty()
        # # Specify the data types for both key and value pairs

        # # Dict with key as strings and values of type float array
        # dict_param1 = Dict.empty(
        #     key_type=types.unicode_type,
        #     value_type=types.float64[:],
        # )
    
        # dict_param2 = Dict.empty(
        #     key_type=types.int64,
        #     value_type=types.float64[:],
        # )

        # from irap_tools.spectral_analysis_pack import interp_grid_4d_dict
        # ## Building a dictionary of data
        # datadic = {}
        # for it, teff in enumerate(teffs):
        #     for il, logg in enumerate(loggs):
        #         for im, mh in enumerate(mhs):
        #             for ia, alpha in enumerate(alphas):
        #                 for bf in range(len(grid_n)):
        #                     datadic[(teff,logg,mh, alpha, bf)] = grid_n[bf, it, il, im, ia]

        # ## That is the new version
        # _Bspec2 = np.zeros((self.d5, self.d6, self.d7))
        # itime = time.time()
        # for b in range(100):
        #     for i in range(self.d5):
        #         _, s = interp_grid_4d_dict(
        #                                             T, L, M, A, bf,
        #                                             teffs, loggs, mhs, alphas,
        #                                             datadic,
        #                                             function=self.interpFunc)
        #     _Bspec2[i] = s
        # etime = time.time()
        # print(etime-itime)


        # _Bspec = np.zeros((self.d5, self.d6, self.d7))
        # itime = time.time()
        # for b in range(100):
        #     ## That is what we had before
        #     for i in range(self.d5):
        #         _, s = wrap_function_fine_linear_4d(
        #                                             T, L, M, A,
        #                                             teffs, loggs, mhs, alphas,
        #                                             grid_n[i], 
        #                                             function=self.interpFunc)
        #     _Bspec[i] = s
        # etime = time.time()
        # print(etime-itime)



        # exit()
        # ######

        for i in range(self.d5):
            try:
                _, s = wrap_function_fine_linear_4d(
                                                    T, L, M, A,
                                                    teffs, loggs, mhs, alphas,
                                                    grid_n[i], 
                                                    function=self.interpFunc)
                # s = grid_n[i][0,0,0,0]
            except:
                raise Exception("Interpolation failed for parameters: {} {} {} {} {}".format(T, L, M , A, self.bs[i]))
            _Bspec[i] = s

            #############
            # spectra = np.empty(s.shape)
            # for jj in range(len(s)):
            #     from irap_tools import effects
            #     _spectra = effects.broaden_spectrum_2(nwvls[jj], s[jj], 
            #                                 vinstru=0, 
            #                                 vsini=vsini, epsilon=0.6, 
            #                                 vmac=vmac, vmac_mode=self.vmacMode)
            #     spectra[jj] = _spectra
            # _Bspec[i] = spectra
            #############


        if self.logCoeffs:
            tosum = [np.exp(coeffs[i]) * _Bspec[i] for i in range(len(coeffs))]
        else:
            tosum = [coeffs[i] * _Bspec[i] for i in range(len(coeffs))]
        mergedspec = np.sum(tosum, axis=0) ## non-broad non-adj magnetic model
        #
        ## IF we have a second temperature
        if T2 is not None:
            _Bspec = np.zeros((self.d5, self.d6, self.d7))
            for i in range(self.d5):
                _, s = wrap_function_fine_linear_4d(
                                                    T2, L, M, A,
                                                    teffs, loggs, mhs, alphas,
                                                    grid_n[i], 
                                                    function=self.interpFunc)
                _Bspec[i] = s
            if self.logCoeffs:
                tosum = [np.exp(coeffs[i]) * _Bspec[i] for i in range(len(coeffs))]
            else:
                tosum = [coeffs[i] * _Bspec[i] for i in range(len(coeffs))]
            mergedspec2 = np.sum(tosum, axis=0) ## non-broad non-adj magnetic model
            mergedspec = fillTeffs[0]*mergedspec + fillTeffs[1]*mergedspec2
        ## Apply the doppler shift to the wavelength solution of the
        ## SYNTHETIC spectrum. IF it is applied to the observation spectrum
        ## This means that we will not have the synhtetic spectrum on the same
        ## wavelength grid, and that will be a major problem.
        nwvls_shift = nwvls * dopshift
        ## We need to make a check here: if the radial velocity is so large that
        ## the spectrum is out the window we are going to have a problem.
        ## I is equivalent to adding a prior, but we do not want to the code to fail here.

        ## The total broadening (gaussian) of the instrument is
        totvb = np.sqrt(self.vinstru**2 + vb**2 + self.smoothSpectraVel**2)

        ## Here we determine the correct veiling
        # myveiling = veiling_function(veilingFac, nwvls_shift)
        
        args = [0, nwvls_shift, mergedspec, obs_wvl, obs_flux, obs_err, 
                nan_mask, totvb, vmac, vsini, 
                0, 0, 0, '0', self.adjcont, 'line']
        ## fit is the model after broadening and adjustment
        try:
            _, _, _, fit, _, _, [cs, cs2], _, _ = broaden_spectra(args, 
                                                            macProf=self.vmacMode)
        except:
            from IPython import embed
            embed()

        # ## Here we determine the correct veiling
        ## Here I forbid the veiling from the other bands to compensate for the veiling
        ## in the YJHK bands.
        ## Now it gets tricky. I will put the values that we actually fit in place in the array
        ## For the veiling, we therefore must know: which bands we pass to the function
        ## What are the values provided for all bands
        ## For which band we fit the values
        veilingFac = self.veilingFac
        fitveilpos = np.zeros(self.nbFitVeil, dtype=int)
        if self.fitVeiling:
            ## Check that fitBands contain something
            if self.fitBands=="": raise Exception('fitBands empty but fitVeiling==True')
            for ib, band in enumerate(self.fitBands):
                fitveilpos[ib] = self.veilingBands.find(band)
            veilingFac[fitveilpos] = veilingFacToFit
        myveiling = veiling_function(veilingFac, obs_wvl, self.veilingBands)
        fit_v = (fit + myveiling) / (1 + myveiling)# veiled spectrum
        # fit_v = fit

        return fit_v

    ##############################################
    #### ---- FUNCTIONS TO MCMC ANALYSIS ---- ####
    ##############################################


    def return_labels(self):
        '''This function helps unpack the argument passed to lnlike and 
        lnprob
        Under development, trying to be sligthly more clever'''

        ## Automatically detect the type of analysis from the magnetic field
        ## array. This technique assumes that there ALWAYS is at least the
        ## one magnetic value of field (typically 0, but could in theory be
        ## anything).

        self.AVAILABLE_LABELS = [r'$T_{\rm eff}$ (K)', r"$\log{g}$ (dex)",
                                 r"$\rm [M/H]$ (dex)", r"$\rm [\alpha/Fe]$ (dex)",
                                 r"$v_{\rm b}\,(km\,s^{-1})$", r"$\rm RV\,(km\,s^{-1})$",
                                 r"$v\sin{i}\,(km\,s^{-1})$", r"$\zeta\,(km\,s^{-1})$",
                                 r"$r_{\rm I}$", r"$r_{\rm Y}$", r"$r_{\rm J}$", 
                                 r"$r_{\rm H}$", r"$r_{\rm K}$", r"$r_{\rm L}$"]

        labels = []
        ## Run through conditions
        nbOfFields = len(self.bs) ## This will helps us unpack par
        idxStart = 0
        if self.fitFields:
            idxStart = nbOfFields-1
            for i in range(0, idxStart):
                labels.append(r"$a_{"+f"{self.bs[1:][i]}"+r"}$")
        ## Grab the T, L, M, A
        i = idxStart
        if self.fitTeff:
            labels.append(r'$T_{\rm eff}$ (K)')
            i += 1
        if self.fitLogg:
            labels.append(r"$\log{g}$ (dex)")
            i += 1
        if self.fitMh:
            labels.append(r"$\rm [M/H]$ (dex)")
            i += 1
        if self.fitAlpha:
            labels.append(r"$\rm [\alpha/Fe]$ (dex)")
            i += 1
        ## Loop through the parameters
        if self.fitbroad:
            labels.append(r"$v_{\rm b}\,(km\,s^{-1})$")
            i += 1
        if self.fitrv:
            labels.append(r"$RV\,(km\,s^{-1})$")
            i+=1
        if self.fitrot:
            labels.append(r"$v\sin{i}\,(km\,s^{-1})$")
            i+=1
        if self.fitmac:
            labels.append(r"$\zeta\,(km\,s^{-1})$")
            i+=1        
        if self.fitVeiling:
            for band in self.fitBands:
                labels.append("$r_{\\rm "+band+"}$")
            i+=1+self.nbFitVeil
        if self.fitTeff2: ## Second temperature
            labels.append(r'$T_{\rm eff, 2}$ (K)')
            i += 1
            labels.append(r"$f_{T_{\rm eff, 2}}$")
            i += 1
        
        self.labels =labels
        return labels


    def unpackpar(self, par):
        '''This function helps unpack the argument passed to lnlike and 
        lnprob
        Under development, trying to be sligthly more clever'''

        ## Automatically detect the type of analysis from the magnetic field
        ## array. This technique assumes that there ALWAYS is at least the
        ## one magnetic value of field (typically 0, but could in theory be
        ## anything).

        ## Set default values:
        vb = self.vb; rv = self.rv; vsini = self.vsini; vmac = self.vmac
        _T = self._T; _T2 = self._T2; _L = self._L; _M = self._M; _A = self._A
        veilingFacToFit = self.veilingFacToFit; _fillTeffs = self.fillTeffs
        coeffs = self.coeffs

        ## Run through conditions
        nbOfFields = len(self.bs) ## This will helps us unpack par
        # if nbOfFields==1:
        #     ## We have no fields, so we know that par contains other stuff
        #     coeffs = np.array([1])
        #     idxStart = 0
        # elif nbOfFields>1:
        #     coeffs = np.zeros(nbOfFields)
        #     idxStart = nbOfFields-1
        #     coeffs[1:nbOfFields] = par[:idxStart]
        #     if self.logCoeffs:
        #         coeffs[0] = np.log(1 - np.sum(np.exp(coeffs[1:])))
        #     else:
        #         coeffs[0] = 1 - np.sum(coeffs[1:])
        # if not self.fitFields:
        #     coeffs = self.coeffs
        idxStart = 0
        if self.fitFields:
            idxStart = nbOfFields-1
            coeffs = np.zeros(nbOfFields)
            idxStart = nbOfFields-1
            coeffs[1:nbOfFields] = par[:idxStart]
            if self.logCoeffs:
                if np.sum(np.exp(coeffs[1:]))<1.0:
                    coeffs[0] = np.log(1 - np.sum(np.exp(coeffs[1:])))
                else:
                    coeffs[0] = 1
                # if np.isnan(coeffs[0]): ## I do not want to carry NaNs
                #     coeffs[0] = -np.inf
            else:
                coeffs[0] = 1 - np.sum(coeffs[1:])
        ## Grab the T, L, M, A
        i = idxStart
        if self.fitTeff:
            _T = par[i]
            i += 1
        if self.fitLogg:
            _L = par[i]
            i += 1
        if self.fitMh:
            _M = par[i]
            i += 1
        if self.fitAlpha:
            _A = par[i]
            i += 1
        ## Loop through the parameters
        if self.fitbroad:
            vb = par[i]
            i += 1
        if self.fitrv:
            rv = par[i]
            i+=1
        if self.fitrot:
            vsini = par[i]
            i+=1
        if self.fitmac:
            vmac = par[i]
            i+=1        
        if self.fitVeiling:
            veilingFacToFit = par[i:i+self.nbFitVeil]
            i+=1+self.nbFitVeil
        if self.fitTeff2: ## Second temperature
            _T2 = par[i]
            i += 1
            _fillTeffs2 = par[i]
            _fillTeffs = np.array([1-_fillTeffs2, _fillTeffs2])
            i += 1
        ## Special case: if we have set the autoLogg
        if self.autoLogg:
            _L = self.compute_logg(_T, _M)

        return coeffs, _T, _L, _M, _A, vb, rv, vsini, vmac, veilingFacToFit, _T2, _fillTeffs

    def compute_normFactor(self, normFactor=None):
        '''Function computing the normalization factor
        This function should now also read from a file located in the
        supporting data folder, which can be used to store the chi2 values
        from previous runs. This should allow us to refine estimates as we go.
        '''
        ## Try to read from renormFactors from file
        # factor_fname = paths.support_data + 'spectral_analysis/normFactors.txt'
        factor_fname = self.normfacfile
        stars_data = {}
        # Check that the file exists:
        if not os.path.isfile(factor_fname): pass # file does not exist
        else: 
            # Read the file and store data
            ff = open(factor_fname, 'r')
            for line in ff.readlines():
                if line[0]=="#": continue # Comment
                if line.strip()=="": continue # empty line
                star = line.split()[0]; val = float(line.split()[1])
                stars_data[star] = val
            ff.close()
        # format star name to standards
        _starname = self.star.lower()
        _starname = _starname.replace(" ", "").strip().replace('gl', 'gj')

        if (normFactor is None) | (normFactor==1): ## User did not provide
            if self.renorm: ## User wants to renorm
                if _starname in stars_data.keys(): ## We search file data
                        self.normFactor = stars_data[_starname]
                else: ## Don't have data, we guess the value
                    # _ = self.lnlike()
                    # nbPointsFitted = len(self.obs_flux_tofit[self.IDXTOFIT])
                    # minchi2 = np.sum(self._res)
                    # self.normFactor = minchi2 / nbPointsFitted
                    self.normFactor = 1.
            else: ## User does not want to renorm
                self.normFactor = 1
        else: ## User provided a specific value
            if self.renorm: ## And wants to renorm
                self.normFactor = normFactor
            else: ## But does not want to renorm
                self.normFactor = 1
        ## normFactor cannot be 0. That would lead to a bug.
        if self.normFactor < 1e-6:
            self.normFactor = 1.

    def save_normFactor(self, normFactor):
        '''This function saves the normFactor to for the current star
        - First we check if the star is in the file
        - We rewrite the file with the new value of append the data'''

        ## Try to read from renormFactors from file
        # factor_fname = paths.support_data + 'spectral_analysis/normFactors.txt' 
        factor_fname = self.normfacfile
        stars_data = {}
        # Check that the file exists:
        if not os.path.isfile(factor_fname): pass # file does not exist
        else: 
            # Read the file and store data
            ff = open(factor_fname, 'r')
            for line in ff.readlines():
                if line[0]=="#": continue # Comment
                if line.strip()=="": continue # empty line
                star = line.split()[0]; val = float(line.split()[1])
                stars_data[star] = val
            ff.close()
        # format star name to standards
        _starname = self.star.lower()
        _starname = _starname.replace(" ", "").strip().replace('gl', 'gj')
        # if _starname in stars_data.keys():
        stars_data[_starname] = normFactor
        # And now we create the new file:
        gg = open(factor_fname, 'w')
        for star in stars_data.keys():
            val = stars_data[star]
            gg.write("{} {}\n".format(star, val))
        gg.close()

    def lnlike(self, par=None):
        '''This function returns should return 
        something that looks like a chi2.
        Inputs:
        - par    : filling factors'''

        ## Set the values to default, then unpack the par variable.
        if par is None:
            _T = self._T; _T2 = self._T2; _L = self._L; _M = self._M; _A = self._A
            vb = self.vb; rv = self.rv; vsini = self.vsini; vmac = self.vmac
            coeffs = self.coeffs; veilingFacToFit = self.veilingFacToFit; _fillTeffs = self.fillTeffs
        else:
            coeffs, _T, _L, _M, _A, vb, rv, vsini, vmac, veilingFacToFit, \
               _T2, _fillTeffs = self.unpackpar(par)

        # ##################
        # ##################
        # ## It looks like this implementation would not work... Saved here as a draft
        # ##################
        # # ## Here coeffs contains the whole list of magnetic fields filling factors, including for the 0 magnetic field.
        # # ## I am not sure how to implement a prior on the sum of the filling factors with dynasty, so here is a small trick
        # # ## We want to do that only if we are running with dynesty
        # if self.dynesty:
        #     sumfillingfactors = np.sum(coeffs)
        #     # print(sumfillingfactors)
        #     if sumfillingfactors>1:
        #         return -np.inf
        #         coeffs = coeffs / sumfillingfactors ## This ensures that the sum is 1
        #         ## Note this means we will need to rescale the coefficients in the results at
        # ##################
        # ##################
        
        ## Avoid computing spectrum if par is out of bounds anyway
        priorval = self.lnprior(par)
        if not np.isfinite(priorval):
            return priorval
        _itime = time.time()
        fit = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, 
                            self.nan_mask, self.nwvls, self.grid_n, 
                            coeffs, _T, _L, _M, _A,
                            self.teffs, self.loggs, self.mhs, self.alphas, vb,
                            rv, vsini, vmac, veilingFacToFit, _T2, _fillTeffs)
        _etime = time.time()
        # print("{:0.4f}".format(_etime-_itime))
        myerr = self.obs_err[self.IDXTOFIT] * np.sqrt(self.normFactor)
        # myerr2 = myerr**2 + fit[self.IDXTOFIT]**2

        if self.fitDeriv:
            _obs_flux_tofit_deriv = np.copy(self.obs_flux_tofit)
            for r in range(len(self.obs_flux_tofit)):
                _obs_flux_tofit_deriv[r][:-1] = np.diff(self.obs_flux_tofit[r])
                fit[r][:-1] = np.diff(fit[r])
            _resup = (_obs_flux_tofit_deriv[self.IDXTOFIT] - fit[self.IDXTOFIT])**2
            _resdown = myerr**2
        elif 10<0:  ## This is for the test, we will need a new flag if we want that
            #### ------------------------------------------------------------------ ####
            #### Trying a new way do do things:
            #### What if instead of adjusting the continuum, I try to minimize the residuals
            #### by fitting a line through them?
            #### This needs to be done for each of the regions
            #### on the points that are being fitted(???), otherwise other points in the region could influence the fit?
            ##
            ## Iterate through the regions
            _resup = np.empty(np.shape(self.obs_flux_tofit[self.IDXTOFIT]))
            for r in range(len(self.obs_flux)):
                locidx = self.IDXTOFIT[0]==r 
                indices =  self.IDXTOFIT[1][locidx] ## position of points fitted for this specific region
                # _wvl = self.obs_wvl[r][indices]
                _obs = self.obs_flux_tofit[r][indices]
                _fit = fit[r][indices]
                # _err = self.obs_err[r][indices] * np.sqrt(self.normFactor)
                ## Compute the residuals for this specific region
                _loc_resup = (_obs - _fit)**2
                # _loc_resdown = _err**2
                ## Instead of fitting a line, right now, I just remove the median of the residuals.
                med = np.median(_loc_resup)
                _resup[locidx] = _loc_resup - med
            _resdown = myerr**2
            #### Maybe for later
            # ## Now fit a line through that
            # coeffs, _ = norm_tools.fit_poly(_wvl, _resup, degree=1)
            # line = norm_tools.poly1d(_wvl, coeffs) ## this should be the continuum
            # coeffs = tls.fit_line_dxdy(_wvl, _resup, _err)
            # line2 = norm_tools.poly1d(_wvl, coeffs) ## this should be the continuum
            # std = np.std(_resup)
            # ##

            # plt.figure()
            # # plt.plot(self.obs_wvl[r][indices], self.obs_flux[r][indices])
            # # plt.plot(self.obs_wvl[r][indices], fit[r][indices])
            # plt.plot(_resup)
            # plt.plot(_resupold)
            # # plt.axhline(med, color='black')
            # plt.show()
            ## 
            #### ------------------------------------------------------------------ ####
        else:
            _resup = (self.obs_flux_tofit[self.IDXTOFIT] - fit[self.IDXTOFIT])**2
            _resdown = myerr**2

        # ## ------------------------------------------------------------------------------
        # ## PIC test: Here if is like computign a chi2 (almost), but I would like to
        # ## remove the median to the residuals in each small window (line per line)
        # ## The reason is simply that I am afraid that small uncertainties on the conitnuum
        # ## could lead to big biases in the results. So I'm saying I want to look at the
        # ## STD of the residuals centered on zero, rather than the chi2.
        # ## 
        # ## Split the waevelength array
        # mywvl = self.obs_wvl[self.IDXTOFIT]
        # _resupcont = np.zeros(_resup.shape)
        # _mywvlidx = np.where(np.diff(mywvl)>1)[0]
        # if len(_mywvlidx)>0:
        #     _resupcont[:_mywvlidx[0]] = np.median(_resup[:_mywvlidx[0]])
        #     for i in range(len(_mywvlidx)-1):
        #         _resupcont[_mywvlidx[i]:_mywvlidx[i+1]] = np.median(_resup[_mywvlidx[i]:_mywvlidx[i+1]])
        #     _resupcont[_mywvlidx[-1]:] = np.median(_resup[_mywvlidx[-1]:])

        # _resup = _resup - _resupcont

        # from IPython import embed
        # embed()
        # exit(1)
        # ## ------------------------------------------------------------------------------
    
        # ## Dynamic Online Point Removal
        # ## Unfortunately leads to a non smooth surface so the results are not really good.
        # 
        # ## Here I am trying a new thing. We reject points that are above a given threshold.
        # std = np.std(_resup) ## our reference
        # med = np.median(_resup) ## our reference
        # idxout = (_resup > (med+2*std)) | (_resup < (med-2*std))

        # self.obs_err[self.IDXTOFIT][idxout]+=self.obs_err[self.IDXTOFIT][idxout]*0.1 ## Increase by 10% the points that are away
        # self.obs_err[self.IDXTOFIT][~idxout]-=self.obs_err[self.IDXTOFIT][~idxout]*0.1 ## Decrease by 10% the points that are away

        # _resdown[idxout] = _resdown[idxout]*1e5 ## Lower the weight on those points


        _res = _resup/_resdown


        ## PIC: I found somewhere that the ln(likelihood) computed from a
        ## normal distribution should be of the form 
        ## -1/2 SUM[chi^2 + ln(2pi*err^2)]. So I add this last term for
        ## testing.
        ## This is also, btw, the relation that is in the emcee examples.
        # self._lncorr = np.log(2*np.pi*myerr**2)
        # self._res = res**2/self.normFactor ## that really is the chi2
        # res = self._res + np.log(2*np.pi*myerr2)
        # res = np.log(_res) #+ np.log(2*np.pi*myerr2)
        # res = np.sum(res)
        # if smooth: outval = np.round(outval, 8)

        ## Wikipedia says:
        # ln(p(x)) = -0.5 * (x-mu/sigma)**2 - ln(sigma*sqrt(2*pi)) 
        self._res = _res
        # outval = np.sum(-.5*_res - np.log(2*np.pi*myerr)) ## This would be false
        # outval = np.sum(-.5*_res - np.log(np.sqrt(2*np.pi)*np.sum(myerr)))
        n = len(myerr)
        # outval = -0.5*np.sum(_res) - (n/2)*np.log(np.sum(myerr**2)) - (n/2)*np.log(2*np.pi)
        
        ## Did I mess up the likelihood?
        outval = -0.5*np.sum(_res) - 0.5*np.sum(np.log(2*np.pi*myerr**2)) #- (n/2)*np.log(2*np.pi)
        
        return outval

        
    def gaussian(self, x, sigma=0.5, mu=0.):
        return 1.0/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(x-mu)**2 / sigma**2)

    # def lnprior(theta):
    #     a, b, c = theta
    #     #flat priors on b, c
    #     if not 1.0 < b < 2.0 and c > 0:
    #         return -np.inf
    #     #gaussian prior on a and c
    #     mu = 0.5
    #     sigma = 0.1
    #     ### your prior is gaussian * (1/c), take natural log is the following:
    #     return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(a-mu)**2/sigma**2 - np.log(c)


    def lnprior(self, par):
        '''Function setting priors on the problem.
        - First set of priors prevent values from running out of bounds.
        NB: The interpolation function allows to extrapolate in Teff. So we
        set the limit to 200 K below the lowest Teff.
        - Second set of priors ensures that the filling factors add up to one
        and are all positive.'''

        # c1, c2, c3, c4, _T, _L, _M, _A = par
        # if self.logCoeffs:
        #     c1, c2, c3, c4 = np.exp(c1), np.exp(c2), np.exp(c3), np.exp(c4) 
        # c0 = 1 - (c1+c2+c3+c4)
        # coeffs = np.array([c0, c1, c2, c3, c4])

        if par is None:
            _T = self._T; _T2 = self._T2; _L = self._L; _M = self._M; _A = self._A
            vb = self.vb; rv = self.rv; vsini = self.vsini; vmac = self.vmac
            coeffs = self.coeffs; veilingFacToFit = self.veilingFacToFit; _fillTeffs = self.fillTeffs
        else:
            coeffs, _T, _L, _M, _A, vb, rv, vsini, vmac, veilingFacToFit, \
               _T2, _fillTeffs = self.unpackpar(par)

        # if np.isnan(coeffs[0]):
        #     # print('NaN in coeffs[0]')
        #     # print(coeffs[1:])
        #     # print(np.exp(coeffs[1:]))
        #     coeffs[0] = -np.inf

        ## Priors on the atmospheric parameters (inf if too far from grid)
        if (_T<self.teffs[0]-200) | (_T>self.teffs[-1]):
            return -np.inf
        if _T2 is not None:
            if (_T2<self.teffs[0]-200) | (_T2>self.teffs[-1]):
                return -np.inf
        if (_L<self.loggs[0]-0.5) | (_L>self.loggs[-1]): ## Extrapolate down 0.5 dex.
            return -np.inf
        if (_M<self.mhs[0]) | (_M>self.mhs[-1]):
            return -np.inf
        if (_A<self.alphas[0]) | (_A>self.alphas[-1]):
            return -np.inf

        ## Apply priors to forbid negative vsini, vmac and vb values.
        if vb<0:    return -np.inf
        if vsini<0: return -np.inf
        if vmac<0:  return -np.inf

        ## Priors on radial velocity (our guess RV should be precise to at least 200km/s)
        if abs(rv)>200: return -np.inf

        ## Veiling factor should not be negative
        # if veilingFac<0: return -np.inf
        if np.any(veilingFacToFit<0): return -np.inf
        if np.any(veilingFacToFit>5): return -np.inf

        ## Priors on magnetic filling factors
        valmin = 0; valmax = 0
        returnthis = False
        if self.logCoeffs:
            ## !!! This is broken in the zero-field case
            if np.max(coeffs)>0.:
                # print(f'REASON 1: {coeffs}')
                returnthis = True
            ## Make sure that we cannot have magnetic coeffs summing to more than 1
            # if len(coeffs)>1:
            #     if np.sum(np.exp(coeffs[1:]))>1:
            #         returnthis = True
            # if np.any(np.isnan(coeffs)):
            #     print(f'REASON 2: {coeffs}')
            #     returnthis = True
            # if np.min(coeffs) < -10000:
            #     returnthis = True
            # ## Try something else, that should in theory be sort of equivalent
            # localcoeffs = np.copy(coeffs)
            # localcoeffs = np.exp(localcoeffs)
            # if len(localcoeffs)>1:
            #     localcoeffs[0] = np.sum(localcoeffs[1:])
            # if (np.min(localcoeffs)<0.):
            #     valmin = abs(np.min(localcoeffs))
            #     returnthis = True
            # if (np.max(localcoeffs)>1.):
            #     valmax = np.max(localcoeffs)-1
            #     returnthis = True 
        else:
            if (np.min(coeffs)<0.):
                valmin = abs(np.min(coeffs))
                returnthis = True
            if (np.max(coeffs)>1.):
                valmax = np.max(coeffs)-1
                returnthis = True

        ## Priors on Teff2
        if np.min(_fillTeffs)<0.:
            returnthis = True
        if np.max(_fillTeffs)>1.:
            returnthis = True
        
        # ## Gaussian priors on the filling factors
        # mylng = 0
        # if not returnthis: ## If we are not completely our of bounds
        #     for coeff in coeffs:
        #         print(coeff)
        #         g = self.gaussian(coeff)
        #         lng = np.log(g)
        #         mylng+=lng
        if returnthis:
            # print(f'REASON 2: {coeffs}')
            # print('Returnining -np.inf')
            # print('---')
            val = max(valmin, valmax)
            return -np.inf
        else:
            # print(f'REASON 3: {coeffs}')
            # print('Returnining 0')
            # print('---')
            return 0.0

    def prior_transform(self, u):
        '''I am trying to now implement a Nested sampling approach with dynasty instead of a MCMC.
        This is the prior_transform function required by dynasty for uniform priors.'''
        ##
        # if u is None:
        #     _T = self._T; _T2 = self._T2; _L = self._L; _M = self._M; _A = self._A
        #     vb = self.vb; rv = self.rv; vsini = self.vsini; vmac = self.vmac
        #     coeffs = self.coeffs; veilingFac = self.veilingFac; _fillTeffs = self.fillTeffs
        # else:
        #     coeffs, _T, _L, _M, _A, vb, rv, vsini, vmac, veilingFac, \
        #        _T2, _fillTeffs = self.unpackpar(u)
        
        ## Compute the prior of each of the parameter:
        ## Temperature between
        ## we scale the "unit cube"
        #
        ## Run through conditions
        nbOfFields = len(self.bs) ## This will helps us unpack par
        ranges = [] ## Those are the ranges for priors
        #
        idxStart = 0
        if self.fitFields:
            idxStart = nbOfFields-1
            for i in range(idxStart):
                ranges.append((0, 1))
        ## Grab the T, L, M, A
        i = idxStart
        if self.fitTeff:
            ranges.append((self.teffs[0]-200, self.teffs[-1]))
            i += 1
        if self.fitLogg:
            ranges.append((self.loggs[0], self.loggs[-1]))
            i += 1
        if self.fitMh:
            ranges.append((self.mhs[0], self.mhs[-1]))
            i += 1
        if self.fitAlpha:
            ranges.append((self.alphas[0], self.alphas[-1]))
            i += 1
        ## Loop through the parameters
        if self.fitbroad:
            ranges.append((0, 300))
            i += 1
        if self.fitrv:
            ranges.append((-20, 20))
            i+=1
        if self.fitrot:
            ranges.append((0, 300))
            i+=1
        if self.fitmac:
            ranges.append((0, 300))
            i+=1        
        if self.fitVeiling:
            for j in range(self.nbFitVeil):
                ranges.append((0, 10))
            i+=1+self.nbFitVeil
        if self.fitTeff2: ## Second temperature
            ranges.append((self.teffs[0]-200, self.teffs[-1]))
            i += 1
            ranges.append((self.teffs[0]-200, self.teffs[-1]))
            i += 1
        
        theta = np.zeros_like(u)
        for i in range(len(ranges)):
            theta[i] = ranges[i][0] + u[i] * (ranges[i][1] - ranges[i][0])

        theta[:idxStart] = theta[:idxStart] / (1-np.sum(theta[:idxStart])) ## Ensures the sum of ALL coeffs. 

        return theta

    def lnprob(self, par):
        lp = self.lnprior(par)
        return lp + self.lnlike(par)

    def init_guess(self, initial=None):
        '''Function to produce an educated guess on the parameters to use'''
        
        if initial is not None:
            self.initial = initial
            return self.initial
        
        _fname = files("asap.support_data.ref_params").joinpath("cristofari_2022b.txt")
        f = open(_fname, 'r')

        refstars = {}
        for line in f.readlines():
            sl = line.split()
            if len(sl)<2:
                continue
            refstars[sl[0]] = {'teff':float(sl[1]),
                                'logg':float(sl[3]),
                                'mh':float(sl[5]),
                                'alpha':float(sl[7])}
        f.close()

        coeffs = self.coeffs
        # if len(self.bs)==1:
        #     coeffs = None
        # else:
        #     coeffs = np.ones(len(self.bs)-1) * 0.01 ## Only magnetic components
        # # coeffs[0] = 1. - (len(self.bs)-1) * 0.01

        # coeffs = np.array([0.95, 0.02, 0.01, 0.01, 0.01])
        if coeffs is not None:
            ## PIC: I say that if in log mode, the equation is such that:
            ## S = Sum Si*exp(log(ai)) 
            if self.logCoeffs: coeffs = np.log(coeffs) ## If log coeffs
        # c0, c1, c2, c3, c4 = coeffs

        _star = self.star.lower().replace('gl', 'gj').strip()
        if _star in refstars.keys():
            _T = refstars[_star]['teff']
            _L = refstars[_star]['logg']
            _M = refstars[_star]['mh']
            _A = refstars[_star]['alpha']
            # initial = np.array([c1, c2, c3, c4, _T, _L, _M, _A])
        else:
            _T, _L, _M, _A = 3350, 5.05, -0.34, 0.1
            # initial = np.array([c1, c2, c3, c4, 3350, 5.05, -0.34, 0.1])
        if self.fitTeff2:
            _T2 = _T
        else:
            _T2 = None ## The default value is None if we don't fit, _T if we fit _T2
        if self._T is not None: _T = self._T
        if self._T2 is not None: _T2 = self._T2
        if self._L is not None: _L = self._L
        if self._M is not None: _M = self._M
        if self._A is not None: _A = self._A
        fillTeffs = self.fillTeffs
        ## check that not out of bounds (Commented out to allow for extrapolation)
        # if _T < self.teffs[0]: _T = self.teffs[0]+5 #???
        # if _T2 is not None:
        #     if _T2 < self.teffs[0]: _T2 = self.teffs[0]+5
        if _L < self.loggs[0]: _L = self.loggs[0]+0.01
        if _M < self.mhs[0]: _M = self.mhs[0]+0.01
        if _A < self.alphas[0]: _A = self.alphas[0]+0.01
        if _T > self.teffs[-1]: _T = self.teffs[-1]-5
        if _L > self.loggs[-1]: _L = self.loggs[-1]-0.01
        if _M > self.mhs[-1]: _M = self.mhs[-1]-0.01
        if _A > self.alphas[-1]: _A = self.alphas[-1]-0.01
        # if coeffs is not None:
        # if len(coeffs)>1:
        if self.fitFields:
            initial = np.array(coeffs[1:])
            if self.fitTeff:
                initial = np.append(initial, _T)
            if self.fitLogg:
                initial = np.append(initial, _L)
            if self.fitMh:
                initial = np.append(initial, _M)
            if self.fitAlpha:
                initial = np.append(initial, _A)
        else:
            initial = np.array([])
            if self.fitTeff:
                initial = np.append(initial, _T)
            if self.fitLogg:
                initial = np.append(initial, _L)
            if self.fitMh:
                initial = np.append(initial, _M)
            if self.fitAlpha:
                initial = np.append(initial, _A)
        if self.fitbroad:
            initial = np.append(initial, self.vb) ## initial guess
        if self.fitrv:
            initial = np.append(initial, self.rv) ## initial guess
        if self.fitrot:
            initial = np.append(initial, self.vsini) ## initial guess
        if self.fitmac:
            initial = np.append(initial, self.vmac) ## initial guess
        if self.fitVeiling:
            initial = np.append(initial, self.veilingFacToFit) ## initial guess
        if self.fitTeff2:
            initial = np.append(initial, _T2)
            initial = np.append(initial, fillTeffs[1]) ## We only fit the second component and deduce the first one
        self.initial = initial
        self._T = _T; self._T2 = _T2; self._L = _L; self._M = _M; self._A = _A
        # if coeffs is None: ## It should never be None now
        #     self.coeffs = np.array([1])
        # else:
        ## Define the ndim parameter
        # self.coeffs = coeffs
        ndim = len(self.initial)
        self.ndim = ndim
        if (not self.fitFields) & (not self.fitTeff) & (not self.fitLogg) \
            & (not self.fitMh) & (not self.fitAlpha) & (not self.fitbroad) \
                & (not self.fitrv) & (not self.fitrot) & (not self.fitmac):
                self.sampler = None
                self.save_results()
                print("You requested none of the parameters to be "
                                + "fitted. We will save the results assuming the provided values are absolutely correct.")
                exit(0)
        return self.initial

    def init_weights(self, weights=None):
        '''Function to produce an educated guess on the parameters to use'''
        
        if weights is not None:
            self.weights = weights
            return self.weights
        # w = np.array([1e-2, 1e-2, 1e-2, 1e-2, 5, 1e-2, 1e-2, 1e-2])
        w = np.ones(self.ndim) * 1e-2
        ## The weight for temperature must be update (at least this one
        # for now). It is the first right after the mag fields
        ## Here the number of fields is actually the number of fields used to FIT
        ## So, if we do not fit the data, the number of fields is 1
        if not self.fitFields:
            nbOfFields = 1
        else:
            nbOfFields = len(self.bs) ## This will helps us unpack par
        # if self.logCoeffs:
        #     w[:nbOfFields-1] = np.log(w[:nbOfFields-1])
        ## The order for parameters are [bi, bi+1, ..., Teff, Teff2, logg, mg, alpha, ]
        if self.fitTeff:
            w[nbOfFields-1] = 5
        if self.fitTeff2: 
            w[-3] = 5 ## The fillTeffs are the last two
        # ## Try a larger set of priors for the magnetic field
        # w[:nbOfFields-1] = w[:nbOfFields-1]*0 + 0.1
        # if expmode: w = np.log(w)
        # p0 = [np.array(self.initial) + w * np.random.randn(ndim) \
        #         for i in range(self.nwalkers)]
        ## ----
        ## The following block of code tries to generate the initial position
        ## of the walkers while avoiding them to start from outside the 
        ## boundaries set by priors.
        ## If for some reason we can't do that after some time, the program
        ## terminates raising an Exception.
        p0 = []
        i=0; nit=0
        while (i < self.nwalkers) & (nit<100*self.nwalkers) :
            _mylocinitial = np.copy(self.initial)
            ## Those are manipulations on the filling factors
            ## They only make sense if we have more than one.
            # if nbOfFields>1:
            #     if self.logCoeffs:
            #         ## Convert these filling factors to a exp()
            #         _mylocinitial[:nbOfFields-1] = np.exp(_mylocinitial[:nbOfFields-1])
            ini = np.array(_mylocinitial) + w * np.random.randn(self.ndim)
            # if nbOfFields>1:
            #     if self.logCoeffs:
            #         ## Convect the the initial back to log
            #         ini[:nbOfFields-1] = np.log(ini[:nbOfFields-1])
            test = self.lnprior(ini)
            if np.isfinite(test):
                p0.append(ini)
                i+=1
            nit+=1
            # print("step: {}".format(nit), end='\r')
            # print("step: {}".format(test), end='\r')
        if len(p0) < self.nwalkers:
            print(p0)
            raise Exception("Fatal error - could not generate weights "
                            + "to begin MCMC while avoiding out of prior "
                            + "initial  guesses.")
        ## ----

        self.weights = p0

        return np.copy(self.weights)

    def mcmc(self):
        # Set up the backend
        # Don't forget to clear it in case the file already exists
        filename = self.opath + "tutorial.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(self.nwalkers, self.ndim)

        # import corner
        with Pool(self.ncores) as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, 
                                            self.lnprob, pool=pool, 
                                            backend=backend) ## save to file
            itime = time.time()
            sampler.run_mcmc(self.weights, self.nsteps, progress=True)
            etime = time.time()
            print("Time = {:.2f} seconds".format(etime - itime))
            f = open(self.opath+'time.txt', 'w')
            f.write("Initial guess: " + str(self.initial) + " \n")
            f.write("Time = {:.2f} seconds\n".format(etime - itime))
            f.close()
        ## Save stuff to files
        tau = sampler.get_autocorr_time(tol=0)
        log_prob_walkers_noflat = sampler.get_log_prob()
        np.save(self.opath+'log_prob_walkers_noflat.npy', 
                log_prob_walkers_noflat)
        np.save(self.opath+'tau.npy', tau)
        print("Max autocorrelation time: {:0.2f}".format(np.max(tau)))
        f = open(self.opath+'time.txt', 'a')
        f.write("Max autocorrelation time: {:0.2f}\n".format(np.max(tau)))
        f.close()

        ncpu = cpu_count()
        f = open(self.opath+'time.txt', 'a')
        f.write("{0} CPUs\n".format(ncpu))
        f.close()
        print("{0} CPUs".format(ncpu))
        self.sampler = sampler
        return sampler

    #### Some functions
    ## This implementation is much faster. Becomes the default.
    def count_elements(self, arr):
        '''To create a histogram'''
        # sortarr = np.sort(arr)
        values, counts = np.unique(arr, return_counts = True)
        hist = {}
        for iv in range(len(values)):
            hist[values[iv]] = counts[iv]
        return hist

    def maxdir(self, mydir):
        '''Find max in dir'''
        maxval = 0
        maxkey = 0
        for key in mydir.keys():
            _val = mydir[key]
            if _val > maxval:
                maxkey = key
                maxval = _val
        return maxkey, maxval

    def save_results(self):
        '''Function to save the results of the MCMC'''
        if self.sampler is not None:
            if self.dynesty:
                samples = np.array([self.sampler.results.samples])
            else:
                samples = self.sampler.get_chain()
                log_prob_walkers_noflat_0 = self.sampler.get_log_prob()
            if self.logCoeffs:
                samples[:,:,:len(self.bs)-1] = np.exp(samples[:,:,:len(self.bs)-1])
            np.save(self.opath+"samples", samples)
            np.save(self.opath+"weights", samples)

        ## Reasign        
        samples_noflat_0 = samples
        data = {}
        data['nsteps'] = len(samples_noflat_0)
        data['burning'] = round(0.5*data['nsteps']) ## 50% by default
        data['bs'] = self.bs

        #### REDISCARD - If user requested to discard the samples
        ## Recompute the burning period
        ## Take the samples after burning period
        samples_noflat = samples_noflat_0[data['burning']:]
        log_prob_walkers_noflat = log_prob_walkers_noflat_0[data['burning']:]

        #### Compute the number of fields in the fit
        nbOfFields = len(self.bs) ## This is the number of fields in our model NOT WHAT WE FIT 
        
        #### Flatten the samples
        ishape = np.shape(samples_noflat)
        nshape = (ishape[0] * ishape[1], ishape[2])
        ssamples = np.reshape(np.copy(samples_noflat), nshape) ## Those are the new flatten samples
        log_prob_walkers = np.concatenate(log_prob_walkers_noflat, -1)

        ## This is taking the average of the 5% of the walkers
        percent = .05
        nblim = int(round(percent*len(log_prob_walkers))) ## Thats 5%
        thslikelihood = np.sort(log_prob_walkers)[-nblim]
        idx50 = np.where(log_prob_walkers>=thslikelihood)
        nbofvals2 = len(idx50[0])

        labels = self.return_labels()
        correctRV = False
        if correctRV:
            #### Recenter the radial velocity
            ## We have to change the values of the RV for all the samples
            ## Find the index corresponding to the RV
            is_rv = np.array(['rv' in labels[i].lower() for i in range(len(labels))])
            if np.all(is_rv==False):
                ## RV was not fitted, ignore that step
                pass
            else:
                where_rv = np.where(is_rv)[0][0]
                #### Recenter the radial velocity
                ## !!! This is to make a nice plot but the value then has little sense
                subssamples = ssamples.T[where_rv] ## Those are the rvs
                subssamples = subssamples - np.median(subssamples)
                ssamples.T[where_rv] = ssamples.T[where_rv] - np.median(ssamples.T[where_rv])

        ## Compute the mean field from the samples
        if self.fitFields:
            ## If we are fitting fields, we are fitting nbOfFields-1 filling factors
            subssamples = ssamples.T[:nbOfFields-1]
            meanfield = np.sum(subssamples.T * data['bs'][1:], axis=1) ## only from magnetic coefficients
        
        ## Compute the first coeff and put it in place
        if self.fitFields:
            subssamples = (ssamples.T)[:nbOfFields-1]
            firstcoeff = 1 - np.sum(subssamples, axis=0)
            # Append the first coeff
            nssamples = np.empty((len(ssamples), len(ssamples[0])+1)).T
            nssamples[0] = firstcoeff
            for i in range(len(ssamples[0])):
                nssamples[i+1] = (ssamples.T)[i]
            nssamples = nssamples.T
            ## Add the label for non-magnetic component to list of labels
            labels.insert(0, r'$a_0$')
        else:
            nssamples = ssamples

        ndim = len(labels) ## dimensions of nssamples
        data['ndim'] = ndim

        data['gen_files'] = []

        ## From this point forward, nssamples contains the 0kG component (which we did not fit directly)

        ###################################
        #### PLOT 1 - FULL CORNER PLOT ####
        ###################################
        import corner

        cornerfont = 25
        CORNER_KWARGS = dict(
            smooth=0.5,
            label_kwargs=dict(fontsize=cornerfont),
            title_kwargs=dict(fontsize=cornerfont),
            quantiles=[0.16, 0.5, 0.84], # That's 1 sigma
            # quantiles=[0.02, 0.5, 0.98], # That's 3 sigma
            verbose=False,
            titles=["" for i in range(len(labels))],
            # levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
            # plot_density=False,
            # plot_datapoints=False,
            fill_contours=True,
            show_titles=True,
            max_n_ticks=3,
            # title_fmt=".2E",
            labels=labels
        )

        plottrig = True
        if plottrig:
            print("-> Generating full corner plot")
            fig = corner.corner(nssamples, **CORNER_KWARGS)

            ## Now we want to remove the equal sign from titles
            for i in range(len(fig.axes)):
                fig.axes[i].set_title(fig.axes[i].title.get_text().replace("=", ''))

            ## Make the subplot smaller?
            # fig.subplots_adjust(right=1.5,top=1.5)

            ## Make the ticks bigger
            for ax in fig.get_axes():
                ax.tick_params(axis='both', labelsize=cornerfont-5)
                ax.title.set_fontsize("{}".format(cornerfont))

            max50 = nssamples[idx50]
            max = np.mean(max50, axis=0)

            # Extract the axes
            _ndim = data['ndim']
            axes = np.array(fig.axes).reshape((_ndim, _ndim))
            for i in range(_ndim):
                for j in range(i):
                    ax = axes[i, j]
                    ax.axhline(max[i], color='red')
                    ax.axvline(max[j], color='red')

            for i in range(_ndim):
                ax = axes[i, i]
                ax.axvline(max[i], color='red')

            plt.savefig(self.opath+'corner.pdf', bbox_inches='tight')
            plt.close()
            data['gen_files'].append('corner.pdf')


        ################################
        #### PLOT 2 - <B> HISTOGRAM ####
        ################################


        if plottrig:
            print("-> Generating <B> histogram")
            ## Now plot the B field only
            if self.fitFields:
                _ndim = 1
                _labels = ['<B> (kG)']
                CORNER_KWARGS = dict(
                    smooth=0.5,
                    label_kwargs=dict(fontsize=18),
                    title_kwargs=dict(fontsize=18),
                    quantiles=[0.16, 0.5, 0.84],
                    # titles=["" for i in range(len(labels))],
                    # levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                    # plot_density=False,
                    # plot_datapoints=False,
                    fill_contours=True,
                    show_titles=True,
                    max_n_ticks=3,
                    # title_fmt=".2E",
                    labels=_labels
                )
                ## Make the ticks bigger
                for ax in fig.get_axes():
                    ax.tick_params(axis='both', labelsize=16)
                    ax.title.set_fontsize("16")
                ## Corner plots
                fig = corner.corner(meanfield,**CORNER_KWARGS)
                # Extract the axes
                axes = np.array(fig.axes).reshape((_ndim, _ndim))
                ## Compute max likelihood
                ## There are two alternatives posible
                #
                # 1 - get the maximum of the distributions
                res = self.count_elements(np.round(meanfield, 3))
                maxres = self.maxdir(res)
                #

                # from IPython import embed
                # embed()
                # 2 - get the maximum of likelihood for the distribution
                idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
                maxpos = meanfield[idx]
                #
                # from IPython import embed
                # embed()
                # THISISATEST: we try to take the average of the maxima of the 50 highest points
                # idxsort = np.argsort(log_prob_walkers)
                # sortedmeanfield = meanfield[idxsort]
                # max50 = sortedmeanfield[-50:]
                max50 = meanfield[idx50]
                maxpos = np.array([np.mean(max50)])

                ax = axes[0,0]
                ax.axvline(maxres[0], color='black')
                ax.axvline(maxpos[0], color='red')

                # subssamples = nssamples.T[1:nbOfFields]
                # meanfield_ssamples = np.sum(subssamples.T * self.bs[1:], axis=1)
                mcmc_meanfield = np.percentile(meanfield, [16, 50, 84])
                q_meanfield = np.diff(mcmc_meanfield)
                meanfield_tradi = mcmc_meanfield[1]
                emeanfield_tradi = np.mean(q_meanfield)

                # Store the result in a variable
                maxproba_meanfield = maxpos[0]
                maxdistrib_meanfield = maxres[0]
                #
                emaxproba_meanfield = emeanfield_tradi
                emaxdistrib_meanfield = emeanfield_tradi
                #
                plt.savefig(self.opath+'b_histogram.pdf')
                plt.close()
                data['gen_files'].append('b_histogram.pdf')


        ############################
        #### PLOT 3 - a0 -- <B> ####
        ############################


        if plottrig:
            print("-> Generating the a0 vs <B> plot")
            ## Now plot the B field and non mag component
            if self.fitFields:
                _labels = ['<B> (kG)', r"$a_0$"]
                _ndim = len(_labels)
                CORNER_KWARGS = dict(
                    smooth=0.5,
                    label_kwargs=dict(fontsize=18),
                    title_kwargs=dict(fontsize=18),
                    quantiles=[0.16, 0.5, 0.84],
                    titles=["" for i in range(len(labels))],
                    # levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                    # plot_density=False,
                    # plot_datapoints=False,
                    fill_contours=True,
                    show_titles=True,
                    max_n_ticks=3,
                    # title_fmt=".2E",
                    labels=_labels
                )

                ## Make the ticks bigger
                for ax in fig.get_axes():
                    ax.tick_params(axis='both', labelsize=16)
                    ax.title.set_fontsize("16")
                ## Corner plots
                non_mag = nssamples.T[0]
                nonmag_meanfield = np.array([meanfield, non_mag])
                fig = corner.corner(nonmag_meanfield.T,**CORNER_KWARGS)
                # print('If I am right this is the mean field: {} '.format(np.median(nonmag_meanfield[1])))
                # print('And so this is the max field: {} '.format(np.max(nonmag_meanfield[1])))
                idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
                # print('But I really want the position of the max likelihood: {} '.format(idx))
                # print('Which gives: {} '.format(nonmag_meanfield[1][idx]))
                # print('In the meantime if I take the coeffs for the max likelihood...')

                be = nssamples[idx][0][:nbOfFields];
                maxlikesum = np.sum(be*data['bs'])
                # print('And compute the associate Bf, I get: {}'.format(maxlikesum))
                # print('But if we do what we used to do, then we get the coeffs from the median'.format(maxlikesum))
                meds = []
                for i in range(nbOfFields):
                    nnn = nssamples.T
                    med = np.median(nnn[i])
                    meds.append(med)
                meds = np.array(meds)
                newmeds = np.copy(meds)
                newmeds[0] = 1 - np.sum(meds[1:])
                medlikesum = np.sum(meds*data['bs'])
                ## Now we want to remove the equal sign from titles
                for i in range(len(fig.axes)):
                    fig.axes[i].set_title(fig.axes[i].title.get_text().replace("=", ''))

                # from IPython import embed
                # embed()
                ## What is the maximum of the 0 comp?
                idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
                # maxfirstcoeff = nssamples[idx][0]
                maxfirstcoeff = np.mean(nssamples[idx50], axis=0)
                # Extract the axes
                axes = np.array(fig.axes).reshape((_ndim, _ndim))
                for i in range(_ndim):
                    for j in range(i):
                        ax = axes[i, j]
                        ax.axvline(maxpos[0], color='red')
                        ax.axhline(maxfirstcoeff[0], color='red')

                axes[0, 0].axvline(maxpos[0], color='red')
                axes[1, 1].axvline(maxfirstcoeff[0], color='red')

                plt.savefig(self.opath+'a0_b.pdf')
                plt.close()
                data['gen_files'].append('a0_b.pdf')



        ############################
        #### PLOT 3 - a0 -- <B> ####
        ############################

        if plottrig:

            plt.close('all')

            if self.fitFields:

                params= {'xtick.labelsize': 18,'ytick.labelsize': 18,'axes.labelsize': 20, 'legend.fontsize': 16,   'text.usetex': True,'figure.figsize' : (6.4, 4.8)}
                plt.rcParams.update(params)

                xaxis = self.bs
                width = np.median(np.diff(self.bs))*0.95
                plt.bar(xaxis, self.coeffs, width=width, color='black')
                plt.ylabel('Filling factor')
                plt.xlabel('Field strength (kG)')
                # Extract the axes
                plt.tick_params(which='minor',direction='in',axis='both',bottom='on', top='on', left='on', right='on', length=5)
                plt.tick_params(which='major',direction='in',axis='both',bottom='on', top='on', left='on', right='on', length=10)
                plt.tight_layout()
                plt.savefig(self.opath+'b_distrib.pdf')
                plt.close()
                data['gen_files'].append('b_distrib.pdf')


        ##########################
        #### PLOT 4 - samples ####
        ##########################
        ## I did not reconstruct the zero-magnetic field for the non-flattened samples.
        ## So IF we fit the fields, we need to remove the first one.
        if self.fitFields:
            _ndim = data['ndim']-1
            _labels = labels[1:]
        else:
            _ndim = data['ndim']
            _labels = labels

        figheightfac = len(_labels)/2 # Used to enlarge the figures
        # ----
        ## Without burning
        if plottrig:
            print("-> Generating samples plots")
            fig, axes = plt.subplots(_ndim, figsize=(6.4, figheightfac*4.8), sharex=True)
            if _ndim == 1:
                i = 0
                ax = axes
                ax.plot(samples_noflat_0[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples_noflat_0))
                ax.set_ylabel(_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ax.set_xlabel("step number");
            else:
                for i in range(_ndim):
                    ax = axes[i]
                    ax.plot(samples_noflat_0[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples_noflat_0))
                    ax.set_ylabel(_labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number");
            plt.savefig(self.opath+'samples.pdf')
            # plt.show()
            plt.close()
            data['gen_files'].append('samples.pdf')

        ## With burning
        if plottrig:
            fig, axes = plt.subplots(_ndim, figsize=(6.4, figheightfac*4.8), sharex=True)
            if _ndim == 1:
                i = 0
                ax = axes
                ax.plot(samples_noflat[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples_noflat[:]))
                ax.set_ylabel(_labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ax.set_xlabel("step number");
            else:
                for i in range(_ndim):
                    ax = axes[i]
                    ax.plot(samples_noflat[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples_noflat[:]))
                    ax.set_ylabel(_labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number");
            plt.savefig(self.opath+'samples_postburn.pdf')
            # plt.show()
            plt.close()
            data['gen_files'].append('samples_postburn.pdf')

        #### Here we save the values of the results to be stored
        ## Grab values
        mcmcs_tradi = []
        emcmcs_tradi = []
        mcmcs_maxdistrib = []
        emcmcs_maxdistrib = []
        mcmcs_maxproba = []
        emcmcs_maxproba = []

        def magnitude(x):
            return int(round(np.log10(x), 0))
        
        for i in range(len(nssamples[0])):
            ## Compute the median and error bars "traditionally"
            mcmc = np.percentile(nssamples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            # 1 - get the maximum of the distributions
            roundfac = -1*magnitude(np.mean(q))
            if roundfac<0: roundfac=0
            res = self.count_elements(np.round(nssamples[:, i], roundfac))
            maxdistrib = self.maxdir(res)
            #
            # THISISATEST
            # idxsort = np.argsort(log_prob_walkers)
            # sortedsamples = nssamples.T[i][idxsort]
            # max50 = sortedsamples[-50:]
            max50 = nssamples.T[i][idx50]
            maxproba = np.array([np.mean(max50)])
            # Raise a warning if multiple maxima were found
            if len(maxproba)>1:
                if np.any(np.diff(maxproba)>0.001): ## We have different walkers yielding maxima in different places
                    print('CAUTION: Possible multiple maxima detected')

            ## Save the results
            mcmcs_tradi.append(mcmc[1])
            mcmcs_maxproba.append(maxproba[0])
            mcmcs_maxdistrib.append(maxdistrib[0])
            #
            emcmcs_tradi.append(np.mean(q))
            emcmcs_maxproba.append(np.mean(q)) #emaxproba) ## Default to percentiles
            emcmcs_maxdistrib.append(np.mean(q)) ## Default to percentiles

        # ## With the results we can compute the missing magnetic coeff (for 0~kG)
        # ## Actually this is re-computing the missing coeff from the others... Is this a good idea?
        # if (self.fitFields and (nbOfFields>1)):
        #     coeffs_tradi = mcmcs_tradi[:nbOfFields+1]
        #     coeffs_tradi[0] = 1 - np.sum(self.coeffs[1:]) ## This apperrs to make a copy of the mcmcs array
        #     ecoeffs_tradi = emcmcs_tradi[:nbOfFields+1]
        #     #
        #     coeffs_maxproba = mcmcs_maxproba[:nbOfFields+1]
        #     ecoeffs_maxproba = emcmcs_maxproba[:nbOfFields+1]
        #     #
        #     coeffs_maxdistrib = mcmcs_maxdistrib[:nbOfFields+1]
        #     ecoeffs_maxdistrib = emcmcs_maxdistrib[:nbOfFields+1]
        # else:
        #     coeffs_tradi = np.zeros(len(self.bs))
        #     coeffs_tradi[0] = 1
        #     ecoeffs_tradi = np.zeros(len(self.bs))
        #     #
        #     coeffs_maxproba = np.zeros(len(self.bs))
        #     coeffs_maxproba[0] = 1
        #     ecoeffs_maxproba = np.zeros(len(self.bs))
        #     #
        #     coeffs_maxdistrib = np.zeros(len(self.bs))
        #     coeffs_maxdistrib[0] = 1
        #     ecoeffs_maxdistrib = np.zeros(len(self.bs))

        if (self.fitFields and (nbOfFields>1)):
            subssamples = nssamples.T[1:nbOfFields] ## Without the 0kG component
            meanfield_ssamples = np.sum(subssamples.T * self.bs[1:], axis=1)
            mcmc_meanfield = np.percentile(meanfield_ssamples, [16, 50, 84])
            q_meanfield = np.diff(mcmc_meanfield)
            meanfield_tradi = mcmc_meanfield[1]
            emeanfield_tradi = np.mean(q_meanfield)
            #
            # 1 - get the maximum of the distributions
            res = self.count_elements(np.round(meanfield_ssamples, 3))
            maxres = self.maxdir(res)
            #
            # 2 - get the maximum of likelihood for the distribution
            # idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
            # maxpos = meanfield[idx]
            max50 = meanfield[idx50]
            maxpos = np.array([np.mean(max50)])
            emaxpos = (np.max(max50) - np.min(max50))/2
            # from IPython import embed
            # embed()
            #
            # Store the result in a variable
            maxproba_meanfield = maxpos[0]
            maxdistrib_meanfield = maxres[0]
            #
            emaxproba_meanfield = emeanfield_tradi #emaxpos
            emaxdistrib_meanfield = emeanfield_tradi
        else:
            meanfield = 0
            emeanfield = 0
            maxproba_meanfield = 0
            maxdistrib_meanfield = 0
            emaxproba_meanfield = 0
            emaxdistrib_meanfield = 0

        mcmcs = np.array(mcmcs_maxproba); emcmcs = np.array(emcmcs_maxproba)
        if self.fitFields:
            coeffs = np.array(mcmcs[0:nbOfFields]); ecoeffs = np.array(emcmcs[0:nbOfFields])
        else: ## No magnetic field fitted
            coeffs = np.zeros(nbOfFields)
            coeffs[0] = 1.
            ecoeffs = np.zeros(nbOfFields) 
        ##
        meanfield = np.array(maxproba_meanfield); emeanfield = np.array(emaxproba_meanfield)
        # Compute the average magnetic field
        avfield = np.sum(self.bs * coeffs)
        eavfield = np.sqrt(np.sum((self.bs*ecoeffs)**2))

        resdict = self.get_PARAMS(mcmcs, emcmcs)

        strcoeffs = [str(resdict['a'+str(i)]) for i in range(len(self.bs))]
        strecoeffs = [str(resdict['e_a'+str(i)]) for i in range(len(self.bs))]

        resveil = [resdict['r{}'.format(band)] for band in self.veilingBands]
        eresveil = [resdict['e_r{}'.format(band)] for band in self.veilingBands]
        resveil_tofit = [resdict['r{}'.format(band)] for band in self.fitBands]
        resFillTeffs = [resdict['fillteff_0'], resdict['fillteff_1']]
        eresFillTeffs = [resdict['e_fillteff_0'], resdict['e_fillteff_1']]
        

        ## Here coeffs include the 0kG component
        fit = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, 
                    self.nan_mask, self.nwvls, self.grid_n, 
                    coeffs, resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha'],
                    self.teffs, self.loggs, self.mhs, self.alphas, resdict['vb'],
                    resdict['rv'], resdict['vsini'], resdict['vmac'], resveil_tofit, resdict['teff2'], resFillTeffs)
        ## Same, mcmc contains the 0kG component we do not want to feed to lnlike.
        ## But if there are no magnetic fields, mcmcs will NOT contain the 0kG factor !
        if self.fitFields and (nbOfFields>1):
            mcmcsForLnLike = mcmcs[1:] ## Without magnetic field component
        else:
            mcmcsForLnLike = mcmcs
        _  = self.lnlike(mcmcsForLnLike)
        minchi2 = np.sum(self._res)
        coeffsnomag = coeffs*0
        coeffsnomag[0] = 1
        fitnomag = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, 
                self.nan_mask, self.nwvls, self.grid_n, 
                coeffsnomag, resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha'],
                self.teffs, self.loggs, self.mhs, self.alphas, resdict['vb'],
                resdict['rv'], resdict['vsini'], resdict['vmac'], resveil_tofit, resdict['teff2'], resFillTeffs)
        ## Here, we want the same as the results of the mcmcs, but the magnetic components should be set to zero.
        if self.fitFields and (nbOfFields>1):
            mcmcsForLnLike_0kG = np.copy(mcmcsForLnLike)
            mcmcsForLnLike_0kG[0] = 1
            mcmcsForLnLike_0kG[1:nbOfFields] = 0
        else:
            mcmcsForLnLike_0kG = mcmcsForLnLike
        _  = self.lnlike(mcmcsForLnLike_0kG)
        minchi2exp = np.sum(self._res)
        #
        hdu = fits.PrimaryHDU()
        hdu.header['OBJECT'] = (self.star, 'object observed')
        hdu.header['NORMFAC'] = (self.normFactor, 'object observed')
        hdu1 = fits.ImageHDU(data=self.obs_wvl, name='WVL')
        hdu2 = fits.ImageHDU(data=self.obs_flux, name='FLUX')
        hdu3 = fits.ImageHDU(data=self.obs_flux_tofit, name='FLUXFIT')
        hdu4 = fits.ImageHDU(data=self.obs_err, name='ERROR')
        hdu5 = fits.ImageHDU(data=fit, name='FIT')
        hdu6 = fits.ImageHDU(data=fitnomag, name='FITNOMAG')
        hdu7 = fits.ImageHDU(data=self.IDXTOFIT, name='IDXTOFIT')
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
        hdul.writeto(self.opath+'fit-data.fits', overwrite=True)

        ## Save the normalization factor to file:
        nbPointsFitted = len(self.obs_flux_tofit[self.IDXTOFIT]) 

        p = len(mcmcs) ## number of parameters
        new_normFactor = minchi2 * self.normFactor / (nbPointsFitted - p)
        self.save_normFactor(new_normFactor)

        strcoeffs = [str(coeffs[i]) for i in range(len(coeffs))]
        strecoeffs = [str(ecoeffs[i]) for i in range(len(ecoeffs))]
        resFillTeffsString = [str(resFillTeffs[i]) for i in range(len(resFillTeffs))]
        eresFillTeffsString = [str(eresFillTeffs[i]) for i in range(len(eresFillTeffs))]

        f = open(self.opath+'factors.txt', 'w')
        f.write(" ".join(strcoeffs) + " \n")
        f.write(" ".join(strecoeffs) + " \n")
        f.write("{} {} {} {} \n".format(resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha']))
        f.write("{} {} {} {}\n".format(resdict['e_teff'], resdict['e_logg'], resdict['e_mh'], resdict['e_alpha']))
        f.write("chi2 min: {:0.5f}\n".format(minchi2))
        f.write("chi2 min no field: {:0.5f}\n".format(minchi2exp))
        f.close()

        ## See output.txt for a description of the lines
        f = open(self.opath+'results_raw.txt', 'w')
        f.write(" ".join(strcoeffs) + " \n")
        f.write(" ".join(strecoeffs) + " \n")
        f.write("{} {} {} {} \n".format(resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha']))
        f.write("{} {} {} {}\n".format(resdict['e_teff'], resdict['e_logg'], resdict['e_mh'], resdict['e_alpha']))
        f.write("Mean. field: {} {} \n".format(meanfield, emeanfield))
        f.write("Av. field: {} {} \n".format(avfield, eavfield))
        f.write("vb: {} {}\n".format(resdict['vb'], resdict['e_vb']))
        f.write("GussRV: {} {}\n".format(self.guessed_rv, 0))
        f.write("RV: {} {}\n".format(resdict['rv'], resdict['e_rv']))
        f.write("vsini: {} {}\n".format(resdict['vsini'], resdict['e_vsini']))
        f.write("vmac[{}]: {} {}\n".format(self.vmacMode, resdict['vmac'], resdict['e_vmac']))
        f.write("chi2 min: {:0.5f}\n".format(minchi2))
        f.write("chi2 min no field: {:0.5f}\n".format(minchi2exp))
        f.write("Nb. of points: {}\n".format(nbPointsFitted))
        f.write("normFactor: {}\n".format(self.normFactor))
        f.write("veilingFac: {}\n".format(resveil).replace('[', '').replace(']', '').replace(',', ' '))
        f.write("e_veilingFac: {}\n".format(eresveil).replace('[', '').replace(']', '').replace(',', ' '))
        f.write("bolLum: {} {}\n".format(self.rL, self.drL))
        f.write("absMk: {} {}\n".format(self.Mk, self.dMk))
        f.write("dist: {} {}\n".format(self.dist, self.ddist))
        f.write("Multi-teff model Teff2: {} \n".format(resdict['teff2']))
        f.write("Multi-teff model errTeff2: {} \n".format(resdict['e_teff2']))
        f.write("Multi-teff model fillTeffs: {} \n".format(" ".join(resFillTeffsString)))
        f.write("Multi-teff model errfillTeffs: {} \n".format(" ".join(eresFillTeffsString)))
        f.write("logCoeffs: {} \n".format(" ".format(self.logCoeffs)))
        f.write("Adjusted errors: {}\n".format(self.errorsAdj))
        f.write("Fit Derivative mode [T/F]: {}\n".format(self.fitDeriv))
        f.write("Error type: {}\n".format(self.errType))
        f.write("vinstru: {}\n".format(self.vinstru))
        f.close()

        print('ANALYSIS COMPLETE')

        return 0


        # labels = ["C0", "C1", "C2", "C3", "Teff", "log(g)", "[M/H]", "[a/Fe]"]
        if len(self.bs)>1:
            labels = ["C{}".format(i) for i in range(1, self.ndim-4+1)]
            labels.append("Teff"); labels.append("log(g)");
            labels.append("[M/H]"); labels.append("[a/Fe]")
            if self.fitbroad:
                labels.append("vb")
        else:
            labels = ["Teff", "log(g)", "[M/H]", "[a/Fe]"]
            if self.fitbroad:
                labels.append("vb")
            if self.fitrv:
                labels.append("RV")
            if self.fitrot:
                labels.append("vsini")
            if self.fitmac:
                labels.append("vmac[{}]".format(self.vmacMode))
            if self.fitVeiling:
                labels.append("r_veil")

        ## Plot the samples
        self.plot = False
        if self.plot:
            fig, axes = plt.subplots(self.ndim, figsize=(10, 7), sharex=True)
            if self.ndim == 1:
                i = 0
                ax = axes
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                ax.set_xlabel("step number");
            else:
                for i in range(self.ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number");
            plt.savefig(self.opath+'samples.pdf')
            # plt.show()
            plt.close()

        ## Burn samples
        burnin = 0.5
        if self.sampler is not None:
            if self.dynesty:
                ssamples = samples[0]
                weights = self.sampler.results.importance_weights()
            else:
                ssamples = self.sampler.get_chain(flat=True, 
                                            discard = int(burnin*self.nsteps))
                                            #, thin=250)
                weights = np.ones(samples.shape[0])
            if self.logCoeffs:
                ssamples[:len(self.bs)-1] = np.exp(ssamples[:len(self.bs)-1])
            if self.dynesty:
                log_prob_walkers = self.sampler.results.logl
            else:
                log_prob_walkers = self.sampler.get_log_prob(flat=True, 
                                                    discard = int(burnin*self.nsteps))
            np.save(self.opath+"log_prob_walkers", log_prob_walkers)
            np.save(self.opath+"ssamples", ssamples)
            np.save(self.opath+"weights", weights)

            ## Grab values
            if self.dynesty:
                from dynesty.utils import quantile as dyn_quantile
                # q = (0.025, 0.5, 0.975)
                q = (0.016, 0.5, 0.84)
                values = []
                for i in range(ssamples.shape[1]):
                    values.append(dyn_quantile(ssamples[:,i], q, 
                                               weights=self.sampler.results.importance_weights()))
                values = np.array(values)
                max_likelihood_idx = np.argmax(self.sampler.results.logl)
                best_fit_params = ssamples[max_likelihood_idx]
                mcmcs = best_fit_params ## replace median by max likelihood
                emcmcs = np.max(np.diff(values), axis=-1)
            else:
                mcmcs = []
                emcmcs = []
                for i in range(len(ssamples[0])):
                    mcmc = np.percentile(ssamples[:, i], [16, 50, 84])
                    q = np.diff(mcmc)
                    mcmcs.append(mcmc[1])
                    emcmcs.append(np.mean(q))
        elif self.sampler is None:
            mcmcs = [-99]; emcmcs = [-99]
        
        nbPointsFitted = len(self.obs_flux_tofit[self.IDXTOFIT]) 

        ## Compute the coefficients and seperate fundamental parameters
        # if len(self.bs)>1:
        if self.fitFields:
            coeffs = mcmcs[:len(self.bs)-1]
            # if self.logCoeffs: coeffs = np.exp(coeffs)
            coeffs = np.append(1 - np.sum(coeffs), coeffs) ## This is the last coeff
            ecoeffs = emcmcs[:len(self.bs)-1]
            # if self.logCoeffs: ecoeffs = np.exp(mcmcs[:len(self.bs)-1]) * ecoeffs
            ecoeffs = np.append(np.mean(ecoeffs), ecoeffs) ## This is the last coeff
        else:
            coeffs = self.coeffs
            ecoeffs = np.zeros(self.coeffs.shape)
            # coeffs = np.zeros(len(self.bs))
            # coeffs[0] = 1
            # ecoeffs = np.zeros(len(self.bs))

        ## Compute the resulting average field and its propagated errors
        avfield = np.sum(self.bs * coeffs)
        ## CA C'EST N'IMP : 
        eavfield = np.sqrt(np.sum((self.bs*ecoeffs)**2))
        ## La vrai relation sera plutot ??? Je ne sais pas en fait. 
        ## La relation au dessus marcherait si les filling facteurs etaient
        ## independants...
        ## We want to build the average field histogram
        # if len(self.bs)>1:
        if self.fitFields:
            subssamples = ssamples.T[:len(self.bs)-1]
            meanfield_ssamples = np.sum(subssamples.T * self.bs[1:], axis=1)
            np.save(self.opath+"ssamples_meanfield", meanfield_ssamples)
            mcmc_meanfield = np.percentile(meanfield_ssamples, [16, 50, 84])
            q_meanfield = np.diff(mcmc_meanfield)
            meanfield = mcmc_meanfield[1]
            emeanfield = np.mean(q_meanfield)
        else:
            meanfield = 0
            emeanfield = 0

        _  = self.lnlike(np.array(mcmcs))
        minchi2 = np.sum(self._res)
        # T = 3311; L = 5.11; M = -0.37; A = 0.16
        ## We compute here the results obtained assuming no magnetic fields
        ## at all.
        ## Here we also use the last values obtained with self._T etc.
        # if len(self.bs)==1:
        if not self.fitFields:
            mcmcsexp = np.array([])
            if self.fitTeff:
                mcmcsexp = np.append(mcmcsexp, self._T)
            if self.fitLogg:
                mcmcsexp = np.append(mcmcsexp, self._L)
            if self.fitMh:
                mcmcsexp = np.append(mcmcsexp, self._M)
            if self.fitAlpha:
                mcmcsexp = np.append(mcmcsexp, self._A)
            # mcmcsexp  = np.array([self._T, self._L, self._M, self._A])
        else:
            mcmcsexp = np.zeros(len(self.bs)-1)
            if self.fitTeff:
                mcmcsexp = np.append(mcmcsexp, self._T)
            if self.fitLogg:
                mcmcsexp = np.append(mcmcsexp, self._L)
            if self.fitMh:
                mcmcsexp = np.append(mcmcsexp, self._M)
            if self.fitAlpha:
                mcmcsexp = np.append(mcmcsexp, self._A)
            # mcmcsexp  = [0, 0, 0, 0, self._T, self._L, self._M, self._A]
        if self.fitbroad:
            mcmcsexp = np.append(mcmcsexp, self.vb) ## Default value
        if self.fitrv:
            mcmcsexp = np.append(mcmcsexp, self.rv) ## Default value
        if self.fitrot:
            mcmcsexp = np.append(mcmcsexp, self.vsini) ## Default value
        if self.fitmac:
            mcmcsexp = np.append(mcmcsexp, self.vmac) ## Default value
        if self.fitVeiling:
            mcmcsexp = np.append(mcmcsexp, self.veilingFac) ## Default value
        if self.fitTeff2:
            mcmcsexp = np.append(mcmcsexp, self._T2)
            mcmcsexp = np.append(mcmcsexp, self.fillTeffs)
        # minchi2exp = -2*self.lnlike(mcmcsexp)
        _  = self.lnlike(mcmcsexp)
        minchi2exp = np.sum(self._res)

        ## Here again the number of fields must be the nunber that we fit +1
        if self.fitFields:
            nbOfFields = len(self.bs)
        else:
            nbOfFields = 1
        idxStart = nbOfFields - 1
        i = idxStart
        if self.fitTeff:
            resT = mcmcs[i]; eresT = emcmcs[i]
            _teff_id = i 
            i+=1
        else:
            resT = self._T; eresT = 0
        if self.fitLogg:
            resL = mcmcs[i]; eresL = emcmcs[i]
            i+=1
        else:
            resL = self._L; eresL = 0
        if self.fitMh:
            resM = mcmcs[i]; eresM = emcmcs[i]
            _mh_id = i 
            i+=1
        else:
            resM = self._M; eresM = 0
        if self.fitAlpha:
            resA = mcmcs[i]; eresA = emcmcs[i]
            i+=1
        else:
            resA = self._A; eresA = 0
        if self.fitbroad:
            resvb = mcmcs[i]; eresvb = emcmcs[i]
            i+=1
        else:
            resvb = self.vb; eresvb = 0
        if self.fitrv:
            resrv = mcmcs[i]; eresrv = emcmcs[i]
            i+=1
        else:
            resrv = self.rv; eresrv = 0
        if self.fitrot:
            resvsini = mcmcs[i]; eresvsini = emcmcs[i]
            i+=1
        else:
            resvsini = self.vsini; eresvsini = 0
        if self.fitmac:
            resvmac = mcmcs[i]; eresvmac = emcmcs[i]
            i+=1
        else:
            resvmac = self.vmac; eresvmac = 0
        if self.fitVeiling:
            resveil = mcmcs[i:i+self.nbFitVeil]; eresveil = emcmcs[i:i+self.nbFitVeil]
            i+=1+self.nbFitVeil
        else:
            resveil = self.veilingFac; eresveil = np.zeros(len(self.veilingFac))
        if self.fitTeff2:
            resT2 = mcmcs[i]; eresT2 = emcmcs[i]
            i+=1
            resFillTeffs = np.array([1-mcmcs[i], mcmcs[i]])
            eresFillTeffs = np.array([emcmcs[i], emcmcs[i]])
            i+=1
        else:
            resT2 = self._T2; eresT2 = 0
            resFillTeffs = self.fillTeffs
            eresFillTeffs = np.array([0, 0])

        ## Trigger for debugging
        if self.debugMode: embed()

        ## Now there is one specific case: What if we triggered autoLogg?
        if self.autoLogg:
            _teff_samples = ssamples[:, _teff_id]
            _mh_samples = ssamples[:, _mh_id]
            _logg_samples = self.compute_logg(_teff_samples, _mh_samples)
            _logg_mcmc = np.percentile(_logg_samples, [16, 50, 84])
            _q = np.diff(_logg_mcmc)
            resL = _logg_mcmc[1]
            eresL = np.mean(_q)

        ## Save a fits file containing the wavelength solution, best fit
        ## Input spectrum and error. Can then be used to recompute the values of 
        ## chi2 based on values of likelihood
        #
        fit = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, 
                    self.nan_mask, self.nwvls, self.grid_n, 
                    coeffs, resT, resL, resM, resA,
                    self.teffs, self.loggs, self.mhs, self.alphas, resvb,
                    resrv, resvsini, resvmac, resveil, resT2, resFillTeffs)
        coeffsnomag = coeffs*0
        coeffsnomag[0] = 1
        fitnomag = self.gen_spec(self.obs_wvl, self.obs_flux, self.obs_err, 
                self.nan_mask, self.nwvls, self.grid_n, 
                coeffsnomag, resT, resL, resM, resA,
                self.teffs, self.loggs, self.mhs, self.alphas, resvb,
                resrv, resvsini, resvmac, resveil, resT2, resFillTeffs)
        #
        hdu = fits.PrimaryHDU()
        hdu.header['OBJECT'] = (self.star, 'object observed')
        hdu.header['NORMFAC'] = (self.normFactor, 'object observed')
        hdu1 = fits.ImageHDU(data=self.obs_wvl, name='WVL')
        hdu2 = fits.ImageHDU(data=self.obs_flux, name='FLUX')
        hdu3 = fits.ImageHDU(data=self.obs_flux_tofit, name='FLUXFIT')
        hdu4 = fits.ImageHDU(data=self.obs_err, name='ERROR')
        hdu5 = fits.ImageHDU(data=fit, name='FIT')
        hdu6 = fits.ImageHDU(data=fitnomag, name='FITNOMAG')
        hdu7 = fits.ImageHDU(data=self.IDXTOFIT, name='IDXTOFIT')
        hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
        hdul.writeto(self.opath+'fit-data.fits', overwrite=True)

        ## Save the normalization factor to file:
        p = len(mcmcs) ## number of parameters
        new_normFactor = minchi2 * self.normFactor / (nbPointsFitted - p)
        self.save_normFactor(new_normFactor)

        strcoeffs = [str(coeffs[i]) for i in range(len(coeffs))]
        strecoeffs = [str(ecoeffs[i]) for i in range(len(ecoeffs))]
        resFillTeffsString = [str(resFillTeffs[i]) for i in range(len(resFillTeffs))]
        eresFillTeffsString = [str(eresFillTeffs[i]) for i in range(len(eresFillTeffs))]

        f = open(self.opath+'factors.txt', 'w')
        f.write(" ".join(strcoeffs) + " \n")
        f.write(" ".join(strecoeffs) + " \n")
        f.write("{} {} {} {} \n".format(resT, resL, resM, resA))
        f.write("{} {} {} {}\n".format(eresT, eresL, eresM, eresA))
        f.write("chi2 min: {:0.5f}\n".format(minchi2))
        f.write("chi2 min no field: {:0.5f}\n".format(minchi2exp))
        f.close()

        ## See output.txt for a description of the lines
        f = open(self.opath+'results_raw.txt', 'w')
        f.write(" ".join(strcoeffs) + " \n")
        f.write(" ".join(strecoeffs) + " \n")
        f.write("{} {} {} {} \n".format(resT, resL, resM, resA))
        f.write("{} {} {} {}\n".format(eresT, eresL, eresM, eresA))
        f.write("Mean. field: {} {} \n".format(meanfield, emeanfield))
        f.write("Av. field: {} {} \n".format(avfield, eavfield))
        f.write("vb: {} {}\n".format(resvb, eresvb))
        f.write("GussRV: {} {}\n".format(self.guessed_rv, 0))
        f.write("RV: {} {}\n".format(resrv, eresrv))
        f.write("vsini: {} {}\n".format(resvsini, eresvsini))
        f.write("vmac[{}]: {} {}\n".format(self.vmacMode, resvmac, eresvmac))
        f.write("chi2 min: {:0.5f}\n".format(minchi2))
        f.write("chi2 min no field: {:0.5f}\n".format(minchi2exp))
        f.write("Nb. of points: {}\n".format(nbPointsFitted))
        f.write("normFactor: {}\n".format(self.normFactor))
        f.write("veilingFac: {}\n".format(resveil).replace('[', '').replace(']', '').replace(',', ' '))
        f.write("e_veilingFac: {}\n".format(eresveil).replace('[', '').replace(']', '').replace(',', ' '))
        f.write("bolLum: {} {}\n".format(self.rL, self.drL))
        f.write("absMk: {} {}\n".format(self.Mk, self.dMk))
        f.write("dist: {} {}\n".format(self.dist, self.ddist))
        f.write("Multi-teff model Teff2: {} \n".format(resT2))
        f.write("Multi-teff model errTeff2: {} \n".format(eresT2))
        f.write("Multi-teff model fillTeffs: {} \n".format(" ".join(resFillTeffsString)))
        f.write("Multi-teff model errfillTeffs: {} \n".format(" ".join(eresFillTeffsString)))
        f.write("logCoeffs: {} \n".format(" ".format(self.logCoeffs)))
        f.write("Adjusted errors: {}\n".format(self.errorsAdj))
        f.write("Fit Derivative mode [T/F]: {}\n".format(self.fitDeriv))
        f.write("Error type: {}\n".format(self.errType))
        f.write("vinstru: {}\n".format(self.vinstru))
        f.close()

        print('ANALYSIS COMPLETE')

        return 0
