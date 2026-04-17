from distutils.command.config import config
# from asap import paths
import matplotlib.pyplot as plt
from asap.SpectralAnalysis import SpectralAnalysis
import numpy as np
from asap.spectral_analysis_pack import wrap_function_fine_linear_4d
from asap.spectral_analysis_pack import broaden_spectra
import sys
import os
import argparse ## To read optional arguments
import configparser
from asap import io_tools
# from schwimmbad import MPIPool

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
import ultranest
import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing import get_context
import emcee
from asap.sampler_utils import extract_emcee, extract_dynesty, extract_ultranest


parser = argparse.ArgumentParser()
parser.add_argument("star", nargs='?', type=str, default=None)
parser.add_argument("folderid", nargs='?', type=str, default=None)
# parser.add_argument("-e", "--extension", type=str, default=None)
parser.add_argument("-i", "--interactive", action='store_true', 
                    help='Turn on interactive mode allowing the user to see '\
                        +'files that can be read in the input directory.' )
parser.add_argument("-c", "--nbofcores", type=int, default=None)
parser.add_argument("-m", "--mpi", type=bool, default=False)
parser.add_argument("-p", "--profile", type=bool, default=False)
parser.add_argument("-d", "--dynesty", action='store_true', default=False,
                    help='Use dynesty nested sampling instead of emcee')
parser.add_argument("-u", "--run_ultranest", action='store_true', default=False,
                    help='Use UltraNest reactive nested sampling instead of emcee')
parser.add_argument("--nlive", type=int, default=None,
                    help='Override live points for nested sampling '
                         + '(config-first when omitted).')
## BUG: These variable have no effects on the MCMC run.
parser.add_argument("--nsteps", type=int, default=None,
                    help='Override step-sampler nsteps for UltraNest '
                         + '(config-first when omitted).')
parser.add_argument("--magfields", nargs='+', type=float, default=None,
                    help='Override magFields from config (space-separated kG '
                         + 'values, e.g. --magfields 0 2 4)')
parser.add_argument("--fillfactors", nargs='+', type=float, default=None,
                    help='Override fillFactors from config (space-separated '
                         + 'values summing to 1, e.g. '
                         + '--fillfactors 0.5 0.3 0.2)')
parser.add_argument("--student", action='store_true', default=False,
                    help='Use the student-t distribution and fit for the DOF')

args = parser.parse_args()
# plotfit = args.plotfit
locpath = os.getcwd()
config_file = locpath + '/config.ini'


def resolve_sampler_type(cli_args, config_path):
    """Resolve sampler with precedence: CLI flags > config.ini > default."""
    if cli_args.run_ultranest:
        return "ultranest"
    if cli_args.dynesty:
        return "dynesty"

    if not os.path.isfile(config_path):
        return "emcee"

    cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    cfg.read(config_path)
    if not cfg.has_option('MAIN', 'sampler'):
        return "emcee"

    sampler_from_config = cfg['MAIN']['sampler'].strip().lower()
    allowed_samplers = ('emcee', 'dynesty', 'ultranest')
    if sampler_from_config not in allowed_samplers:
        raise ValueError(
            "Invalid [MAIN] sampler='{}'. Expected one of {}.".format(
                sampler_from_config, allowed_samplers)
        )
    return sampler_from_config


# Determine sampler type from CLI/config.
sampler_type = resolve_sampler_type(args, config_file)
# Keep dynesty default behavior unchanged if no explicit live-point override is set.
nlive = args.nlive if args.nlive is not None else 400

# ---------------------------------------------------------------------------
# Early MPI detection — needed before any file I/O so that only rank 0
# creates directories, copies config files, etc.  Worker ranks wait at
# a barrier and then read the config that rank 0 wrote.
# ---------------------------------------------------------------------------
mpi_size = 1
mpi_rank = 0
if sampler_type == "ultranest":
    try:
        from mpi4py import MPI
        mpi_size = MPI.COMM_WORLD.Get_size()
        mpi_rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        pass

if args.star is not None:
    star = args.star.strip()
    if star[-5:] == '.fits':
        star = star[:-5]
else:
    star = None
folderid = args.folderid
mpi = args.mpi
profile = args.profile

# from IPython import embed
# embed()

#
# import config as config
infile = None

if star is None:
    if args.interactive:
        SA = SpectralAnalysis()
        SA.read_config(config_file)
        ## Grab the input directory
        _pathtodata = SA.pathtodata
        fname = io_tools.interactive_list_file(_pathtodata)
        if fname is None:
            raise Exception('No observation file found')
        star = fname.split('/')[-1].replace('_templates.fits', '')
        star = star.replace('.fits', '')
        infile = fname
    else:
        raise Exception('No observation file provided. Try running with -i.')

###############################
#### ---- USER INPUTS ---- ####
###############################
## Input star
# star = sys.argv[1].lower().strip()
## Folder ID - optional identifier added at the end of the folder name.
# if len(sys.argv)>2: 
    # folderid = sys.argv[2].strip()
if folderid is not None:
    folderid = '_'+folderid
else : 
    folderid = ""
## Mode switch
expmode = False ## Do you want to use the log of the components?
smooth = False ## Attempt to smooth the surface to avoid numerical noise
            ## The smoothing is performed by rounding the likelihood value
## Output folder
if '/' in star:
    _star = star.replace('/', '_')
else:
    _star = star
opath = 'output_{}{}/'.format(_star, folderid)
ultranest_logdir = os.path.join(opath, 'ultranest_logdir')
config_file_copy = opath+"config_copy.ini"

# Only rank 0 creates the output directory and copies the config file.
# Under MPI every rank runs this script; without the guard the mkdir and
# file-copy race against each other and crash.
if mpi_rank == 0:
    os.makedirs(opath, exist_ok=True)
    if sampler_type == "ultranest":
        # Always keep UltraNest logs inside the retrieval output folder.
        os.makedirs(ultranest_logdir, exist_ok=True)
    if os.path.isfile(config_file_copy):
        print('Caution, overwriting previous run config.ini')
        os.system("rm -f {}".format(config_file_copy))
    os.system("cp {} {}".format(config_file, config_file_copy))
    os.chmod(config_file_copy, 0o444)

# Worker ranks wait until rank 0 has written the config copy.
if mpi_size > 1:
    MPI.COMM_WORLD.Barrier()

SA = SpectralAnalysis()
SA.set_opath(opath)
SA.set_star(star) ## Dummy variable to identify the star
# SA.simbad_grep()
SA.read_config(config_file_copy)

SA.set_student(args.student) ## Must happen before SA.init_PARAMS

## Override magFields and/or fillFactors from CLI if provided
if args.magfields is not None or args.fillfactors is not None:
    ## Validate consistency between magfields and fillfactors
    n_bs = len(args.magfields) if args.magfields is not None else len(SA.bs)
    n_ff = len(args.fillfactors) if args.fillfactors is not None else len(SA.fillFactors)
    if n_bs != n_ff:
        raise ValueError(
            f'Mismatch: {n_bs} magnetic field component(s) but {n_ff} filling factor(s). '
            f'These must have the same length.'
        )
    ## Update the in-memory SA object on ALL ranks
    if args.magfields is not None:
        SA.update_bs(np.array(args.magfields))
        if mpi_rank == 0:
            print(f'CLI override: magFields set to {args.magfields}')
    if args.fillfactors is not None:
        SA.update_fillFactors(np.array(args.fillfactors))
        if mpi_rank == 0:
            print(f'CLI override: fillFactors set to {args.fillfactors}')
    SA.init_PARAMS()

## Keep config copy in sync with CLI overrides for reproducibility
if mpi_rank == 0:
    should_update_config_copy = (
        args.magfields is not None
        or args.fillfactors is not None
        or args.nlive is not None
        or args.nsteps is not None
        or args.run_ultranest
        or args.dynesty
    )
    if should_update_config_copy:
        os.chmod(config_file_copy, 0o644)
        _cfg = configparser.ConfigParser()
        _cfg.read(config_file_copy)
        if not _cfg.has_section('MAIN'):
            _cfg.add_section('MAIN')
        if args.magfields is not None:
            _cfg['MAIN']['magFields'] = ' '.join(str(v) for v in args.magfields)
        if args.fillfactors is not None:
            _cfg['MAIN']['fillFactors'] = ' '.join(str(v) for v in args.fillfactors)
        _cfg['MAIN']['sampler'] = sampler_type

        if not _cfg.has_section('ULTRANEST'):
            _cfg.add_section('ULTRANEST')
        if args.nlive is not None:
            _cfg['ULTRANEST']['min_num_live_points'] = str(args.nlive)
        if args.nsteps is not None:
            _cfg['ULTRANEST']['nsteps'] = str(args.nsteps)

        with open(config_file_copy, 'w') as _f:
            _cfg.write(_f)
        os.chmod(config_file_copy, 0o444)

if mpi_rank == 0:
    print('Sampler type: {}'.format(sampler_type))
SA.sampler_type = sampler_type
if mpi_rank == 0:
    print('CONFIG READ')

## Update the sampling method in the object to keep track of it
SA.set_samplerType(sampler_type.upper())


labels = SA.return_labels()

## Observation file
# infile = SA.pathtodata + "{}_templates.fits".format(star)
# if not os.path.isfile(infile):
#     infile = SA.pathtodata + "{}.fits".format(star)
if infile is None:
    infile = SA.pathtodata + "{}.fits".format(star)
    infile2 = SA.pathtodata + "{}_templates.fits".format(star)
    infile3 = SA.pathtodata + "{}_template.fits".format(star)
    fileFound = False
    if os.path.isfile(infile):
        print(f"File found: {infile}")
        fileFound = True
    if os.path.isfile(infile2):
        print(f"File found: {infile2} -- using this one")
        infile3 = infile2
        infile = infile2
        infile2 = infile3
        fileFound = True
    if not fileFound:
        if args.interactive:
            fname = io_tools.interactive_list_file(SA.pathtodata)
            if fname is None:
                raise Exception('No observation file found')
            star = fname.split('/')[-1].replace('_templates.fits', '')
            star = star.replace('.fits', '')
            infile = fname
            SA.set_star(star)
        else:
            raise Exception(f'Template file {infile} or {infile2} not found')
## Regions file
region_file = SA.linelist
#
# SA.update_teffs(np.arange(4500, 5400, 100)) ## don't have 3900 on laptop
# SA.update_teffs(np.arange(3600, 4500, 100)) ## don't have 3900 on laptop
# SA.update_teffs(np.arange(3000, 3500, 100))
# SA.update_loggs(np.arange(4.0, 6.0, 0.5))
#SA.update_mhs(np.arange(-1.0, 1.0, 0.5))
# SA.update_mhs(np.arange(-0.75, 1.0, 0.25))
# SA.update_teffs(np.arange(3700, 4000, 100)) ## don't have 3900 on laptop
# SA.update_loggs(np.arange(4.0, 5.5, 0.5)) ## don't have 3900 on laptop
# SA.update_mhs(np.arange(-0.5, 1.0, 0.5)) ## don't have 3900 on laptop
# SA.update_loggs(np.arange(4.0, 6.0, 0.5)) ## don't have 3900 on laptop

# The following is for fast debugging
# SA.update_teffs(np.arange(3200, 3400, 100))
# SA.update_loggs(np.arange(5.0, 6.0, 0.5))
# SA.update_mhs(np.arange(-0.50, 0.0, 0.25))
##############################################
#### ---- LOAD OBS, REGIONS AND GRID ---- ####
##############################################
#
print('---- Loading observations ----')
med_wvl, med_spectrum, med_err, berv = SA.load_obs(infile)

print('done loading observation')
print('------------------------------')
obs_wvl, obs_flux, obs_err, nan_mask, regions = SA.create_regions(
                                                    region_file, med_wvl,
                                                    med_spectrum, med_err, 
                                                    berv)

# SA.replace_observation_from_file()

nwvls, grid_n, teffs, loggs, mhs, alphas = SA.load_grid(SA.pathtogrid, regions)
print('Done loading grid')

# # from IPython import embed
# # embed()
# # Are the spectra here the same as the spectra in the stored directory?
# # Load the file:
# specfile = '/Users/pcristofari/Data/zeeturbo-grids/spectra-zeeturbo-v2/hdf5-spirou-highteff/4750g3.5z-0.25a0.00b0000p0.0rot90.00beta0.00.hdf5'
# import h5py
# with h5py.File(specfile, 'r') as h5f:
#     if 'wavelink' in h5f.keys():
#         w = h5f['wavelink']['wave'][()]
#     else:
#         w = h5f['wave'][()]
#     from irap_tools import analysis_tools as tls
#     w = tls.convert_lambda_in_vacuum(w)
#     s = h5f['norm_flux'][()]
# # plt.figure()
# # plt.plot(nwvls.T, grid_n[0, 0, 0, 0, 0].T)
# # plt.plot(w.T, s.T)
# # plt.show()

# _T = 4750; _L=3.5; _M=-0.25; _A=0.00
# vb = 0
# SA.vinstru=0.1

# fit = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
#                     SA.nan_mask, SA.nwvls, SA.grid_n, 
#                     SA.coeffs, _T, _L, _M, _A,
#                     SA.teffs, SA.loggs, SA.mhs, SA.alphas, vb,
#                     SA.rv, SA.vsini, SA.vmac, SA.veilingFacToFit, 
#                     SA._T2, SA.fillTeffs)


# ## --------------------
# ## DEBUGGING
# from IPython import embed
# embed()
# # specfile = '/Users/pcristofari/Data/zeeturbo-grids/spectra-zeeturbo-v2/hdf5-spirou-highteff/4750g3.5z-0.25a0.00b0000p0.0rot90.00beta0.00.hdf5'
# # import h5py
# # # from irap_tools import analysis_tools as tls
# # with h5py.File(specfile, 'r') as h5f:
# #     if 'wavelink' in h5f.keys():
# #         w = h5f['wavelink']['wave'][()]
# #     else:
# #         w = h5f['wave'][()]
# #     w = tls.convert_lambda_in_vacuum(w)
# #     s = h5f['norm_flux'][()]
# plt.figure()
# plt.plot(nwvls.T, grid_n[0, 0, 0, 0, 0].T, color='black')
# plt.plot(nwvls.T, grid_n[0, 0, 0, 1, 0].T, color='cyan')
# plt.plot(SA.obs_wvl.T, fit.T, '--', color='red')
# plt.show()

# ## ----------------------


# plt.figure()
# plt.plot(nwvls.T, grid_n[0, 0, 0, 0, 0].T, color='black')
# plt.plot(SA.obs_wvl.T, fit.T, color='red')
# plt.show()

## This is debugging
# SA.adjust_errors() ## This will adjust the error bars based on the results file

## Create a mask to avoid pixels know to contain systematics.
# SA.exclude_pixels()

# from IPython import embed
# embed()
# regions = SA.mask_small_lines(0.1)
# SA.pre_norm_obs() ## We renormalize the spectra
# SA.pre_norm_grid() ## We renormalize the spectra
# SA.set_adjcont(False)

# plt.figure()
# plt.plot(nwvls.T, grid_n[0,0,0,1,1].T, color='black')
# plt.plot(SA.nwvls.T, SA.grid_n[0,0,0,1,1].T, color='red')
# plt.show()

# from IPython import embed
# embed()

## SA.get_grid was used for debugging -> Reproduces what was done in the 
## chi2 minimization code. Does not support magnetic fields, does not read
## zTurbo.
# nwvls, grid_n, teffs, loggs, mhs, alphas = SA.get_grid()

##############################################################################
##############################################################################

## We are now ready to perform the MCMC -- VERSION 5
## In this version we fit everything but adapt to the user input
## We can therefore run the program with no magnetic field by simply adapting
## the magnetic field array to an array containing only the value 0.
## In version 5 we add the possibility to fit RV. We also rely more on the
## Spectral analysis object, and remove re-definitions of functions as we
## find them to not improve the effeciency of the code (see below).

##########################################
#### ---- INTIIALIZATION AND RUN ---- ####
##########################################
#
nwalkers    = SA.nwalkers
nsteps      = SA.nsteps
# if ncores is None: ## If not we keep what we passed
if args.nbofcores is not None:
    ncores      = SA.set_ncores(args.nbofcores)
# if not SA.parallel: ncores = 1

ncores = SA.ncores

# SA.set_nwalkers(nwalkers)
# SA.set_nsteps(nsteps)
initial = SA.init_guess() ## returns the initial guess of parameters
weights = SA.init_weights() ## returns the initial position of the walkers
if SA.renorm:
    SA.compute_normFactor(SA.normFactor) ## This will bypass apply a normalization factor to
                            ## the lnlike function
# SA.set_ncores(ncores) ## Number of cores to use for the MCMC
# SA.mcmc()

#################################
#### ---- MCMC ANALYSIS ---- ####
#################################
## It may seem extreamely dumb, but I was unable to find a way include the
## MCMC in the object class without breaking speed. It would appear to come
## from pool, which re-pickles the object every time it need to read 
## something. Ergo, passing a function from a class of from self breaks the
## speed. To be most efficient, I therefore load everything I computed so far
## in variables, and redefine the functions, including gen_spec so that there
## is no call whatsoever to the class at run time.
## UPDATE: Code is just as fast as long as the function lnprob is not that of
## the object. So we remove the re-definitions to gen_spec, unpackpar, lnlike 
## and lnprior. These are defined in the object.

###############################################
#### ---- FUNCTIONS FOR MCMC ANALYSIS ---- ####
###############################################
#
def define_ranges():
    nbOfFields = len(SA.bs) ## This will helps us unpack par
    ranges = [] ## Those are the ranges for priors
    #
    idxStart = 0
    if SA.fitFields:
        idxStart = nbOfFields-1
        for i in range(idxStart):
            ranges.append((0, 1))
    ## Grab the T, L, M, A
    i = idxStart
    if SA.fitTeff:
        ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
        i += 1
    if SA.fitLogg:
        ranges.append((SA.loggs[0], SA.loggs[-1]))
        i += 1
    if SA.fitMh:
        ranges.append((SA.mhs[0], SA.mhs[-1]))
        i += 1
    if SA.fitAlpha:
        ranges.append((SA.alphas[0], SA.alphas[-1]))
        i += 1
    ## Loop through the parameters
    if SA.fitbroad:
        ranges.append((0, 300))
        i += 1
    if SA.fitrv:
        ranges.append((-20, 20))
        i+=1
    if SA.fitrot:
        ranges.append((0, 300))
        i+=1
    if SA.fitmac:
        ranges.append((0, 10))
        i+=1
    if SA.fitVeiling:
        for j in range(SA.nbFitVeil):
            ranges.append((0, 10))
        i+=1+SA.nbFitVeil
    if SA.fitTeff2: ## Second temperature
        ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
        i += 1
        ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
        i += 1
    return ranges, idxStart

ranges, idxStart = define_ranges()

def prior_transform(u):
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
    # nbOfFields = len(SA.bs) ## This will helps us unpack par
    # ranges = [] ## Those are the ranges for priors
    # #
    # idxStart = 0
    # if SA.fitFields:
    #     idxStart = nbOfFields-1
    #     for i in range(idxStart):
    #         ranges.append((0, 1))
    # ## Grab the T, L, M, A
    # i = idxStart
    # if SA.fitTeff:
    #     ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
    #     i += 1
    # if SA.fitLogg:
    #     ranges.append((SA.loggs[0], SA.loggs[-1]))
    #     i += 1
    # if SA.fitMh:
    #     ranges.append((SA.mhs[0], SA.mhs[-1]))
    #     i += 1
    # if SA.fitAlpha:
    #     ranges.append((SA.alphas[0], SA.alphas[-1]))
    #     i += 1
    # ## Loop through the parameters
    # if SA.fitbroad:
    #     ranges.append((0, 300))
    #     i += 1
    # if SA.fitrv:
    #     ranges.append((-20, 20))
    #     i+=1
    # if SA.fitrot:
    #     ranges.append((0, 300))
    #     i+=1
    # if SA.fitmac:
    #     ranges.append((0, 300))
    #     i+=1        
    # if SA.fitVeiling:
    #     for j in range(SA.nbFitVeil):
    #         ranges.append((0, 10))
    #     i+=1+SA.nbFitVeil
    # if SA.fitTeff2: ## Second temperature
    #     ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
    #     i += 1
    #     ranges.append((SA.teffs[0]-200, SA.teffs[-1]))
    #     i += 1
    
    theta = np.zeros_like(u)
    for i in range(idxStart, len(ranges)):
        theta[i] = ranges[i][0] + u[i] * (ranges[i][1] - ranges[i][0])

    # Magnetic filling factors: symmetric Dirichlet(1,...,1) prior over ALL
    # nbOfFields components (including the zero-field component, which is
    # derived as 1 - sum(free)).  Only u[:idxStart] are consumed here;
    # u[idxStart] onward belong to the atmospheric/broadening parameters.
    #
    # We draw (idxStart+1) Gamma(1,1) variates from only idxStart cube dims
    # by using a fixed variate (1.0 = the mean of Exp(1)) for the zero-field
    # component.  This gives a symmetric Dirichlet draw that sums to 1.
    ## PIC: There is something funky here. If you want to draw from a dirichlet
    ## you can't quite fix the first coeff to 1... (I think).
    ## It's still not quite clear to me whether we should just draw N random
    ## values to deduce the first, or if we should pass N+1 values to
    ## cleanly draw from a dirichlet distribution.
    if idxStart > 0:
        v = np.zeros(idxStart+1)
        v[1:] = u[:idxStart] ## All the coefficients we passed
        v[0] = (idxStart - np.sum(u[:idxStart]))/idxStart
        gamma_all = -np.log(np.clip(v, 1e-300, None))  # Exp(1) variates
        gamma_free = gamma_all[1:]/gamma_all.sum()
        theta[:idxStart] = gamma_free
        # gamma_zero = 1.0  # fixed variate for the zero-field component
        # gamma_sum = gamma_free.sum() + gamma_zero
        # theta[:idxStart] = gamma_free / gamma_sum   # free fractions; zero-field = gamma_zero / gamma_sum

    return theta

SA.return_warning_nanlikelidhood = False
def lnprob(par):
    # For nested samplers the prior is encoded in prior_transform;
    # lnprior is only needed by emcee.
    if sampler_type not in ("dynesty", "ultranest"):
        lp = SA.lnprior(par)
        if not np.isfinite(lp):
            return -np.inf
    else:
        lp = 0
    try:
        like = SA.lnlike(par)
    except ValueError as e:
        if "could not broadcast" in str(e) or "Shape mismatch" in str(e):
            if SA.debugMode:
                print(f"\n[WARNING] Array shape mismatch (likely valid during ultranest init):")
                print(f"Parameters: {SA.PARAMS_FIT}")
                print(f"Values: {par}")
                print(f"Error: {e}")
            # Use finite penalty for nested samplers (-inf creates plateaus
            # UltraNest cannot traverse); emcee expects -inf for rejection.
            return -1e100 if sampler_type in ("dynesty", "ultranest") else -np.inf
        else:
            raise
    if not np.isfinite(like):
        return -1e100 if sampler_type in ("dynesty", "ultranest") else -np.inf
    return lp + like
#
ndim = SA.ndim ## To avoid class call in MCMC

# Vectorized wrappers for UltraNest's vectorized=True mode.
# These receive (N, ndim) arrays and return (N,) / (N, ndim) arrays.
def lnprob_vectorized(params_batch):
    return np.array([lnprob(par) for par in params_batch])

def prior_transform_vectorized(u_batch):
    return np.array([prior_transform(u) for u in u_batch])


#####################################
#### ---- RUN MCMC ANALYSIS ---- ####
#####################################

os.environ["OMP_NUM_THREADS"] = "1"

# MPI detection was moved earlier (after argparse) so that file I/O
# in the initialisation phase can be guarded by rank == 0.
if mpi_size > 1:
    print(f'MPI detected: rank {mpi_rank} of {mpi_size}')

# Set up the emcee backend (only needed for emcee)
CONTINUE_BACKEND = False
backend = None
if sampler_type == "emcee" and SA.savebackend:
    filename = opath + "backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    if os.path.isfile(filename):
        print('!!! By default, I continue the chain (no backend reset)')
        CONTINUE_BACKEND = True
        nwalkers = backend.shape[0]
    else:
        CONTINUE_BACKEND = False
        backend.reset(nwalkers, ndim)

# ---------------------------------------------------------------------------
# UltraNest parallelisation guide:
#
#   UltraNest does NOT use Python multiprocessing pools.  Instead it relies
#   on MPI for distributing likelihood evaluations across cores/nodes.
#
#   Single-core run:
#       python -m asap <star> -u
#
#   Multi-core run (MPI, recommended):
#       mpiexec -n <ncores> python -m asap <star> -u
#
#   Make sure OMP_NUM_THREADS=1 (set above) to prevent numpy/BLAS from
#   spawning threads that compete with MPI ranks.
#
#   Do NOT combine --parallel / multiprocessing with UltraNest MPI; they are
#   mutually exclusive parallelisation strategies.
# ---------------------------------------------------------------------------

# Decide whether to use a multiprocessing pool.
# UltraNest handles its own parallelism via MPI, so skip the pool for it.
use_pool = SA.parallel and sampler_type not in ("ultranest",) and mpi_size == 1

if sampler_type == "ultranest":
    # --- UltraNest path (no multiprocessing pool) ---
    if args.nlive is not None:
        nlive = args.nlive
    elif SA.ultranest_min_num_live_points is not None:
        nlive = SA.ultranest_min_num_live_points
    else:
        nlive = 400

    nsteps_slice = args.nsteps if args.nsteps is not None else SA.ultranest_nsteps
    dlogz = SA.ultranest_dlogz
    if dlogz is None:
        dlogz = 0.5 + 0.1 * ndim
    update_interval_volume_fraction = SA.ultranest_update_interval_volume_fraction
    if update_interval_volume_fraction is None:
        update_interval_volume_fraction = 0.4 if ndim > 20 else 0.2

    print("Launching UltraNest (ndim={})".format(ndim))
    if mpi_size > 1:
        print(f"  MPI parallelisation active: {mpi_size} ranks")
    else:
        print("  Running single-core. For parallel execution use:")
        print("    mpiexec -n {} python -m asap {} -u --nlive {}".format(
            ncores, star, nlive))
    print("  UltraNest settings: nlive={}, dlogz={}, min_ess={}, resume={}, vectorized={}".format(
        nlive, dlogz, SA.ultranest_min_ess, SA.ultranest_resume, SA.ultranest_vectorized))
    print(f"  UltraNest logs: {ultranest_logdir}")

    resume_policy = SA.ultranest_resume
    sampler = ultranest.ReactiveNestedSampler(
        labels, lnprob_vectorized, prior_transform_vectorized,
        log_dir=ultranest_logdir,
        resume=resume_policy,
        vectorized=SA.ultranest_vectorized,
    )

    # Configure optional step sampler using config-first settings.
    step_sampler_mode = SA.ultranest_step_sampler
    if step_sampler_mode == 'auto':
        enable_step_sampler = (nsteps_slice is not None) or (ndim > 10)
        requested_step_sampler = 'population' if mpi_size > 1 else 'slice'
    else:
        enable_step_sampler = step_sampler_mode != 'none'
        requested_step_sampler = step_sampler_mode

    if enable_step_sampler:
        if nsteps_slice is None:
            nsteps_slice = 2 * ndim

        if requested_step_sampler == 'population' and mpi_size > 1:
            import ultranest.popstepsampler
            popsize = mpi_size
            print(f"  Using PopulationSliceSampler (popsize={popsize}, nsteps={nsteps_slice})")
            sampler.stepsampler = ultranest.popstepsampler.PopulationSliceSampler(
                popsize=popsize,
                nsteps=nsteps_slice,
                generate_direction=ultranest.popstepsampler.generate_region_oriented_direction,
            )
        else:
            import ultranest.stepsampler
            if requested_step_sampler == 'population' and mpi_size == 1:
                print("  Requested population step sampler without MPI; falling back to SliceSampler")
            print(f"  Using SliceSampler step sampler (nsteps={nsteps_slice})")
            sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=nsteps_slice,
                generate_direction=ultranest.stepsampler.generate_region_oriented_direction,
            )

    itime = time.time()
    result_raw = sampler.run(
        min_num_live_points=nlive,
        dlogz=dlogz,
        min_ess=SA.ultranest_min_ess,
        max_num_improvement_loops=SA.ultranest_max_num_improvement_loops,
        update_interval_volume_fraction=update_interval_volume_fraction,
    )
    etime = time.time()

    # Worker ranks must exit after UltraNest completes — only rank 0
    # should do post-processing (saving results, plotting, etc.).
    # Without this guard every MPI rank writes to the same output files,
    # causing race conditions and potential file corruption.
    if mpi_size > 1 and mpi_rank != 0:
        MPI.Finalize()
        sys.exit(0)

elif use_pool:
    print(f'Running in parallel mode with ncores={ncores}')

    if sys.platform == "darwin":
        print('OS detected: MacOS')
        __p = get_context("fork").Pool(ncores)
    else:
        __p = Pool(ncores)

    with __p as pool:
        # --- Phase 1: Create sampler ---
        if sampler_type == "dynesty":
            print("Launching dynesty in Parallel")
            sampler = NestedSampler(
                lnprob, prior_transform, ndim,
                pool=pool, queue_size=ncores, nlive=nlive,
            )
        else:
            print("Launching emcee in Parallel")
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
                pool=pool, backend=backend,
            )

        # --- Phase 2: Run sampler ---
        itime = time.time()
        if sampler_type == "dynesty":
            sampler.run_nested()
        else:
            if CONTINUE_BACKEND:
                sampler.run_mcmc(None, nsteps, progress=True)
            else:
                sampler.run_mcmc(weights, nsteps, progress=True)
        etime = time.time()

else:
    print('Running in non parallel mode')

    # --- Phase 1: Create sampler ---
    if sampler_type == "dynesty":
        sampler = NestedSampler(lnprob, prior_transform, ndim, nlive=nlive)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob, backend=backend,
        )

    # --- Phase 2: Run sampler ---
    itime = time.time()
    if profile and sampler_type == "emcee":
        import cProfile
        if CONTINUE_BACKEND:
            cProfile.run('sampler.run_mcmc(None, nsteps, progress=True)', sort=True)
        else:
            cProfile.run('sampler.run_mcmc(weights, nsteps, progress=True)', sort=True)
    else:
        if sampler_type == "dynesty":
            print("Launching dynesty")
            sampler.run_nested()
        else:
            print("Launching emcee")
            if CONTINUE_BACKEND:
                sampler.run_mcmc(None, nsteps, progress=True)
            else:
                sampler.run_mcmc(weights, nsteps, progress=True)
    etime = time.time()

# --- Phase 3: Extract results ---
print("Time = {:.2f} seconds".format(etime - itime))
SA.runTime = etime - itime
f = open(opath+'time.txt', 'w')
f.write("Initial guess: " + str(initial) + " \n")
f.write("Time = {:.2f} seconds\n".format(etime - itime))
f.close()

if sampler_type == "ultranest":
    sampler_result = extract_ultranest(result_raw)
    import pickle
    with open(opath + 'ultranest_results.pkl', 'wb') as outp:
        pickle.dump(result_raw, outp)
elif sampler_type == "dynesty":
    sampler_result = extract_dynesty(sampler)
    import pickle
    with open(opath + 'dynesty_results.pkl', 'wb') as outp:
        pickle.dump(sampler.results, outp)
else:
    sampler_result = extract_emcee(sampler, burn_frac=0.5)
    np.save(opath + 'tau.npy', sampler_result.metadata['tau'])

np.save(opath + 'log_prob_walkers_noflat.npy', sampler_result.log_likelihood)

if sampler_type == "emcee":
    print("Max autocorrelation time: {:0.2f}".format(
        np.max(sampler_result.metadata['tau'])))
    f = open(opath+'time.txt', 'a')
    f.write("Max autocorrelation time: {:0.2f}\n".format(
        np.max(sampler_result.metadata['tau'])))
    f.close()

if sampler_type in ("dynesty", "ultranest"):
    print("ln(Z) = {:.2f} +/- {:.2f}".format(
        sampler_result.evidence, sampler_result.evidence_err))
    f = open(opath+'time.txt', 'a')
    f.write("ln(Z) = {:.2f} +/- {:.2f}\n".format(
        sampler_result.evidence, sampler_result.evidence_err))
    f.close()

from multiprocessing import cpu_count
ncpu = cpu_count()
f = open(opath+'time.txt', 'a')
f.write("{0} CPUs AVAILABLE\n".format(ncpu))
f.write("{0} CPUs USED\n".format(ncores))
f.close()
print("{0} CPUs AVAILABLE".format(ncpu))
print("{0} CPUs USED".format(ncores))

SA.sampler_result = sampler_result
# SA.plotfit = plotfit
SA.save_results()

if SA.return_warning_nanlikelidhood:
    print('CAUTION: NaN likelihood !')

print('SCRIPT END')
exit()
