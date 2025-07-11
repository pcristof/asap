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
# from schwimmbad import MPIPool

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
import ultranest


parser = argparse.ArgumentParser()
parser.add_argument("star", type=str)
parser.add_argument("folderid", nargs='?', type=str, default=None)
parser.add_argument("-c", "--nbofcores", type=int, default=None)
parser.add_argument("-m", "--mpi", type=bool, default=False)
parser.add_argument("-p", "--profile", type=bool, default=False)
parser.add_argument("-d", "--dynesty", type=bool, default=False)
parser.add_argument("-u", "--run_ultranest", type=bool, default=False)

args = parser.parse_args()
ncores = args.nbofcores
dynesty = args.dynesty
star = args.star.strip()
if star[-5:] == '.fits':
    star = star[:-5]
folderid = args.folderid
mpi = args.mpi
profile = args.profile
run_ultranest = args.run_ultranest

# from IPython import embed
# embed()

#
# import config as config

locpath = os.getcwd()
config_file = locpath + '/config.ini'

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
opath = 'output_{}{}/'.format(star, folderid)
if not os.path.isdir(opath): os.mkdir(opath)
## Make a READ ONLY copy of the config file in the output folder
config_file_copy = opath+"config_copy.ini"
if os.path.isfile(config_file_copy):
    print('Caution, overwriting previous run config.ini')
    os.system("rm -f {}".format(config_file_copy)) ## Make the copy read only. This will help prevent future mistakes (e.g. oups, I modified the wrong file)
os.system("cp {} {}".format(config_file, config_file_copy))
os.chmod(config_file_copy, 0o444) ## Make the copy read only. This will help prevent future mistakes (e.g. oups, I modified the wrong file)

SA = SpectralAnalysis()
SA.set_opath(opath)
SA.set_star(star) ## Dummy variable to identify the star
# SA.simbad_grep()
SA.read_config(config_file_copy)

print('dynesty: {}'.format(dynesty))
print(SA.dynesty)
SA.set_dynesty(dynesty)
print(SA.dynesty)
print('CONFIG READ')

labels = SA.return_labels()

## Observation file
# infile = SA.pathtodata + "{}_templates.fits".format(star)
# if not os.path.isfile(infile):
#     infile = SA.pathtodata + "{}.fits".format(star)
infile = SA.pathtodata + "{}.fits".format(star)
if not os.path.isfile(infile):
    print(f"Template file {infile} not found")
    raise Exception('Template file not found')
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
if ncores is None: ## If not we keep what we passed
    ncores      = SA.ncores
# if not SA.parallel: ncores = 1

# SA.set_nwalkers(nwalkers)
# SA.set_nsteps(nsteps)
initial = SA.init_guess()
weights = SA.init_weights()
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
        ranges.append((0, 300))
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
    
    theta = np.zeros_like(u)
    for i in range(len(ranges)):
        theta[i] = ranges[i][0] + u[i] * (ranges[i][1] - ranges[i][0])

    theta[:idxStart] = theta[:idxStart] / (1-np.sum(theta[:idxStart])) ## Ensures the sum of ALL coeffs. 

    return theta

SA.return_warning_nanlikelidhood = False
def lnprob(par):
    if dynesty:
        lp = SA.lnprior(par)
    else:
        lp = 0
    return lp + SA.lnlike(par)
#
ndim = SA.ndim ## To avoid class call in MCMC


#####################################
#### ---- RUN MCMC ANALYSIS ---- ####
#####################################

import time
import multiprocessing
from multiprocessing import Pool
from multiprocessing import get_context
import emcee
import os

os.environ["OMP_NUM_THREADS"] = "1"
#multiprocessing.set_start_method("fork")

# Set up the backend
# Don't forget to clear it in case the file already exists
if SA.savebackend:
    # !! If the file exists, we may want to continue and not reset
    filename = opath + "backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    if os.path.isfile(filename):
        print('!!! By default, I continue the chain (no backend reset)')
        CONTINUE_BACKEND = True
        nwalkers = backend.shape[0] ## The number of walkers is that which was was set in the previous run
        # weights = None
    else:
        CONTINUE_BACKEND = False ## Cannot continue because no file found
        backend.reset(nwalkers, ndim) ## This resets the file
else:
    CONTINUE_BACKEND = False ## We are not saving the backend, and therefore not continuing
    backend=None

# import corner
if SA.parallel:
    print('Running in parallel mode')
    
    if sys.platform == "darwin": ## This should be a mac
        print('OS detected: MacOS')
        __p = get_context("fork").Pool(ncores)
    # elif mpi:
    #     __p = MPIPool(ncores)
    else:
        __p = Pool(ncores)

    with __p as pool:
        # if mpi:
        #     if not pool.is_master():
        #         pool.wait()
        #         sys.exit(0)
        if run_ultranest:
            print("Launching dynesty in Parallel")
            sampler = ultranest.ReactiveNestedSampler(labels, lnprob, prior_transform, pool=pool)   
        elif dynesty:
            print("Launching dynesty in Parallel")
            sampler = NestedSampler(lnprob, prior_transform, ndim, pool=pool, queue_size=ncores, nlive=nsteps)
        else:
            print("Launching emcee in Parallel")
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            pool=pool, 
                                            backend=backend) ## to save to file
        itime = time.time()
        if dynesty:
            sampler.run_nested()
            # res = sampler.results
            # from dynesty.utils import quantile as dyn_quantile
            # q = (0.025, 0.5, 0.975)
            # values = []
            # for i in range(res.samples.shape[1]):
            #     values.append(dyn_quantile(res.samples[:,i], q, weights=res.importance_weights()))
            # values = np.array(values)
            # max_likelihood_idx = np.argmax(res.logl)
            # best_fit_params = res.samples[max_likelihood_idx]
            # fig, axes = dyplot.cornerplot(res, show_titles=True,
            #                               truths=best_fit_params,
            #                             #   truth_color='black', ## Adds lines on the plots
            #                               )
            # fig.savefig(opath+"cornerplotnosmooth.png")
            # samples = res.samples
            # from IPython import embed
            # embed()
        else:
            if CONTINUE_BACKEND:
                sampler.run_mcmc(None, nsteps, progress=True)
            else:
                sampler.run_mcmc(weights, nsteps, progress=True)
                # import cProfile
                # cProfile.run('sampler.run_mcmc(weights, nsteps, progress=True)', sort='cumtime')
        etime = time.time()
else:
    print('Running in non parallel mode')
    # from IPython import embed
    # embed()
    if dynesty:
        sampler = NestedSampler(lnprob, prior_transform, ndim, nlive=nsteps)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        backend=backend) ## to save to file
    itime = time.time()
    if profile:
        import cProfile
        cProfile.run('sampler.run_mcmc(weights, nsteps, progress=True)', sort=True)
    else:
        if dynesty:
            print("Launching dynesty")
            sampler.run_nested()
        else:
            print("Launching emcee")
            if CONTINUE_BACKEND:
                sampler.run_mcmc(None, nsteps, progress=True)
            else:
                sampler.run_mcmc(weights, nsteps, progress=True)
    etime = time.time()
# Save to output
print("Time = {:.2f} seconds".format(etime - itime))
f = open(opath+'time.txt', 'w')
f.write("Initial guess: " + str(initial) + " \n")
f.write("Time = {:.2f} seconds\n".format(etime - itime))
f.close()

if dynesty:
    res = sampler.results
    log_prob_walkers_noflat = np.array([res.logl])
    np.save(opath+'log_prob_walkers_noflat.npy', log_prob_walkers_noflat)
    ## If dynesty, pickle the results object so that we can later load it
    ## and make the plot with dynesty.plotting.corerplot. 
    import pickle
    with open(opath+'dynesty_results.pkl', 'wb') as outp:
        pickle.dump(res, outp)
else:
    tau = sampler.get_autocorr_time(tol=0)
    log_prob_walkers_noflat = sampler.get_log_prob()
    np.save(opath+'tau.npy', tau)
    np.save(opath+'log_prob_walkers_noflat.npy', log_prob_walkers_noflat)
    print("Max autocorrelation time: {:0.2f}".format(np.max(tau)))
    f = open(opath+'time.txt', 'a')
    f.write("Max autocorrelation time: {:0.2f}\n".format(np.max(tau)))
    f.close()

from multiprocessing import cpu_count
ncpu = cpu_count()
f = open(opath+'time.txt', 'a')
f.write("{0} CPUs AVAILABLE\n".format(ncpu))
f.write("{0} CPUs USED\n".format(ncores))
f.close()
print("{0} CPUs AVAILABLE".format(ncpu))
print("{0} CPUs USED".format(ncores))

SA.sampler = sampler ## So that save_results can work
SA.save_results() ## Save the results and plots.

if SA.return_warning_nanlikelidhood:
    print('CAUTION: NaN likelihood !')

print('SCRIPT END')
exit()
