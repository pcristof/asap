
def main():
    import shutil
    import numpy as np
    import argparse
    from asap.SpectralAnalysis import SpectralAnalysis, read_res
    import os
    import matplotlib.pyplot as plt
    '''Script to re-plot the corners from the samples if available'''

    #### Scripting
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('opath', type=str, help='Path containing the emcee results', nargs='+')
    parser.add_argument('-p', '--plot', action='store_false', help='Prevents program from plotting')
    parser.add_argument('-t', '--latex', action='store_false', help='Removes LaTex formating')
    parser.add_argument('-b', '--burn', type=float, help='Burning fraction or number', default=0.5)
    parser.add_argument('-v', '--verbose', type=int, help='[int] Verbose level, 0,1,2, default 2', default=2)
    args = parser.parse_args()
    opathlist = args.opath
    verbose = args.verbose

    opath = opathlist[0]
    if opath[-1]!='/': opath+='/'

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

    samples_noflat_0 = np.load(opath + 'samples.npy') ## Not flattened
    log_prob_walkers_noflat_0 = np.load(opath + 'log_prob_walkers_noflat.npy')

    config_file = opath + 'config.ini'
    if not os.path.isfile(config_file):
        config_file = opath + 'config_copy.ini'
    resfile = opath + 'results_raw.txt'
    ## Read the configuration file
    SA = SpectralAnalysis()
    SA.read_config(config_file)
    ## Locate the results file
    results_data = read_res(resfile)

    ## Sometimes we run into problems with latex. Let's check if latex is usable:
    if shutil.which('latex'): latex = True

    ## Reasign        
    # samples_noflat_0 = samples
    data = {}
    data['nsteps'] = len(samples_noflat_0)
    data['burning'] = round(0.5*data['nsteps']) ## 50% by default
    data['bs'] = SA.bs

    #### REDISCARD - If user requested to discard the samples
    ## Recompute the burning period
    ## Take the samples after burning period
    samples_noflat = samples_noflat_0[data['burning']:]
    log_prob_walkers_noflat = log_prob_walkers_noflat_0[data['burning']:]

    #### Compute the number of fields in the fit
    nbOfFields = len(SA.bs) ## This is the number of fields in our model NOT WHAT WE FIT 
    
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

    labels = SA.return_labels()
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
    if SA.fitFields:
        ## If we are fitting fields, we are fitting nbOfFields-1 filling factors
        subssamples = ssamples.T[:nbOfFields-1]
        meanfield = np.sum(subssamples.T * data['bs'][1:], axis=1) ## only from magnetic coefficients
    
    ## Compute the first coeff and put it in place
    if SA.fitFields:
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

    cornerfont = 18
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

        plt.savefig(opath+'corner_2.pdf', bbox_inches='tight')
        plt.close()
        data['gen_files'].append('corner_2.pdf')


    ################################
    #### PLOT 2 - <B> HISTOGRAM ####
    ################################


    if plottrig:
        print("-> Generating <B> histogram")
        ## Now plot the B field only
        if SA.fitFields:
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
            res = SA.count_elements(np.round(meanfield, 3))
            maxres = SA.maxdir(res)
            #

            # 2 - get the maximum of likelihood for the distribution
            idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
            maxpos = meanfield[idx]
            #
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
            # meanfield_ssamples = np.sum(subssamples.T * SA.bs[1:], axis=1)
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
            plt.savefig(opath+'b_histogram_2.pdf')
            plt.close()
            data['gen_files'].append('b_histogram_2.pdf')


    ############################
    #### PLOT 3 - a0 -- <B> ####
    ############################


    if plottrig:
        print("-> Generating the a0 vs <B> plot")
        ## Now plot the B field and non mag component
        if SA.fitFields:
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

            plt.savefig(opath+'a0_b_2.pdf')
            plt.close()
            data['gen_files'].append('a0_b_2.pdf')


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
    
    max50 = nssamples[idx50]
    max = np.mean(max50, axis=0)
    for i in range(len(nssamples[0])):
        ## Compute the median and error bars "traditionally"
        mcmc = np.percentile(nssamples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        # 1 - get the maximum of the distributions
        roundfac = -1*magnitude(np.mean(q))
        if roundfac<0: roundfac=0
        res = SA.count_elements(np.round(nssamples[:, i], roundfac))
        maxdistrib = SA.maxdir(res)
        #
        # THISISATEST
        # idxsort = np.argsort(log_prob_walkers)
        # sortedsamples = nssamples.T[i][idxsort]
        # max50 = sortedsamples[-50:]
        # max50 = nssamples.T[i][idx50]
        # maxproba = np.array([np.mean(max50)])
        maxproba=np.array([max[i]])
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
    # if (SA.fitFields and (nbOfFields>1)):
    #     coeffs_tradi = mcmcs_tradi[:nbOfFields+1]
    #     coeffs_tradi[0] = 1 - np.sum(SA.coeffs[1:]) ## This apperrs to make a copy of the mcmcs array
    #     ecoeffs_tradi = emcmcs_tradi[:nbOfFields+1]
    #     #
    #     coeffs_maxproba = mcmcs_maxproba[:nbOfFields+1]
    #     ecoeffs_maxproba = emcmcs_maxproba[:nbOfFields+1]
    #     #
    #     coeffs_maxdistrib = mcmcs_maxdistrib[:nbOfFields+1]
    #     ecoeffs_maxdistrib = emcmcs_maxdistrib[:nbOfFields+1]
    # else:
    #     coeffs_tradi = np.zeros(len(SA.bs))
    #     coeffs_tradi[0] = 1
    #     ecoeffs_tradi = np.zeros(len(SA.bs))
    #     #
    #     coeffs_maxproba = np.zeros(len(SA.bs))
    #     coeffs_maxproba[0] = 1
    #     ecoeffs_maxproba = np.zeros(len(SA.bs))
    #     #
    #     coeffs_maxdistrib = np.zeros(len(SA.bs))
    #     coeffs_maxdistrib[0] = 1
    #     ecoeffs_maxdistrib = np.zeros(len(SA.bs))

    if (SA.fitFields and (nbOfFields>1)):
        subssamples = nssamples.T[1:nbOfFields] ## Without the 0kG component
        meanfield_ssamples = np.sum(subssamples.T * SA.bs[1:], axis=1)
        mcmc_meanfield = np.percentile(meanfield_ssamples, [16, 50, 84])
        q_meanfield = np.diff(mcmc_meanfield)
        meanfield_tradi = mcmc_meanfield[1]
        emeanfield_tradi = np.mean(q_meanfield)
        #
        # 1 - get the maximum of the distributions
        res = SA.count_elements(np.round(meanfield_ssamples, 3))
        maxres = SA.maxdir(res)
        #
        # 2 - get the maximum of likelihood for the distribution
        # idx = np.where(log_prob_walkers==np.max(log_prob_walkers))
        # maxpos = meanfield[idx]
        max50 = meanfield[idx50]
        maxpos = np.array([np.mean(max50)])
        emaxpos = (np.max(max50) - np.min(max50))/2
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
    if SA.fitFields:
        coeffs = np.array(mcmcs[0:nbOfFields]); ecoeffs = np.array(emcmcs[0:nbOfFields])
    else: ## No magnetic field fitted
        coeffs = np.zeros(nbOfFields)
        coeffs[0] = 1.
        ecoeffs = np.zeros(nbOfFields) 
    ##
    meanfield = np.array(maxproba_meanfield); emeanfield = np.array(emaxproba_meanfield)
    # Compute the average magnetic field
    avfield = np.sum(SA.bs * coeffs)
    eavfield = np.sqrt(np.sum((SA.bs*ecoeffs)**2))

    ############################
    #### PLOT 3 - a0 -- <B> ####
    ############################

    if plottrig:

        plt.close('all')

        if SA.fitFields:

            params= {'xtick.labelsize': 18,'ytick.labelsize': 18,'axes.labelsize': 20, 'legend.fontsize': 16,   'text.usetex': latex,'figure.figsize' : (6.4, 4.8)}
            plt.rcParams.update(params)

            xaxis = SA.bs
            width = np.median(np.diff(SA.bs))*0.95
            plt.bar(xaxis, coeffs, width=width, color='black')
            plt.ylabel('Filling factor')
            plt.xlabel('Field strength (kG)')
            # Extract the axes
            plt.tick_params(which='minor',direction='in',axis='both',bottom='on', top='on', left='on', right='on', length=5)
            plt.tick_params(which='major',direction='in',axis='both',bottom='on', top='on', left='on', right='on', length=10)
            plt.tight_layout()
            plt.savefig(opath+'b_distrib_2.pdf')
            plt.close()
            data['gen_files'].append('b_distrib_2.pdf')


    ##########################
    #### PLOT 4 - samples ####
    ##########################
    ## I did not reconstruct the zero-magnetic field for the non-flattened samples.
    ## So IF we fit the fields, we need to remove the first one.
    if SA.fitFields:
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
        plt.savefig(opath+'samples_2.pdf')
        # plt.show()
        plt.close()
        data['gen_files'].append('samples_2.pdf')

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
        plt.savefig(opath+'samples_postburn_2.pdf')
        # plt.show()
        plt.close()
        data['gen_files'].append('samples_postburn_2.pdf')

    resdict = SA.get_PARAMS(mcmcs, emcmcs)

    strcoeffs = [str(resdict['a'+str(i)]) for i in range(len(SA.bs))]
    strecoeffs = [str(resdict['e_a'+str(i)]) for i in range(len(SA.bs))]

    resveil = [resdict['r{}'.format(band)] for band in SA.veilingBands]
    eresveil = [resdict['e_r{}'.format(band)] for band in SA.veilingBands]
    resveil_tofit = [resdict['r{}'.format(band)] for band in SA.fitBands]
    resFillTeffs = [resdict['fillteff_0'], resdict['fillteff_1']]
    eresFillTeffs = [resdict['e_fillteff_0'], resdict['e_fillteff_1']]
    

    # ## Here coeffs include the 0kG component
    # fit = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
    #             SA.nan_mask, SA.nwvls, SA.grid_n, 
    #             coeffs, resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha'],
    #             SA.teffs, SA.loggs, SA.mhs, SA.alphas, resdict['vb'],
    #             resdict['rv'], resdict['vsini'], resdict['vmac'], resveil_tofit, resdict['teff2'], resFillTeffs)
    # ## Same, mcmc contains the 0kG component we do not want to feed to lnlike.
    # ## But if there are no magnetic fields, mcmcs will NOT contain the 0kG factor !
    # if SA.fitFields and (nbOfFields>1):
    #     mcmcsForLnLike = mcmcs[1:] ## Without magnetic field component
    # else:
    #     mcmcsForLnLike = mcmcs
    # _  = SA.lnlike(mcmcsForLnLike)
    # minchi2 = np.sum(SA._res)
    # coeffsnomag = coeffs*0
    # coeffsnomag[0] = 1
    # fitnomag = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
    #         SA.nan_mask, SA.nwvls, SA.grid_n, 
    #         coeffsnomag, resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha'],
    #         SA.teffs, SA.loggs, SA.mhs, SA.alphas, resdict['vb'],
    #         resdict['rv'], resdict['vsini'], resdict['vmac'], resveil_tofit, resdict['teff2'], resFillTeffs)
    # ## Here, we want the same as the results of the mcmcs, but the magnetic components should be set to zero.
    # if SA.fitFields and (nbOfFields>1):
    #     mcmcsForLnLike_0kG = np.copy(mcmcsForLnLike)
    #     mcmcsForLnLike_0kG[0] = 1
    #     mcmcsForLnLike_0kG[1:nbOfFields] = 0
    # else:
    #     mcmcsForLnLike_0kG = mcmcsForLnLike
    # _  = SA.lnlike(mcmcsForLnLike_0kG)
    # minchi2exp = np.sum(SA._res)
    # #
    # hdu = fits.PrimaryHDU()
    # hdu.header['OBJECT'] = (SA.star, 'object observed')
    # hdu.header['NORMFAC'] = (SA.normFactor, 'object observed')
    # hdu1 = fits.ImageHDU(data=SA.obs_wvl, name='WVL')
    # hdu2 = fits.ImageHDU(data=SA.obs_flux, name='FLUX')
    # hdu3 = fits.ImageHDU(data=SA.obs_flux_tofit, name='FLUXFIT')
    # hdu4 = fits.ImageHDU(data=SA.obs_err, name='ERROR')
    # hdu5 = fits.ImageHDU(data=fit, name='FIT')
    # hdu6 = fits.ImageHDU(data=fitnomag, name='FITNOMAG')
    # hdu7 = fits.ImageHDU(data=SA.IDXTOFIT, name='IDXTOFIT')
    # hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7])
    # hdul.writeto(opath+'fit-data.fits', overwrite=True)

    # ## Save the normalization factor to file:
    # nbPointsFitted = len(SA.obs_flux_tofit[SA.IDXTOFIT]) 

    # p = len(mcmcs) ## number of parameters
    # new_normFactor = 0. * SA.normFactor / (nbPointsFitted - p)
    # SA.save_normFactor(new_normFactor)

    strcoeffs = [str(coeffs[i]) for i in range(len(coeffs))]
    strecoeffs = [str(ecoeffs[i]) for i in range(len(ecoeffs))]
    resFillTeffsString = [str(resFillTeffs[i]) for i in range(len(resFillTeffs))]
    eresFillTeffsString = [str(eresFillTeffs[i]) for i in range(len(eresFillTeffs))]

    f = open(opath+'factors_2.txt', 'w')
    f.write(" ".join(strcoeffs) + " \n")
    f.write(" ".join(strecoeffs) + " \n")
    f.write("{} {} {} {} \n".format(resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha']))
    f.write("{} {} {} {}\n".format(resdict['e_teff'], resdict['e_logg'], resdict['e_mh'], resdict['e_alpha']))
    f.write("chi2 min: {:0.5f}\n".format(0.))
    f.write("chi2 min no field: {:0.5f}\n".format(0.))
    f.close()

    ## See output.txt for a description of the lines
    f = open(opath+'results_raw_2.txt', 'w')
    f.write(" ".join(strcoeffs) + " \n")
    f.write(" ".join(strecoeffs) + " \n")
    f.write("{} {} {} {} \n".format(resdict['teff'], resdict['logg'], resdict['mh'], resdict['alpha']))
    f.write("{} {} {} {}\n".format(resdict['e_teff'], resdict['e_logg'], resdict['e_mh'], resdict['e_alpha']))
    f.write("Mean. field: {} {} \n".format(meanfield, emeanfield))
    f.write("Av. field: {} {} \n".format(avfield, eavfield))
    f.write("vb: {} {}\n".format(resdict['vb'], resdict['e_vb']))
    f.write("GussRV: {} {}\n".format(SA.guessed_rv, 0))
    f.write("RV: {} {}\n".format(resdict['rv'], resdict['e_rv']))
    f.write("vsini: {} {}\n".format(resdict['vsini'], resdict['e_vsini']))
    f.write("vmac[{}]: {} {}\n".format(SA.vmacMode, resdict['vmac'], resdict['e_vmac']))
    f.write("chi2 min: {:0.5f}\n".format(0.))
    f.write("chi2 min no field: {:0.5f}\n".format(0.))
    f.write("Nb. of points: {}\n".format(0.))
    f.write("normFactor: {}\n".format(SA.normFactor))
    f.write("veilingFac: {}\n".format(resveil).replace('[', '').replace(']', '').replace(',', ' '))
    f.write("e_veilingFac: {}\n".format(eresveil).replace('[', '').replace(']', '').replace(',', ' '))
    f.write("bolLum: {} {}\n".format(SA.rL, SA.drL))
    f.write("absMk: {} {}\n".format(" ", " "))#.format(SA.Mk, SA.dMk))
    f.write("dist: {} {}\n".format(" ", " "))#.format(SA.dist, SA.ddist))
    f.write("------: {} \n".format(" "))#.format(resdict['teff2']))
    f.write("------: {} \n".format(" "))#.format(resdict['e_teff2']))
    f.write("------: {} \n".format(" "))#.format(" ".join(resFillTeffsString)))
    f.write("------: {} \n".format(" "))#.format(" ".join(eresFillTeffsString)))
    f.write("------: {} \n".format(" "))#.format(" ".format(SA.logCoeffs)))
    f.write("------: {}\n".format(" "))#.format(SA.errorsAdj))
    f.write("------: {}\n".format(" "))#.format(SA.fitDeriv))
    f.write("Error type: {}\n".format(SA.errType))
    f.write("vinstru: {}\n".format(SA.vinstru))
    f.close()

if __name__ == "__main__":
    main()