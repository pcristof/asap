def main():
    '''This part of the program aims at using exactly the tools of ASAP, to make a figure with buttons,
    so that one can interactively fit the spectrum.'''

    from asap.SpectralAnalysis import SpectralAnalysis
    import numpy as np

    import os
    import argparse ## To read optional arguments
    from numpy.linalg import svd
    import numpy as np
    import matplotlib.pyplot as plt
    from asap.SpectralAnalysis import read_res, read_res_v2

    parser = argparse.ArgumentParser()
    parser.add_argument("indir", type=str, help="Input grid directory")
    parser.add_argument("-o", "--output", type=str, default='./pca_grid/')
    parser.add_argument("-t", "--tol", type=float, default=0.001)
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("-f", "--fit", action='store_true')
    # parser.add_argument("-m", "--mpi", type=bool, default=False)
    # parser.add_argument("-p", "--profile", type=bool, default=False)
    # parser.add_argument("-d", "--dynesty", type=bool, default=False)
    parser.add_argument("-w", "--overwrite", action='store_true')

    args = parser.parse_args()

    ###############################################################
    ###############################################################

    ## Go to the indir directory:
    # os.chdir(args.indir)
    if (args.indir=='.') | (args.indir=='./'):
        args.indir = os.getcwd()
    if args.indir[-1]!='/': args.indir+='/'

    ## Read the config_interative file
    ## Is there an interactive config file in the current directory?
    readInteractiveConfig = False
    pathInterativeConfig = args.indir
    if os.path.isfile(args.indir+'interactive_config.ini'):
        readInteractiveConfig = True
    elif os.path.isfile(args.indir+'../interactive_config.ini'):
        readInteractiveConfig = True
        pathInterativeConfig=args.indir+'../'

    if readInteractiveConfig:
        from configparser import ConfigParser, ExtendedInterpolation
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(pathInterativeConfig+'interactive_config.ini')
        pathtogrid_modified  = config['PATHS']['pathToGrid']
    else:
        pathtogrid_modified = None

    interactiveReadConfig = False
    if pathtogrid_modified is None:
        interactiveReadConfig = True

    ## We need to read the config file:
    SA = SpectralAnalysis()
    SA.read_config(args.indir+'config_copy.ini', 
                   interactive=interactiveReadConfig, safe=False)
    ## Bypass the path to grid if already provided in a previous run
    if pathtogrid_modified is not None:
        print(f'Setting pathtogrid to {pathtogrid_modified} '
              +'as requested with interactive_config.ini')
        SA.set_pathtogrid(pathtogrid_modified)

    ## IF we have a relative path in the config file, we need to append ../
    # if (SA.pathtodata[0]!='/') & (SA.pathtodata[0]!='~'): 
    #     SA.pathtodata='../'+SA.pathtodata
    # if (SA.pathtogrid[0]!='/') & (SA.pathtogrid[0]!='~'): 
    #     SA.pathtogrid='../'+SA.pathtogrid
    if SA.pathtodata[-1]!='/': SA.pathtodata+='/'
    if SA.pathtogrid[-1]!='/': SA.pathtogrid+='/'
    # if (SA.linelist[0]!='/') & (SA.linelist[0]!='~'): 
    #     SA.linelist = '../' +  SA.linelist
    if not os.path.isfile(SA.linelist):
        SA.linelist = '../' +  SA.linelist
    if not os.path.isfile(SA.linelist):
        raise Exception('line list not found')

    ## Now store the new path to grid if the user changed it.
    if pathtogrid_modified is None:
        with open('interactive_config.ini', 'w') as f:
            f.write(f'[PATHS]\npathToGrid = {SA.pathtogrid}')

    ## In the future the name of the star should be stored in the results file.
    ## For now I need to guess it.
    if args.indir[-1]=='/': args.indir = args.indir[:-1]
    star = args.indir.split('/')[-1].replace('output_', '')
    if args.indir[-1]!='/': args.indir+='/'
    ## Check if the file exists:
    FileNotFound = False
    if not os.path.isfile(SA.pathtodata+star+'.fits'):
        if not os.path.isfile(SA.pathtodata+star+'_templates.fits'):
            # raise Exception('Cannot find input observation file.')  
            FileNotFound = True    
    if FileNotFound:
        SA.pathtodata='../'+SA.pathtodata     
        if not os.path.isfile(SA.pathtodata+star+'.fits'):
            if not os.path.isfile(SA.pathtodata+star+'_templates.fits'):
                raise Exception('Cannot find input observation file.')      

    ## Observation file
    infile = SA.pathtodata + "{}.fits".format(star)
    infile2 = SA.pathtodata + "{}_templates.fits".format(star)
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
        raise Exception(f'Template file {infile} or {infile2} not found')

    SA.set_star(star) ## Dummy variable to identify the star

    region_file = SA.linelist
    med_wvl, med_spectrum, med_err, berv = SA.load_obs(infile)
    obs_wvl, obs_flux, obs_err, nan_mask, regions = SA.create_regions(
                                                        region_file, med_wvl,
                                                        med_spectrum, med_err, 
                                                        berv)
    nwvls, grid_n, teffs, loggs, mhs, alphas = SA.load_grid(SA.pathtogrid, regions)

    ## Cool. But now I need to read the results file:
    # results = read_res(args.indir+'results_raw.txt')
    results = read_res_v2(args.indir+'results.txt')
    SA.set_from_file(args.indir+'results_raw.txt')

    ## Load the fit-data.fits file if available
    if os.path.isfile(args.indir+'fit-data.fits'):
        from astropy.io import fits
        with fits.open(args.indir+'fit-data.fits') as hdu:
            wvl = hdu['WVL'].data
            flx = hdu['FLUX'].data
            err = hdu['ERROR'].data
            flx_fit = hdu['FLUXFIT'].data
            fit = hdu['FIT'].data
            fitnomag = hdu['FITNOMAG'].data
            idxtofit = tuple(hdu['IDXTOFIT'].data)
            fit_data = {'wvl': wvl, 'flx': flx, 'err': err,
                        'flx_fit': flx_fit, 'fit': fit, 'fitnomag': fitnomag,
                        'idxtofit': idxtofit}
    else:
        args.fit = False


    ## Choose a VALD line list to use to identify the lines.
    from irap_tools import analysis_tools as tls
    from irap_tools import vald_tools
    vald_files_path = '/Users/pcristofari/Data/line-lists/vald-linelists/vald-nohf/'
    vald_files = [vald_files_path+'3500_5.0_9000_15000_ths0.01_nohf.vald',  
                vald_files_path+'3500_5.0_15000_20000_ths0.01_nohf.vald', 
                vald_files_path+'3500_5.0_20000_25000_ths0.01_nohf.vald', ]
    vald_data_list = vald_tools.read_vald(vald_files)


    vald_data = {}
    for listnb in range(0,1):
        _wvl = []; _elements = []; _depth = [];
        for i in range(len(vald_data_list[listnb])):
            _wvl.append(tls.convert_lambda_in_vacuum(float(vald_data_list[listnb][i]['wvl'])))
            _elements.append(vald_data_list[listnb][i]['spec_ion'].split()[0].replace("'", ""))
            _depth.append(float(vald_data_list[listnb][i]['depth']))
        vald_data['wvl'] = np.array(_wvl)
        vald_data['elements'] = np.array(_elements)
        vald_data['depth'] = np.array(_depth)
        # vald_data['wvl'] = np.array([tls.convert_lambda_in_vacuum(float(vald_data_list[listnb][i]['wvl'])) for i in range(len(vald_data_list[listnb]))])
        # vald_data['elements'] = np.array([vald_data_list[listnb][i]['spec_ion'].split()[0].replace("'", "") for i in range(len(vald_data_list[listnb]))])
        # vald_data['depth'] = np.array([float(vald_data_list[listnb][i]['depth']) for i in range(len(vald_data_list[listnb]))])

    fit = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
                        SA.nan_mask, SA.nwvls, SA.grid_n, 
                        SA.coeffs, SA._T, SA._L, SA._M, SA._A,
                        SA.teffs, SA.loggs, SA.mhs, SA.alphas, SA.vb,
                        SA.rv, SA.vsini, SA.vmac, SA.veilingFacToFit, SA._T2, SA.fillTeffs)


    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, TextBox

    SA.valdplots = []
    SA.valdlabels = []

    print('Initialization from configuration file complete.')
    ###############################################################
    ###############################################################

    # SA.set_from_file(results_file)
    SA.region = 0 ## This is just an iterator for the interactive plot
    SA.depthThreshold = 0.1 ## This is a threshold for the identification of lines from VALD
    if SA._T2 is None:
        SA._T2 = SA._T ## Otherwise the slider will have a problem with the initial point

    # SA._T = 4000; SA._L = 4.2; SA._M = 0.; SA._A = 0. 

    # SA.coeffs = [0.13, 0.31, 0.40, 0.03, 0.07, 0.06]
    # SA.veilingFac = [0.02, 0.44, 0.41, 0.03, 0.04, 0.05]

    fit = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
                        SA.nan_mask, SA.nwvls, SA.grid_n, 
                        SA.coeffs, SA._T, SA._L, SA._M, SA._A,
                        SA.teffs, SA.loggs, SA.mhs, SA.alphas, SA.vb,
                        SA.rv, SA.vsini, SA.vmac, SA.veilingFacToFit, SA._T2, SA.fillTeffs)
    # for r in range(len(fit)):
    #     fit[r][:-1] = np.diff(fit[r])/fit[r][:-1]+1
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, TextBox

    def replot(rescale=False):
        # global valdplots
        # global valdlabels
        valdplots = SA.valdplots
        valdlabels = SA.valdlabels
        # print(SA._T, SA._L, SA._M, SA.vsini, SA.vmac)
        newefit = SA.gen_spec(SA.obs_wvl, SA.obs_flux, SA.obs_err, 
                    SA.nan_mask, SA.nwvls, SA.grid_n, 
                    SA.coeffs, SA._T, SA._L, SA._M, SA._A,
                    SA.teffs, SA.loggs, SA.mhs, SA.alphas, SA.vb,
                    SA.rv, SA.vsini, SA.vmac, SA.veilingFacToFit, SA._T2, SA.fillTeffs)
        # for r in range(len(newefit)):
        #     newefit[r][:-1] = np.diff(newefit[r])/newefit[r][:-1]+1
        #
        newefit[:, :10] = np.nan
        newefit[:, -10:] = np.nan
        line.set_xdata(SA.obs_wvl[SA.region])
        line.set_ydata(newefit[SA.region])
        dataline.set_xdata(SA.obs_wvl[SA.region])
        dataline.set_ydata(SA.obs_flux[SA.region])
        for _i in range(len(fit_data['wvl'])):
            if (fit_data['wvl'][_i][0]>=SA.obs_wvl[SA.region][0]) \
                & (fit_data['wvl'][_i][-1]<=SA.obs_wvl[SA.region][-1]):
                dataline_fit_data.set_xdata(fit_data['wvl'][_i])
                dataline_fit_data.set_ydata(fit_data['fit'][_i])
                break
        resline.set_xdata(SA.obs_wvl[SA.region])
        resline_all.set_xdata(SA.obs_wvl[SA.region])
        resline_all.set_ydata(SA.obs_flux[SA.region]-newefit[SA.region])
        resline.set_ydata(SA.obs_flux_tofit[SA.region]-newefit[SA.region])
        ## Compute the chi2 for that specific region
        ones = np.ones(SA.obs_flux_tofit[SA.region].shape)
        count = np.sum(ones[~np.isnan(SA.obs_flux_tofit[SA.region])])
        _chi2 = np.nansum((SA.obs_flux_tofit[SA.region]-newefit[SA.region])/SA.obs_err[SA.region])**2/count
        ax.set_title(r"$\chi^2_r=$ {:0.4f}".format(_chi2))
        # ax2.autoscale()
        #
        ## Replot the vald lines:
        ## Plot the VALD lines
        _idx = np.where((vald_data['wvl']>SA.obs_wvl[SA.region][0]) 
                        & (vald_data['wvl']<SA.obs_wvl[SA.region][-1])
                        & (vald_data['depth']>SA.depthThreshold))
        # for vald_plot in valdplots:
        #     vald_plot.axes.cla()    
        # valdplots = []
        # if len(vald_data['wvl'][_idx])<len(valdplots):
        ## FOR NOW I AM TURNING OFF THE LINES THAT ARE MESSING UP INTERACTIVE
        ## TODO: FIX THIS
        # try:
        #     if len(vald_data['wvl'][_idx])>len(valdplots):
        #         for i in range(len(valdplots)):
        #             valdplots[i].set_xdata(vald_data['wvl'][_idx][i])
        #             _min_y, _max_y = ax.get_ylim()
        #             _span = _max_y - _min_y
        #             valdlabels[i].set_position((vald_data['wvl'][_idx][i], _max_y+0.05*_span))
        #             valdlabels[i].set_text(vald_data['elements'][_idx][i])
        #         for i in range(len(valdplots), len(vald_data['wvl'][_idx])-1):
        #             vald_plot = ax.axvline(vald_data['wvl'][_idx][i])
        #             _min_y, _max_y = ax.get_ylim()
        #             _span = _max_y - _min_y
        #             vald_label = ax.text(vald_data['wvl'][_idx][i], _max_y+0.05*_span, vald_data['elements'][_idx][i])
        #             valdplots.append(vald_plot)
        #             valdlabels.append(vald_label)
        #     if len(vald_data['wvl'][_idx])<len(valdplots):
        #         for i in range(len(vald_data['wvl'][_idx])):
        #             valdplots[i].set_xdata(vald_data['wvl'][_idx][i])
        #             _min_y, _max_y = ax.get_ylim()
        #             _span = _max_y - _min_y
        #             valdlabels[i].set_position((vald_data['wvl'][_idx][i], _max_y+0.05*_span))
        #             valdlabels[i].set_text(vald_data['elements'][_idx][i])
        #             # valdplots.append(vald_plot)
        #         for i in range(len(vald_data['wvl'][_idx]), len(valdplots)):
        #             valdplots[i].set_xdata(None)
        #             valdlabels[i].set_text(None)
        #             valdlabels[i].set_position((None, None))
        #     if len(vald_data['wvl'][_idx])==len(valdplots):
        #         for i in range(len(valdplots)):
        #             valdplots[i].set_xdata(vald_data['wvl'][_idx][i])
        #             _min_y, _max_y = ax.get_ylim()
        #             _span = _max_y - _min_y
        #             valdlabels[i].set_position((vald_data['wvl'][_idx][i], _max_y+0.05*_span))
        #             valdlabels[i].set_text(vald_data['elements'][_idx][i])
        # except:
        #     from IPython import embed; embed()
        ##
        if rescale:
            ax.relim()
            ax.autoscale()
            ax2.relim()
        ##
        sumcoefffs = np.sum(SA.coeffs)
        bf = np.sum(SA.coeffs*np.arange(0, 2*len(SA.coeffs), 2))

        ## Update values:
        text_asbox_teff.set_val("{:0.2f}".format(SA._T))
        text_asbox_logg.set_val("{:0.2f}".format(SA._L))
        text_asbox_mh.set_val("{:0.2f}".format(SA._M))
        text_asbox_alpha.set_val("{:0.2f}".format(SA._A))
        text_asbox_vsini.set_val("{:0.2f}".format(SA.vsini))
        text_asbox_vmac.set_val("{:0.2f}".format(SA.vmac))
        text_asbox_rv.set_val("{:0.2f}".format(SA.rv))

        text_box.set_val("{:0.3f}".format(sumcoefffs))  # Trigger `submit` with the initial string.
        text_box2.set_val("{:0.3f}".format(bf))  # Trigger `submit` with the initial string.
        if sumcoefffs > 1.0:
            text_box.text_disp.set_color('red')  # Trigger `submit` with the initial string.
        else:
            text_box.text_disp.set_color('black')  # Trigger `submit` with the initial string.
        fig.canvas.draw_idle()

        _refwvls = []
        _chi2s = []
        for i in range(len(SA.obs_flux_tofit)):
            ones = np.ones(SA.obs_flux_tofit[i].shape)
            count = np.sum(ones[~np.isnan(SA.obs_flux_tofit[i])])
            _refwvls.append(np.mean(SA.obs_wvl[i]))
            _chi2s.append(np.nansum((SA.obs_flux_tofit[i]-newefit[i])/SA.obs_err[i])**2/count)

        dataline21_all.set_ydata(np.concatenate(SA.obs_flux))
        dataline21.set_ydata(np.concatenate(SA.obs_flux_tofit))
        modelline21.set_ydata(np.concatenate(newefit))
        chi2line22.set_ydata(_chi2s)
        ones = np.ones(SA.obs_flux_tofit.shape)
        count = np.sum(ones[~np.isnan(SA.obs_flux_tofit)])

        myerr = SA.obs_err[SA.IDXTOFIT] * np.sqrt(SA.normFactor)
        _resup = (SA.obs_flux_tofit[SA.IDXTOFIT] - newefit[SA.IDXTOFIT])**2
        _resdown = myerr**2
        mychi2 = np.nansum(_resup/_resdown)
        # ax22.set_title(round(mychi2/count))
        ax22.set_title("{} {:0.0f}".format(round(mychi2/count), np.nansum(_chi2s)/len(_chi2s)/1000))

        ax22.relim()
        ax22.autoscale()
        fig2.canvas.draw_idle()


    ## This is ugly but this is a fix for when we don't provide enough filling factors.
    inicoeffs = np.copy(SA.coeffs)
    if len(inicoeffs)< 6:
        while len(inicoeffs)<6:
            inicoeffs = np.append(inicoeffs, 0)

    # print(inicoeffs)

    # The function to be called anytime a slider's value changes
    ## PARAMETERS CONTROLLED BY A SINGLE VALUE 
    def update_teff(__T):
        SA._T = float(__T) 
        replot()
    def update_logg(__L):
        SA._L = float(__L)
        replot()
    def update_mh(__M):
        SA._M = float(__M)
        replot()
    def update_alpha(__A):
        SA._A = float(__A)
        replot()
    def update_vsini(_vsini):
        SA.vsini = float(_vsini)
        replot()
    def update_vmac(_vmac):
        SA.vmac = float(_vmac)
        replot()
    def update_rv(_rv):
        SA.rv = float(_rv)
        replot()

    def update_teff_increase(event):
        SA._T+=5
        replot()
    def update_logg_increase(event):
        SA._L+=0.05
        replot()
    def update_mh_increase(event):
        SA._M+=0.05
        replot()
    def update_alpha_increase(event):
        SA._A+=0.05
        replot()
    def update_vsini_increase(event):
        SA.vsini+=0.05
        replot()
    def update_vmac_increase(event):
        SA.vmac+=0.05
        replot()
    def update_rv_increase(event):
        SA.rv+=0.05
        replot()
    # --
    def update_teff_decrease(event):
        SA._T-=5
        replot()
    def update_logg_decrease(event):
        SA._L-=0.05
        replot()
    def update_mh_decrease(event):
        SA._M-=0.05
        replot()
    def update_alpha_decrease(event):
        SA._A-=0.05
        replot()
    def update_vsini_decrease(event):
        SA.vsini-=0.05
        replot()
    def update_vmac_decrease(event):
        SA.vmac-=0.05
        replot()
    def update_rv_decrease(event):
        SA.rv-=0.1
        replot()

    ## MAGNETIC FIELD COEFFICIENTS (Right now assumes that they must be 2, 4, 6, 8 and 10 kG)
    def update_coeff1(_coeff1):
        SA.coeffs[1] = _coeff1
        SA.coeffs[0] = 1 - np.sum(SA.coeffs[1:])
        if SA.coeffs[0]<0: SA.coeffs[0] = 0
        replot()
    def update_coeff2(_coeff2):
        SA.coeffs[2] = _coeff2
        SA.coeffs[0] = 1 - np.sum(SA.coeffs[1:])
        if SA.coeffs[0]<0: SA.coeffs[0] = 0
        replot()
    def update_coeff3(_coeff3):
        SA.coeffs[3] = _coeff3
        SA.coeffs[0] = 1 - np.sum(SA.coeffs[1:])
        if SA.coeffs[0]<0: SA.coeffs[0] = 0
        replot()
    def update_coeff4(_coeff4):
        SA.coeffs[4] = _coeff4
        SA.coeffs[0] = 1 - np.sum(SA.coeffs[1:])
        if SA.coeffs[0]<0: SA.coeffs[0] = 0
        replot()
    def update_coeff5(_coeff5):
        SA.coeffs[5] = _coeff5
        SA.coeffs[0] = 1 - np.sum(SA.coeffs[1:])
        if SA.coeffs[0]<0: SA.coeffs[0] = 0
        replot()

    ## VEILING PARAMETERS (This is were gets tricky)
    ## I will assume that we need values for YIJHKL
    # global ii
    jj = 0 ## the index in the veiling array
    kk = 0
    indices = {}
    indices_fit = {}
    for letter in ['I', 'Y', 'J', 'H', 'K', 'L']:
        if letter in SA.veilingBands:
            indices[letter] = jj
            jj+=1
        if letter in SA.fitBands:
            indices_fit[letter] = kk
            kk+=1   


    if 'I' in SA.veilingBands:
        def update_veil0(_veil):
            ii = indices['I']
            SA.veilingFac[ii] = _veil
            if 'I' in SA.fitBands:
                kk = indices_fit['I']
                SA.veilingFacToFit[kk] = _veil
            replot()
    if 'Y' in SA.veilingBands:
        def update_veil1(_veil):
            ii = indices['Y']
            SA.veilingFac[ii] = _veil
            if 'Y' in SA.fitBands:
                kk = indices_fit['Y']
                SA.veilingFacToFit[kk] = _veil
            replot()
    if 'J' in SA.veilingBands:
        def update_veil2(_veil):
            ii = indices['J']
            SA.veilingFac[ii] = _veil
            if 'J' in SA.fitBands:
                kk = indices_fit['J']
                SA.veilingFacToFit[kk] = _veil
            replot()
    if 'H' in SA.veilingBands:
        def update_veil3(_veil):
            ii = indices['H']
            SA.veilingFac[ii] = _veil
            if 'H' in SA.fitBands:
                kk = indices_fit['H']
                SA.veilingFacToFit[kk] = _veil
            replot()
    if 'K' in SA.veilingBands:
        def update_veil4(_veil):
            ii = indices['K']
            SA.veilingFac[ii] = _veil
            if 'K' in SA.fitBands:
                kk = indices_fit['K']
                SA.veilingFacToFit[kk] = _veil
            replot()
    if 'L' in SA.veilingBands:
        def update_veil5(_veil):
            ii = indices['L']
            SA.veilingFac[ii] = _veil
            if 'L' in SA.fitBands:
                kk = indices_fit['L']
                SA.veilingFacToFit[kk] = _veil
            replot()

    def update_regionplus():
        SA.region = SA.region+1
        replot(rescale=True)
    def update_regionmoins():
        SA.region = SA.region-1
        replot(rescale=True)

    def update_teff2(__T2):
        SA._T2 = __T2 
        replot()

    def update_fillteffs(_fill1):
        SA.fillTeffs[1] = _fill1
        SA.fillTeffs[0] = 1 - SA.fillTeffs[1] 
        replot()


    # Create the figure and the line that we will manipulate
    fig2, (ax21, ax22) = plt.subplots(2, 1, figsize=(2*6.4, 4.8), sharex=True)
    _refwvls = []
    _chi2s = []
    for i in range(len(SA.obs_flux_tofit)):
        ones = np.ones(SA.obs_flux_tofit[i].shape)
        count = np.sum(ones[~np.isnan(SA.obs_flux_tofit[i])])
        _refwvls.append(np.mean(SA.obs_wvl[i]))
        _chi2s.append(np.nansum((SA.obs_flux_tofit[i]-fit[i])/SA.obs_err[i])**2/count)

    dataline21_all, = ax21.plot(np.concatenate(SA.obs_wvl), 
                                np.concatenate(SA.obs_flux), 
                                '-', color='gray', lw=2)
    dataline21, = ax21.plot(np.concatenate(SA.obs_wvl), 
                            np.concatenate(SA.obs_flux_tofit), 
                            '-', color='black', lw=2)
    modelline21, = ax21.plot(np.concatenate(SA.obs_wvl), 
                             np.concatenate(fit), color='red', lw=2)
    ax22.plot(_refwvls, _chi2s, 'o', lw=2, color='gray')
    chi2line22, = ax22.plot(_refwvls, _chi2s, 'o', lw=2)

    ones = np.ones(SA.obs_flux_tofit.shape)
    count = np.sum(ones[~np.isnan(SA.obs_flux_tofit)])

    myerr = SA.obs_err[SA.IDXTOFIT] * np.sqrt(SA.normFactor)
    _resup = (SA.obs_flux_tofit[SA.IDXTOFIT] - fit[SA.IDXTOFIT])**2
    _resdown = myerr**2
    mychi2 = np.nansum(_resup/_resdown)
    ax22.set_title("{} {:0.2f}".format(round(mychi2/count), np.nansum(_chi2s)))

    # Create the figure and the line that we will manipulate
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(2*6.4, 4.8), sharex=True)
    # -- plot the data
    dataline,  = ax.plot(SA.obs_wvl[SA.region], SA.obs_flux[SA.region], lw=2)
    line, = ax.plot(SA.obs_wvl[SA.region], fit[SA.region], lw=2)
    ## Either we have the same number of regions or we don't...
    dataline_fit_data, = ax.plot([],[], '--', lw=2)
    for _i in range(len(fit_data['wvl'])):
        if (fit_data['wvl'][_i][0]>=SA.obs_wvl[SA.region][0]) \
            & (fit_data['wvl'][_i][-1]<=SA.obs_wvl[SA.region][-1]):
            dataline_fit_data.set_xdata(fit_data['wvl'][_i])
            dataline_fit_data.set_ydata(fit_data['fit'][_i])
            break
    # -- plot the residuals
    resline_all,  = ax2.plot(SA.obs_wvl[SA.region], 
                             SA.obs_flux[SA.region]-fit[SA.region], 
                             color='gray', lw=2)
    resline,  = ax2.plot(SA.obs_wvl[SA.region], 
                         SA.obs_flux_tofit[SA.region]-fit[SA.region], lw=2)
    ax2.set_ylim(-0.04, 0.04)

    # #########################################################
    # ## Add the regions from two line lists to compare them:
    # linelist1 = '/Users/pcristofari/Applications/irap_tools/data/line_lists/new_final_list.txt'
    # linelist2 = '/Users/pcristofari/Applications/irap_tools/data/line_lists/asap-list-filtered-2.txt'
    # ## Read the data in the lists:
    # f = open(linelist1, 'r')
    # list1_windows = []
    # for line in f.readlines():
    #     if line.strip()[0]=='#': continue ## Comment
    #     if line.strip()=='': continue ## Empty line
    #     sl = line.split()
    #     list1_windows.append([float(sl[1]), float(sl[2])])
    # f.close()
    # #
    # f = open(linelist2, 'r')
    # list2_windows = []
    # for line in f.readlines():
    #     if line.strip()[0]=='#': continue ## Comment
    #     if line.strip()=='': continue ## Empty line
    #     sl = line.split()
    #     list2_windows.append([float(sl[1]), float(sl[2])])
    # f.close()
    # ## Plot the windows
    # for i in range(len(list1_windows)):
    #     ax.axvspan(list1_windows[i][0]*10, list1_windows[i][1]*10, facecolor='g', alpha=0.5)
    # for i in range(len(list2_windows)):
    #     ax.axvspan(list2_windows[i][0]*10, list2_windows[i][1]*10, facecolor='r', alpha=0.5)
    # #########################################################

    ## Plot the VALD lines
    _idx = np.where((vald_data['wvl']>SA.obs_wvl[SA.region][0]) 
                    & (vald_data['wvl']<SA.obs_wvl[SA.region][-1])
                    & (vald_data['depth']>SA.depthThreshold))

    SA.valdplots = []
    SA.valdlabels = []
    for i in range(len(vald_data['wvl'][_idx])):
        vald_plot = ax.axvline(vald_data['wvl'][_idx][i])
        _min_y, _max_y = ax.get_ylim()
        vald_label = ax.text(vald_data['wvl'][_idx][i], _max_y, vald_data['elements'][_idx][i])
        # vald_label.set_position((vald_data['wvl'][_idx][i], _max_y/2))
        # vald_label.set_text('test')
        SA.valdplots.append(vald_plot)
        SA.valdlabels.append(vald_label)

    ## Compute the chi2 for that specific region
    ones = np.ones(SA.obs_flux_tofit[SA.region].shape)
    count = np.sum(ones[~np.isnan(SA.obs_flux_tofit[SA.region])])
    _chi2 = np.nansum((SA.obs_flux_tofit[SA.region]-fit[SA.region])/SA.obs_err[SA.region])**2/count
    ax.set_title(r"$\chi^2_r=$ {:0.4f}".format(_chi2))

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.20, bottom=0.05, right=0.85)

    teff_valinit=SA._T
    ypos = 0.95
    xpos = 0.05
    text_asbox_teff = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$T_{\rm eff}$", 
                              textalignment="center")
    text_asbox_teff.set_val("{:0.2f}".format(SA._T))
    text_asbox_teff.on_submit(update_teff)
    teff_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    teff_decrease.on_clicked(update_teff_decrease)
    teff_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    teff_increase.on_clicked(update_teff_increase)
    #
    logg_valinit=SA._L
    ypos = 0.90
    xpos = 0.05
    text_asbox_logg = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$\log{g}$", 
                              textalignment="center")
    text_asbox_logg.set_val("{:0.2f}".format(SA._L))
    text_asbox_logg.on_submit(update_logg)
    logg_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    logg_decrease.on_clicked(update_logg_decrease)
    logg_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    logg_increase.on_clicked(update_logg_increase)
    #
    mh_valinit=SA._M
    ypos = 0.85
    xpos = 0.05
    text_asbox_mh = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$\rm [M/H]$", 
                              textalignment="center")
    text_asbox_mh.set_val("{:0.2f}".format(SA._M))
    text_asbox_mh.on_submit(update_mh)
    mh_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    mh_decrease.on_clicked(update_mh_decrease)
    mh_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    mh_increase.on_clicked(update_mh_increase)
    #
    alpha_valinit=SA._A
    ypos = 0.80
    xpos = 0.05
    text_asbox_alpha = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$\rm [\alpha/Fe]$", 
                              textalignment="center")
    text_asbox_alpha.set_val("{:0.2f}".format(SA._A))
    text_asbox_alpha.on_submit(update_alpha)
    alpha_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    alpha_decrease.on_clicked(update_alpha_decrease)
    alpha_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    alpha_increase.on_clicked(update_alpha_increase)
    #
    vsini_valinit=SA.vsini
    ypos = 0.75
    # xpos = 0.05
    text_asbox_vsini = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$v\sin{i}$", 
                              textalignment="center")
    text_asbox_vsini.set_val("{:0.2f}".format(SA.vsini))
    text_asbox_vsini.on_submit(update_vsini)
    vsini_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    vsini_decrease.on_clicked(update_vsini_decrease)
    vsini_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    vsini_increase.on_clicked(update_vsini_increase)
    #
    vmac_valinit=SA.vmac
    ypos = 0.70
    text_asbox_vmac = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$\zeta_{\rm RT}$", 
                              textalignment="center")
    text_asbox_vmac.set_val("{:0.2f}".format(SA.vmac))
    text_asbox_vmac.on_submit(update_vmac)
    vmac_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    vmac_decrease.on_clicked(update_vmac_decrease)
    vmac_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    vmac_increase.on_clicked(update_vmac_increase)
    #
    rv_valinit=SA.rv
    ypos = 0.65
    text_asbox_rv = TextBox(fig.add_axes([xpos, ypos, 0.06, 0.04]), 
                              r"$RV$", 
                              textalignment="center")
    text_asbox_rv.set_val("{:0.2f}".format(SA.rv))
    text_asbox_rv.on_submit(update_rv)
    rv_decrease = Button(fig.add_axes([xpos+0.06, ypos, 0.03, 0.04]), 
                              '-', hovercolor='0.975')
    rv_decrease.on_clicked(update_rv_decrease)
    rv_increase = Button(fig.add_axes([xpos+0.06+0.03, ypos, 0.03, 0.04]), 
                              '+', hovercolor='0.975')
    rv_increase.on_clicked(update_rv_increase)

    # Make a horizontal slider to control the frequency.
    # axteff = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    # teff_slider = Slider(
    #     ax=axteff,
    #     label='Teff [K]',
    #     valmin=SA.teffs[0],
    #     valmax=SA.teffs[-1],
    #     valinit=SA._T,
    # )

    # # Make a horizontal slider to control the frequency.
    # axlogg = fig.add_axes([0.25, 0.22, 0.65, 0.03])
    # logg_slider = Slider(
    #     ax=axlogg,
    #     label='log(g) [dex]',
    #     valmin=SA.loggs[0],
    #     valmax=SA.loggs[-1],
    #     valinit=SA._L,
    # )
    # # Make a horizontal slider to control the frequency.
    # axmh = fig.add_axes([0.25, 0.18, 0.65, 0.03])
    # mh_slider = Slider(
    #     ax=axmh,
    #     label='[M/H] [dex]',
    #     valmin=SA.mhs[0],
    #     valmax=SA.mhs[-1],
    #     valinit=SA._M,
    # )
    # # Make a horizontal slider to control the frequency.
    # axalpha = fig.add_axes([0.25, 0.14, 0.65, 0.03])
    # alpha_slider = Slider(
    #     ax=axalpha,
    #     label=r'[$\alpha$/H] [dex]',
    #     valmin=SA.alphas[0],
    #     valmax=SA.alphas[-1],
    #     valinit=SA._A,
    # )
    # # Make a horizontal slider to control the frequency.
    # axvsini = fig.add_axes([0.25, 0.10, 0.65, 0.03])
    # vsini_slider = Slider(
    #     ax=axvsini,
    #     label='vsini [km/s]',
    #     valmin=0,
    #     valmax=50,
    #     valinit=SA.vsini,
    # )
    # # Make a horizontal slider to control the frequency.
    # axvmac = fig.add_axes([0.25, 0.06, 0.65, 0.03])
    # vmac_slider = Slider(
    #     ax=axvmac,
    #     label=r'$\zeta_{\rm RT}$ [km/s]',
    #     valmin=0,
    #     valmax=20,
    #     valinit=SA.vmac,
    # )
    # # Make a horizontal slider to control the frequency.
    # axrv = fig.add_axes([0.25, 0.02, 0.65, 0.03])
    # rv_slider = Slider(
    #     ax=axrv,
    #     label=r'RV [km/s]',
    #     valmin=-10,
    #     valmax=10,
    #     valinit=SA.rv,
    # )
    #####
    ##### THIS FOR THE COEFFICIENTS (magnetic)
    width_slider = 0.08
    axcoeff1 = fig.add_axes([0.85, 0.90, width_slider, 0.03])
    coeff1_slider = Slider(
        ax=axcoeff1,
        label=r'$f_{\rm 2\,kG}$ [km/s]',
        valmin=0,
        valmax=1,
        valinit=inicoeffs[1],
    )
    axcoeff2 = fig.add_axes([0.85, 0.86, width_slider, 0.03])
    coeff2_slider = Slider(
        ax=axcoeff2,
        label=r'$f_{\rm 4\,kG}$ [km/s]',
        valmin=0,
        valmax=1,
        valinit=inicoeffs[2],
    )
    axcoeff3 = fig.add_axes([0.85, 0.82, width_slider, 0.03])
    coeff3_slider = Slider(
        ax=axcoeff3,
        label=r'$f_{\rm 6\,kG}$ [km/s]',
        valmin=0,
        valmax=1,
        valinit=inicoeffs[3],
    )
    axcoeff4 = fig.add_axes([0.85, 0.78, width_slider, 0.03])
    coeff4_slider = Slider(
        ax=axcoeff4,
        label=r'$f_{\rm 8\,kG}$ [km/s]',
        valmin=0,
        valmax=1,
        valinit=inicoeffs[4],
    )
    axcoeff5 = fig.add_axes([0.85, 0.74, width_slider, 0.03])
    coeff5_slider = Slider(
        ax=axcoeff5,
        label=r'$f_{\rm 10\,kG}$ [km/s]',
        valmin=0,
        valmax=1,
        valinit=inicoeffs[5],
    )
    ##################################################
    #####
    ##### This for the veiling
    ii=0
    if 'I' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]
        axveil0 = fig.add_axes([0.05, 0.56, 0.10, 0.03])
        veil0_slider = Slider(
            ax=axveil0,
            label=r'$r$  ',
            valmin=-4,
            valmax=5,
            valinit=default,
        )
        ii+=1
    if 'Y' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]
        axveil1 = fig.add_axes([0.05, 0.52, 0.10, 0.03])
        veil1_slider = Slider(
            ax=axveil1,
            label=r'$r_{\rm Y}$  ',
            valmin=-4,
            valmax=4,
            valinit=default,
        )
        ii+=1
        print(ii)
    if 'J' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]

        axveil2 = fig.add_axes([0.05, 0.48, 0.10, 0.03])
        veil2_slider = Slider(
            ax=axveil2,
            label=r'$r_{\rm J}$  ',
            valmin=-4,
            valmax=4,
            valinit=default,
        )
        ii+=1
    if 'H' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]
        axveil3 = fig.add_axes([0.05, 0.44, 0.10, 0.03])
        veil3_slider = Slider(
            ax=axveil3,
            label=r'$r_{\rm H}$  ',
            valmin=-4,
            valmax=4,
            valinit=default,
        )
        ii+=1
    if 'K' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]
        axveil4 = fig.add_axes([0.05, 0.40, 0.10, 0.03])
        veil4_slider = Slider(
            ax=axveil4,
            label=r'$r_{\rm K}$  ',
            valmin=-4,
            valmax=4,
            valinit=default,
        )
        ii+=1
    if 'L' in SA.veilingBands:
        if np.isnan(SA.veilingFac[ii]):
            default = 0
        else:
            default = SA.veilingFac[ii]
        axveil5 = fig.add_axes([0.05, 0.36, 0.10, 0.03])
        veil5_slider = Slider(
            ax=axveil5,
            label=r'$r$  ',
            valmin=-4,
            valmax=4,
            valinit=default,
        )
        ii+=1

    ## Now a second temperature
    axteff2 = fig.add_axes([0.05, 0.32, 0.10, 0.03])
    teff2_slider = Slider(
        ax=axteff2,
        label=r'$T_{\rm eff, 2}$',
        valmin=3000,
        valmax=4200,
        valinit=SA._T2,
    )

    ## Now a second temperature
    axfillteffs = fig.add_axes([0.05, 0.28, 0.10, 0.03])
    fillteffs_slider = Slider(
        ax=axfillteffs,
        label=r'fill Teffs',
        valmin=0,
        valmax=1,
        valinit=SA.fillTeffs[1],
    )


    # axbox = fig.add_axes([0.1, 0.60, 0.05, 0.075])
    # text_box = TextBox(axbox, r"$\sum{f_i}$", textalignment="center")
    # text_box.set_val("{}".format(np.sum(SA.coeffs)))  

    axbox = fig.add_axes([0.85, 0.64, 0.05, 0.075])
    text_box = TextBox(axbox, r"$\sum{f_i}$", textalignment="center")
    text_box.set_val("{:0.3f}".format(np.sum(SA.coeffs)))
    axbox2 = fig.add_axes([0.97, 0.64, 0.05, 0.075])
    text_box2 = TextBox(axbox2, r"$\langle B\rangle$", textalignment="center")
    text_box2.set_val("{:0.2f}".format(
        np.sum(SA.coeffs*np.arange(0, 2*len(SA.coeffs), 2))))

    # register the update function with each slider
    # teff_slider.on_changed(update_teff)
    # logg_slider.on_changed(update_logg)
    # mh_slider.on_changed(update_mh)
    # alpha_slider.on_changed(update_alpha)
    # vsini_slider.on_changed(update_vsini)
    # vmac_slider.on_changed(update_vmac)
    # rv_slider.on_changed(update_rv)
    coeff1_slider.on_changed(update_coeff1)
    coeff2_slider.on_changed(update_coeff2)
    coeff3_slider.on_changed(update_coeff3)
    coeff4_slider.on_changed(update_coeff4)
    coeff5_slider.on_changed(update_coeff5)

    if 'I' in SA.veilingBands:
        veil0_slider.on_changed(update_veil0)
    if 'Y' in SA.veilingBands:
        veil1_slider.on_changed(update_veil1)
    if 'J' in SA.veilingBands:
        veil2_slider.on_changed(update_veil2)
    if 'H' in SA.veilingBands:
        veil3_slider.on_changed(update_veil3)
    if 'K' in SA.veilingBands:
        veil4_slider.on_changed(update_veil4)
    if 'L' in SA.veilingBands:
        veil5_slider.on_changed(update_veil5)
    #
    teff2_slider.on_changed(update_teff2)
    fillteffs_slider.on_changed(update_fillteffs)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.05, 0.055, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    relimax = fig.add_axes([0.05, 0.155, 0.1, 0.04])
    relimbutton = Button(relimax, 'ReLim', hovercolor='0.975')
    nextregion_pos = fig.add_axes([0.10, 0.015, 0.05, 0.04])
    nextregionbutton = Button(nextregion_pos, '>>', hovercolor='0.975')
    prevregion_pos = fig.add_axes([0.05, 0.015, 0.05, 0.04])
    prevregionbutton = Button(prevregion_pos, '<<', hovercolor='0.975')

    def reset(event):
        # teff_slider.reset()
        # logg_slider.reset()
        # mh_slider.reset()
        # vsini_slider.reset()
        # vmac_slider.reset()
        # rv_slider.reset()
        coeff1_slider.reset()
        coeff2_slider.reset()
        coeff3_slider.reset()
        coeff4_slider.reset()
        coeff5_slider.reset()
        veil0_slider.reset()
        veil1_slider.reset()
        veil2_slider.reset()
        veil3_slider.reset()
        veil4_slider.reset()
        veil5_slider.reset()
    def relim(event):
        replot(rescale=True)
    def nextregion(event):
        update_regionplus()
    def prevregion(event):
        update_regionmoins()


    button.on_clicked(reset)
    relimbutton.on_clicked(relim)
    nextregionbutton.on_clicked(nextregion)
    prevregionbutton.on_clicked(prevregion)

    # plt.show(block=False)

    plt.show()


    from IPython import embed
    embed()
    exit()
    