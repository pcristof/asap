def read_VO_fits(fname):
    '''Read in the Polarbase format'''
    from astropy.io import fits
    import numpy as np
    data = {}
    with fits.open(fname) as hdu:
        da = hdu[1].data
        if 'AWAV' in da.dtype.names:
            data['wave'] = np.array([da['AWAV']])
            data['wave_type'] = 'air'
        elif 'WAVE' in da.dtype.names:
            data['wave'] = np.array([da['WAVE']])
            data['wave_type'] = 'vac'
        else:
            print('Could not read wavelength keyword')
            print('Is this really an OV file?')
            # return 1
        data['flux'] = np.array([da['FLUX_NOR']])
        try:
            data['err'] = np.array([da['FLUX_ERR']])
        except:
            data['err'] = np.sqrt(np.array([da['FLUX_NOR']]))
        try:
            data['snr'] = hdu[0].header['SNR_MAX']
        except:
            data['snr'] = hdu[0].header['DER_SNR']
        ## Trying to compute uncertainties from typical SNR:
        ## In the CONTINUUM -> SNR~sqrt(F) -> C=SNR**2
        C = data['snr']**2
        ## So we could get a "flux" such that:
        F = data['flux']*C
        ## But the flux is not normalized here... It was blaze corrected
        ## so we need to adjust:
        maxcont = np.nanpercentile(data['flux'], 99) 
        ## And say, neglecting a lot of things, that the error is sqrt(F)
        data['err'] = np.sqrt(F/maxcont)/C
        ## Errors cannot be negative o
        ## What is the date of the observation?
        data['mjd'] = float(hdu[0].header['TMID'])
        data['jd'] = data['mjd']+2400000.5
        data['header'] = hdu[0].header
        data['RA_TARG'] = float(hdu[0].header['RA_TARG'])
        data['DEC_TARG'] = float(hdu[0].header['DEC_TARG'])
        data['DATE-OBS'] = hdu[0].header['DATE-OBS'].split('T')[0]
    return data

def main():
    '''Script to convert OV observations based on provided list of files'''
    from optparse import OptionParser
    import sys
    from astropy.io import fits
    from importlib.resources import files
    import numpy as np
    from PyAstronomy import pyasl

    KNOWN_INST = ['spirou']

    ## Load the parser
    parser = OptionParser()
    # parser.add_option("-f", "--file_params", dest="file_params", help='Text file containing the default parameters the used wants to set. Any opion in the file will be bypassed by the option provided at runtime.',type='string',default="")
    parser.add_option("--inst", dest="inst", 
                      help="Instrument considered to split orders", 
                      type=str, default='SPIRou')
    ## Search for the non-key filenames
    options, filenames = parser.parse_args(sys.argv[1:])

    ## Check instrument name
    options.inst = options.inst.lower()
    if options.inst not in KNOWN_INST:
        print(f"Error: {options.inst} not known")
        print("SCRIPT END")
        return 1

    ## Get the blaze for this instrument:
    if options.inst=='spirou':
        # blaze_file = paths.support_data + 'blaze_data/blaze_half_flux.txt'
        blaze_file = files("asap.support_data.blaze_data")\
                     .joinpath("blaze_half_flux.txt")

    ## Go through each filename in the list:
    for fname in filenames:
        try:
            data = read_VO_fits(fname)
        except:
            print(f'Could not read {fname}')
            print(f'Is this really an OV file?')
            continue

        ## Choose an output file:
        if fname[-4:]=='.fts':
            output_file = fname.replace('.fts', '_asap.fits')
        ## Choose an output file:
        elif fname[-4:]=='.fits':
            output_file = fname.replace('.fits', '_asap.fits')
        else:
            output_file = fname+'_asap.fits'

        ## Compute the BERV if not provided
        heli, bary = pyasl.baryvel(data['jd'], deq=2000.0)
        ra = float(data['RA_TARG'])
        dec = float(data['DEC_TARG'])
        ## Compute the velocidies
        vh, vb = pyasl.baryCorr(data['jd'], ra, dec, deq=2000.0)
        BERV = vb

        ## Reconstruct the 2D arrays:
        ## Reconstruct the orders:
        wvl_diff = np.diff(data['wave'][0])
        idx_split = np.where((wvl_diff<0))
        wave_2d_list = np.split(data['wave'][0], idx_split[0]+1)
        flux_2d_list = np.split(data['flux'][0], idx_split[0]+1)
        err_2d_list = np.split(data['err'][0], idx_split[0]+1)

        ## Make 2D spectra
        maxlen = 0
        for i in range(len(wave_2d_list)):
            if len(wave_2d_list[i])>maxlen:
                maxlen = len(wave_2d_list[i])
        data['wave_2d'] = np.zeros((len(wave_2d_list), maxlen))*np.nan
        data['flux_2d'] = np.zeros((len(wave_2d_list), maxlen))*np.nan
        data['err_2d'] = np.zeros((len(wave_2d_list), maxlen))*np.nan
        for i in range(len(data['wave_2d'])):
            data['wave_2d'][i][:len(wave_2d_list[i])] = wave_2d_list[i]
            data['flux_2d'][i][:len(flux_2d_list[i])] = flux_2d_list[i]
            data['err_2d'][i][:len(err_2d_list[i])] = err_2d_list[i]

        from asap.spectral_analysis_pack import rebuilt_wavelength
        from asap.spectral_analysis_pack import rebuilt_wavelength_v2
        from asap.spectral_analysis_pack import fill_nans_wavelength
        from asap.spectral_analysis_pack import fill_nans_wavelength_v2

        wvl = data['wave_2d']
        flx = data['flux_2d']
        err = data['err_2d']

        new_wvl, indices = fill_nans_wavelength_v2(wvl)
        new_med_spectrum = np.empty((len(new_wvl), len(new_wvl[0])))*np.nan
        new_med_err = np.empty((len(new_wvl), len(new_wvl[0])))*np.nan
        for r in range(len(new_med_spectrum)):
            new_med_spectrum[r][indices[r]] = flx[r,:-1] 
            new_med_err[r][indices[r]] = err[r, :-1] 
        ## Delete NaNs that are on the right
        myii = 0
        for r in range(len(new_wvl)):
            val = np.nan
            ii = len(new_wvl[r])
            while np.isnan(val):
                ii-=1
                val = new_wvl[r][ii]
            if ii>myii:
                myii=ii
        new_wvl = new_wvl[:, :myii]
        new_med_spectrum = new_med_spectrum[:, :myii]
        new_med_err = new_med_err[:, :myii]

        ## Now we need to actually fill the NaNs with values
        for r in range(len(new_wvl)):
            new_wvl[r] = rebuilt_wavelength_v2(new_wvl[r])

        data['wave_2d'] = new_wvl
        data['flux_2d'] = new_med_spectrum
        data['err_2d'] = new_med_err

        header = fits.Header()
        header['AUTHOR'] = 'Your Name'
        header['COMMENT'] = 'This file was constructed from Polarbase data' 
        header['DATE-OBS'] = data['DATE-OBS']
        header['BERV'] = BERV
        header['SNR'] = data['snr']
        header['SNR-THS'] = 0.
        primary_hdu = fits.PrimaryHDU(header=header)

        # --- Create HDU and write to FITS ---
        hdu1 = fits.ImageHDU(data['wave_2d'], name='WVL')
        hdu2 = fits.ImageHDU(data['flux_2d'], name='TEMPLATE')
        hdu3 = fits.ImageHDU(data['err_2d'], name='ERR_PROPAG')
        hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
        _d = data['DATE-OBS']
        hdul.writeto(output_file, overwrite=True)

        print(f'File created {output_file}')
    
    print('PROGRAM RAN WITH OPTIONS:')
    print(options)
    print('END SCRIPT')


if __name__ == "__main__":

    main()

