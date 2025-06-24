'''
This module contains all the tools used to compute log(g) from data using
empirical relations and/or evolutionary models.
'''

import numpy as np
from scipy.interpolate import interp1d
from astroquery.simbad import Simbad

def boltzmannRadius(T, L, dT=0, dL=0):
    '''Function to compute the radius from effective temperature and 
    bolometric luminosity assuming Boltzmann law.'''
    Ts = 5777. ## effective temperature of the Sun
    rT = T / Ts
    drT = dT / Ts
    R = np.sqrt(L) / rT**2
    A = np.sqrt(L); B = rT**2
    sigmaA = 1/(2*np.sqrt(L))*dL
    sigmaB = 2*rT*drT
    sigma = np.sqrt((A/B)**2*((sigmaA/A)**2 + (sigmaB/B)**2))
    return R, sigma

def BCg_cifuentes(G, J, dG=0, dJ=0):
    '''Correction in the J band for M dwarfs from Cifuentes et al. (2020)
    Uncertainty propagation in Notebook 27/04/2022 page 5.'''
#     output = -2.855844*1e2 + 3.8324*1e-1*X - 2.22*1e-4*X**2 + 7.15*1e-8*X**3 - 1.36*1e-11*X**4 + \
#             1.54*1e-15*X**5 - 9.56*1e-20*X**6 + 2.51*1e-24*X**7
    X = G-J
    dX = np.sqrt(dG**2 + dJ**2)
    if X<1.5:
        print("Color too low")
    if X>5.4:
        print("Color too high")
    a = 0.404; b = 0.161; c = -0.465; d = 0.1159; e = -0.0115
    da = 0.187; db = 0.239; dc = 0.112; dd = 0.0225; de = 0.0017
    output = a + b * X + c * X**2 + d * X**3 + e * X**4
    doutputdX = (b + 2*c*X + 3*d*X**2 + 4*e*X**3)**2 * dX**2 
    outputerr = np.sqrt(da**2 + (X*db)**2 + (X**2*dc)**2 + (X**3*dd)**2 + (X**4*de)**2 + doutputdX)
    # outputerr = np.sqrt(da**2 + (X*db)**2 + (X**2*dc)**2 + (X**3*dd)**2 + (X**4*de)**2)
    # outputerr = np.sqrt((1+b+2*c*X+3*d*X**2)**2 * dG**2 + (-b-2*c*X-3*d*X**2)**2 * dJ**2)
    # print(outputerr)
    # outputerr = np.sqrt(doutputdX)
    return output, outputerr

def BCj_cifuentes(G, J, dG=0, dJ=0):
    '''Correction in the J band for M dwarfs from Cifuentes et al. (2020)
    Uncertainty propagation in Notebook 27/04/2022 page 5.'''
#     output = -2.855844*1e2 + 3.8324*1e-1*X - 2.22*1e-4*X**2 + 7.15*1e-8*X**3 - 1.36*1e-11*X**4 + \
#             1.54*1e-15*X**5 - 9.56*1e-20*X**6 + 2.51*1e-24*X**7
    X = G-J
    dX = np.sqrt(dG**2 + dJ**2)
    if X<1.5:
        print("Color too low")
    if X>5.4:
        print("Color too high")
    a = 0.576; b = 0.735; c = -0.132; d = 0.0115; e = 0.0
    da = 0.085; db = 0.104; dc = 0.038; dd = 0.0045; de = 0.0
    output = a + b * X + c * X**2 + d * X**3 + e * X**4
    doutputdX = ((b + 2*c*X + 3*d*X**2 + 4*e*X**3) * dX)**2 
    outputerr = np.sqrt(da**2 + (X*db)**2 + (X**2*dc)**2 + (X**3*dd)**2 + (X**4*de)**2 + doutputdX)
    # outputerr = np.sqrt(da**2 + (X*db)**2 + (X**2*dc)**2 + (X**3*dd)**2 + (X**4*de)**2)
    # outputerr = doutputdX
    # outputerr = np.sqrt((1+b+2*c*X+3*d*X**2)**2 * dG**2 + (-b-2*c*X-3*d*X**2)**2 * dJ**2)
    # print(outputerr)
    # outputerr = np.sqrt(doutputdX)
    return output, outputerr

def logg_from_lum(teff):
    '''Function computing the logg from the effective temperature'''

    ## Initialize Simbad
    simbad = Simbad()
    # simbad.list_votable_fields()
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

    ## Retrieve from SIMBAD
    result_ids = simbad.query_object(star)

    p = result_ids['PLX_VALUE'][0]; dp = result_ids['PLX_ERROR'][0]
    mg = float(result_ids['FLUX_G'][0]); dmg = float(result_ids['FLUX_ERROR_G'][0])
    mj = float(result_ids['FLUX_J'][0]); dmj = float(result_ids['FLUX_ERROR_J'][0])
    ## Absolute
    d, dd = dist_parsec(p, dp)
    Mg, dMg = convert_mag(mg, d, dmg, dd)
    Mj, dMj = convert_mag(mj, d, dmj, dd)

    ## Bolometric correction
    bcg_cifuentes, dbcg_cifuentes = BCg_cifuentes(Mg, Mj, dMg, dMj)

    bMs = 4.74 ## Sun bolometric magnitude
    bMg_c = Mg + bcg_cifuentes
    dbMg_c = np.sqrt(dMg**2*0 + dbcg_cifuentes**2)

    rL_c = 10**((bMg_c - bMs)/(-2.5))
    drL_c = abs(np.log(10)/(-2.5) * 10**((bMg_c - bMs)/(-2.5)) * dbMg_c)
    
    
    return 0

def compute_veq(Prot, R):
    '''Compute equatorial velocity given a Prot and Radius.
    Prot in days
    Radius in Rsun
    result in km/s.'''
    Rsun = 696340. # km
    num = 2.*np.pi*R*Rsun
    denom = Prot * 24. * 3600.
    res = num / denom
    return res

def read_literature(filename):
    '''Function designed to read literature files and return the values for the 
    stars contained in the file when available.
    Inputs:
    - filename      :   Path to file toread
    Outputs:
    - stars         :   [dict] Dictionary of dictionaries containing the a key
                        per star, then a key per numbered reference then a key
                        per parameter. See structure of the file stars_2.csv 
    '''
    stars = {}
    f = open(filename, 'r')
    for i, line in enumerate(f.readlines()):
        if i==0:
            continue ## Pass headers
        line = line.replace('nan', '0') ## To avoid nans
        l = line.split()
        if l[0]=='END':
            break ## Exit is end reached
        _star = l[0].replace(',', '').strip().lower()
        if _star=='':
            continue ## pass diving lines
        if _star not in stars.keys():
            stars[_star] = {}
        if _star in stars.keys():
            _ref = l[1].replace(',', '').replace('(', '').replace(')', '')
            _teff  = int(l[2].replace(',', ''))
            _dteff  = int(l[4].replace(',', ''))
            _logg   = float(l[5].replace(',', ''))
            _dlogg  =  float(l[7].replace(',', ''))
            _mh     = float(l[8].replace(',', ''))
            _dmh    = float(l[10].replace(',', ''))
            stars[_star][_ref] = {'teff': _teff, 'dteff': _dteff, 
                                  'logg': _logg, 'dlogg': _dlogg,
                                  'mh': _mh, 'dmh': _dmh}
    return stars

def compute_logg(M, R, dM=0, dR=0):
    '''
    Method to compute logg from Mass and Radius (M and R respectively).
    Uses the Sun mass. Compatible with Mann et al. (2015).
    Input parameters:
    - M     :   Stellar Mass ratio (M*/Mo)
    - R     :   Stellar Radius ratio (R*/Ro)
    output parameters:
    - logg  :   Stellar surface gravity (log(g)) [dex] 
    '''
    SunMass = 1.989 * 10**30 #kg
    SunRadius = 6.957 * 10**10 #cm
    G =  6.67430 * 10**(-11) *10**6 # cm3 kg−1 s−2
    sigma = np.sqrt((dM/M)**2 + ((2*dR)/R)**2) / np.log(10)
    return np.log10(G*(M*SunMass)/(R*SunRadius)**2), sigma

def compute_mass(R, logg, dR=0, dlogg=0):
    '''
    Method to compute logg from Mass and Radius (M and R respectively).
    Uses the Sun mass. Compatible with Mann et al. (2015).
    Input parameters:
    - logg  :   log(g) value
    - R     :   Stellar Radius ratio (R*/Ro)
    output parameters:
    - M     :   Stellar Mass ratio (M*/Mo)
    - sigma :   Uncertainty on M. Propag. in notebook Mar. 30, 2022, page 77
    '''
    SunMass = 1.989 * 10**30 #kg
    SunRadius = 6.957 * 10**10 #cm
    G =  6.67430 * 10**(-11) *10**6 # cm3 kg−1 s−2
    # sigma = np.sqrt((dM/M)**2 + ((2*dR)/R)**2) / np.log(10)
    A = (R*SunRadius)**2 / G / SunMass
    B = 10**(logg)
    sigmaA = 2*dR / G / SunMass
    sigmaB = np.log(10)*10**(logg)*dlogg
    sigma = np.sqrt((A*B)**2*((sigmaA/A)**2 + (sigmaB/B)**2))
    # logg = np.log10(G*(M*SunMass)/(R*SunRadius)**2)
    M = (R*SunRadius)**2 * 10**(logg) / G / SunMass
    return M, sigma

def compute_radius(M,logg):
    '''
    Method to compute logg from Mass and Radius (M and R respectively).
    Uses the Sun mass. Compatible with Mann et al. (2015).
    Input parameters:
    - M         :   Stellar Mass ratio (M*/Mo)
    - logg      :   Surface gravity
    output parameters:
    - Radius    :   Stellar Radius ratio (R*/Ro)
    '''
    ## g = GM/R**2

    # Constants
    SunMass = 1.989 * 10**30 #kg
    SunRadius = 6.957 * 10**10 #cm
    G =  6.67430 * 10**(-11) *10**6 # cm3 kg−1 s−2
    ## Compute g
    g = 10**logg

    return np.sqrt((G*M*SunMass)/(g*SunRadius**2))

def mass_lum(M_, band, dM_=0):
    '''
    Funciton to compute the Stellar Mass from Magnitude given a spectral band.
    Uses equations published by Delfosse et al. (2000).
    Input parameters:
    - M_        :   Magnitude
    - band      :   Band used for computation. Can be V, J, H or K
    Output parameters:
    - M*/Mo     :   Stellar Mass ratio (M*/Mo)
    '''
    if band=='V':
        if ((M_>=9) & (M_<=17)):
            a = 0.3
            b = 1.87
            c = 7.6140
            d = -1.6980
            e = 0.060958
        else:
            print(M_)
            raise Exception('Luminosity in V band must be within [9--17]')
    elif band=='J':
        if ((M_>=5.5) & (M_<=11)):
            a = 1.6
            b = 6.01
            c = 14.888
            d = -5.3557
            e = 0.28518*1e-4
        else:
            print(M_)
            raise Exception('Luminosity in J band must be within [5.5--11]')
    elif band=='H':
        if ((M_>=5) & (M_<=10)):
            a = 1.4
            b = 4.76
            c = 10.641
            d = -5.0320
            e = 0.28396
        else:
            print(M_)
            raise Exception('Luminosity in H band must be within [5--10]')
    elif band=='K':
        if ((M_>=4.5) & (M_<=9.5)):
            a = 1.8
            b = 6.12
            c = 13.205
            d = -6.2315
            e = 0.37529
        else:
            print(M_)
            raise Exception('Luminosity in K band must be within [4.5--9.5]')
    
    ## Compute the logg of the masses ratios
    lmass = 1e-3 * (a + b * M_ + c * M_**2 + d * M_**3 + e * M_**4)
    lmassmoins = 1e-3 * (a + b * (M_-dM_) + c * (M_-dM_)**2 + d * (M_-dM_)**3 \
                + e * (M_-dM_)**4)
    lmassplus = 1e-3 * (a + b * (M_+dM_) + c * (M_+dM_)**2 + d * (M_+dM_)**3 \
                + e * (M_+dM_)**4)
    dmasstmp = np.sqrt((1e-3 * (b + 2*M_*c + 3*M_**2*d + 4*M_**3*e) * dM_)**2)
    # dmass = abs(10**lmassplus - 10**lmassmoins)
    dmass = dmasstmp * np.log(10) * 10**lmass
    return 10**lmass, dmass

def mass_lum_mann19(M_, dM_=0, mh=0, band='K'):
    '''Equation 5 in Mann et al. (2019) assuming a zero mass for the primary.'''
    a0 = -0.647; a1 = -0.207; a2 = -6.53*1e-4; a3 = 7.13*1e-3;  
    a4 = 1.84*1e-4; a5 = -1.60*1e-4; f = -0.0035
    if band!='K':
        raise Exception('Wrong band')
    coeffs = [a0, a1, a2, a3, a4, a5]
    resum = np.sum([coeffs[i]*(M_-7.5)**i for i in range(len(coeffs))])
    Mass = (1+f*mh) * (10**(resum))
    subsubsigma = [coeffs[i]*i*(M_ - 7.5)**(i-1) for i in range(len(coeffs))]
    subsigma = np.log(10) * np.sum(subsubsigma) * resum * dM_
    sigma = np.sqrt(subsigma**2 + 0.021**2)
    return Mass, sigma

def mass_lum_mann19_WRONG(M_, band='K', dM_=0):
    '''
    Funciton to compute the Stellar Mass from Magnitude given the Ks band.
    Uses equations published by Mann et al. (2019).
    Input parameters:
    - M_        :   Magnitude
    - band      :   Band used for computation. Can be V, J, H or K
    Output parameters:
    - M*/Mo     :   Stellar Mass ratio (M*/Mo)
    '''
    if band!='K':
        raise Exception('Wrong band')
    if ((M_>3.5) & (M_<=5.0)):
        a = -0.136
        b = 0.36
    elif ((M_>5.0) & (M_<=8.0)):
        a = -0.16
        b = 0.48
    elif ((M_>8.0) & (M_<=11.5)):
        a = -0.11
        b = 0.08
    else:
        print(M_)
        raise Exception('Luminosity in Ks band must be within [3.5--11.5]')
    
    ## Compute the logg of the masses ratios
    lmass = a * M_ + b
    dmass = a * np.log(10) * 10**lmass * dM_
    return 10**lmass, dmass

def dist_parsec(p, dp=0):
    '''
    Function to convert the parallax (expressed in mas, 1mas = 1e-3arcsec) to
    distance in parsecs.
    '''
    dp = 1000/p * dp / p
    return 1000/p, dp

def convert_mag(m, d, dm=0, dd=0):
    '''
    Method to convert magnitude to absolute magnitudes from magnitude and 
    distance.
    Input parameters:
    - m     :   Stellar apparent magnitude
    - d     :   Stellar distance [parsec]
    '''
    M_ = m - 2.5 * np.log10((d/10)**2)
    dM_ = np.sqrt(dm**2 + (-5/(d*np.log(10)) * dd)**2) ## Uncertainty checked and seem correct
    return M_, dM_

def read_baraffe_models(filename=None):
    '''
    Function to read baraffe models and store the values in arrays
    '''
    if filename is None:
        raise Exception('read_baraffe_models: No input file')
        # filename = paths.irap_tools_path + 'baraffe_2015.txt'
    f = open(filename)
    lines = f.readlines()
    f.close()
    ages = []
    data = {}
    for line in lines:
        if len(line)>3:
            if line[3]=='t':
                age = float(line.split()[-1])
                data[age] = {'Mass': [], 'Radius': [], 'Teff': [], 
                             'Luminosity': []}
                ages.append(age)
        if line[0]==" ":
            l = line.split()
            data[age]['Mass'].append(float(l[0]))
            data[age]['Teff'].append(float(l[1]))
            data[age]['Luminosity'].append(float(l[2]))
            data[age]['Radius'].append(float(l[4]))
    return data

def read_dartmouth_models(filename=None, phs='CFHTugriz', mh=0, alpha=0):
    '''
    Function to read dartmouth models for a given metallicity.
    Inputs:
    - phs       :   photometric systems. Can be CHFTugriz or Gaia
    '''
    if filename is None:
        raise Exception('read_dartmouth_models: No input file')
        ##
        # bpath = paths.support_data + 'dartmouth_models/{}/'.format(phs)
        # if mh<0:
        #     mhsign = 'm' # minus sign
        # else:
        #     mhsign = 'p'
        # if alpha<0:
        #     alphasign = 'm' # minus sign
        # else:
        #     alphasign = 'p'
        # metalstr = "feh{}{:02d}".format(mhsign, abs(int(mh*10)))
        # alphastr = "afe{}{:0.0f}".format(alphasign, abs(int(alpha*10)))
        # filename = bpath + metalstr + alphastr + '.{}'.format(phs)
        # filename_2 = bpath + metalstr + alphastr + '.{}_2'.format(phs)
        # print(filename)
        # print(filename_2)

    ages = []
    data = {}
    for filename in [filename, filename_2]:
        f = open(filename)
        lines = f.readlines()
        f.close()
        for line in lines:
            if len(line)>3:
                if 'age' in line.split()[0].lower():
                    age = float(line.split('=')[1].split()[0])
                    # data[age] = {'Mass': [], 'Radius': [], 'Teff': [], 
                    #             'Luminosity': [], 'logg': []}
                    data[age] = {}
                    ages.append(age)
                if 'eep' in line.split()[0].lower():
                    for col in line.split():
                        data[age][col] = [] 
            if line[0]==" ":
                l = line.split()
                for i, key in enumerate(data[age].keys()):
                    data[age][key].append(float(l[i]))
                # data[age]['Mass'].append(float(l[1]))
                # data[age]['Teff'].append(float(l[2]))
                # data[age]['logg'].append(float(l[3]))
                # data[age]['Luminosity'].append(float(l[4]))
    return data

def compute_R_Theta(d, theta, dd, dtheta):
    '''Function designed to compute the R and Theta
    Inputs:
    - d     :   Distance in pc
    - theta :   Angular diameter in mas
    Outputs:
    - r     :   Radius in relative to Sun units.'''
    SunRadius = 6.957 * 10**10 #cm
    dist = d
    disterr = dd
    theta  = theta * 1e-3 ## in arcsec
    thetaerr  = dtheta * 1e-3 ## in arcsec
    theta *= np.pi/(180 * 3600) ## radians
    thetaerr *= np.pi/(180 * 3600) ## radians
    radii    = dist * theta / 2 * 3.086e18 ## in cm
    # radiierr = .5 * dist * theta * np.sqrt((disterr/dist)**2 + (thetaerr/theta)**2) * 3.086e18
    A = 1 / 2 * 3.086e18 / SunRadius
    radiierr = A * np.sqrt((dist * thetaerr)**2 + (theta * disterr)**2)
    radii /= SunRadius
    # radiierr /= SunRadius
    return radii, radiierr

def compute_R_M(M, dM=0, age=5.):
    '''
    Function to compute Radius from Mass based on baraffe models (2015)
    Input parameters:
    - M     :   Stellar Mass of the star
    - age   :   Isochrone used in the evolution model
    Output parameters:
    - R     :   Stellar Radius ratio (R*/Ro)
    '''
    data = read_baraffe_models()
    if M<data[age]['Mass'][0]:
        raise Exception('Mass below range')
    elif M>data[age]['Mass'][-1]:
        raise Exception('Mass above range')
    fun = interp1d(data[age]['Mass'], data[age]['Radius'])
    R = float(fun(M))
    dR = abs(fun(M+dM)-fun(M-dM))
    return R, dR

def compute_M_R(R, dR=0, age=5.):
    '''
    Function to compute Mass from Radius based on baraffe models (2015)
    Input parameters:
    - R     :   Stellar Mass of the star
    - age   :   Isochrone used in the evolution model
    Output parameters:
    - M     :   Stellar Radius ratio (R*/Ro)
    '''
    data = read_baraffe_models()
    if R<data[age]['Radius'][0]:
        raise Exception('Radius below range')
    elif R>data[age]['Radius'][-1]:
        raise Exception('Radius above range')
    fun = interp1d(data[age]['Radius'], data[age]['Mass'])
    M = float(fun(R))
    dM = abs(fun(R+dR)-fun(R-dR))
    return M, dM

def compute_logg_mag(m, band, p, dm=0, dp=0, source='M19'):
    '''
    Function to compute the logg value from the apparent magnitude.
    The function uses parallax to determine distance and absolute magnitude.
    It then uses either Delfosse et al. 2000 (D00) or Mann et al. 2019 (M19) 
    empirical mass-luminosity relations to compute the Mass of the stars. 
    The radius is computed from Baraffe et al. (2015) stellar evolution models,
    assuming an age of 5Gyr (just like Passegger et al. 2019 did). 
    Finally, log(g) is computed from Mass and Radius. 
    Input paramters:
    - m         :   Apparent magnitude of the star.
    - band      :   Band used for magnitude computation.
    - p         :   Parallax. [parsec]
    - dm        :   Uncertainty on magnitude measurement
    - dp        :   Uncertainty on parallax measurement
    - source    :   Relation to use. Either M19 or D00.
    Output parameters:
    - log(g)    :   Surface gravity for the star.
    '''
    d, dd = dist_parsec(p, dp)
    M_, dM_ = convert_mag(m, d, dm, dd)
    if 'M19' in source:
        M, dM = mass_lum_mann19(M_, dM_, 'K')
    elif 'D00' in source:
        M, dM = mass_lum(M_, 'K', dM_)
    else:
        raise Exception("compute_logg_mag --> Unknown source argument")
    R, dR = compute_R_M(M, dM)
    logg, dlogg = compute_logg(M, R, dM)
    return logg, dlogg

def read_simbad_file(filename=None):
    '''
    Function to read files generated by query obtained using 
    generate_simbad_script.py script. Returns data with parallax and magnitude
    for each star.
    '''
    if filename is None:
        raise Exception('read_simbad_file: No input file')
        # filename = paths.irap_tools_path + 'simbad_values.txt'
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = {}
    for line in lines:
        if line[0]=='g': # This is a new star
            star = line.split()[0]
            if star[-1]=='V':
                star = star[:-1]
            data[star] = {'V': np.nan, 'dV': np.nan, 
                            'J': np.nan, 'dJ': np.nan, 
                            'H': np.nan, 'dH': np.nan,
                            'K': np.nan, 'dK': np.nan,
                            'parallax': np.nan, 'dparallax': np.nan}
        if (line[0]=="V") | (line[0]=="J") \
            | (line[0]=="H") | (line[0]=="K"):
            l = line.split()
            data[star][l[0]] = float(l[2])
            data[star]['d'+l[0]] = float(l[3][1:-1])
        if line[:3]=="par":
            l = line.split()
            data[star]['parallax'] = float(l[1])
            data[star]['dparallax'] = float(l[2][1:-1])
    return data

def read_interferometric_data(filename=None):
    '''
    Function to read interferometric-determined values of Theta for some stars
    published by Boyajian et al. (2012).
    Input parameters:
    - filename      :   [Optional] path to file. If None, default file is used.
    Ouptut parameters:
    - data          :   [dict] Dictionary with stars ids as keys. Each dict 
                        contains a dictionary with keys 'Radius' and 'dRadius'.
    '''
    if filename is None:
        raise Exception('read_interferometric_data: No input file')
        # filename = paths.irap_tools_path + 'interfero_values.txt'
    f = open(filename)
    lines = f.readlines()
    f.close()
    data = {}
    for line in lines:
        if line[0].lower()=='g':
            l = line.split()
            data[l[0]] = {'Theta': float(l[1]), 'dTheta': float(l[2])}
    return data

def apply_bolometric_correction(M, C, dM=0, dC=0):
    Mc = M + C
    dMc = np.sqrt(dM**2 + dC**2)
    return Mc, dMc

def compute_lum(Mb, dMb):
    '''
    Input parameters:
    bM  :   bolometric magnitude
    '''
    Mb_sun = 4.74 ## Bolometric magnitude Sun
    rL = 10**((Mb - Mb_sun)/(-2.5))
    drL = abs(np.log(10)/(-2.5) * 10**((Mb - Mb_sun)/(-2.5)) * dMb)
    return rL, drL
