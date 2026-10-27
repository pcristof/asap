from numba import jit
import numpy as np
from importlib.resources import files
from asap import analysis_tools as tls
import glob

'''Functions used to read HITRAN files'''

def id_molecule(i):
    '''
    Returns the molecule name associated to the HITRAN molecule id.
    Molecules ID are hardcoded based on the HITRAN query webpage.
    Caution :  NOT ALL MOLECULES ARE IMPLEMENTED
    
    Input parameters:
    - i     :   [int] Hitran moelecule id.
    
    Output parameters:
    - mol   :   [str] Name of the molecule associated to the id, in lowercase.
    '''
    if i==1:
        mol = 'h2o'
    elif i==2:
        mol = 'co2'
    elif i==3:
        mol = 'o3'
    elif i==4:
        mol = 'n2o'
    elif i==6:
        mol = 'ch4'
    elif i==7:
        mol = 'o2'
    elif i==10:
        mol = 'no2'
    else:
        raise Exception('HITRAN molecule id not in store.')
    return mol

# filename = paths.irap_tools_data_path+'tapas_linelist_files/5f69962e.out'
def read_hitran_linelist(filename):
    f = open(filename)
    lines = []
    for line in f.readlines():
        values = line.split()
        mol_id = int(values[0])
        wavenumber = float(values[3])
        intensity = float(values[4])
        ## Convert wavenumber to nanometers wavelength
        wavelength = 1/wavenumber*1e7 # nm
        ## Convert molecule id to molecule name
        mol_name = id_molecule(mol_id)
        ## Append tuple to list
        lines.append((mol_name, wavelength, intensity))
    rec = np.array(lines[::-1], dtype=[('mol', '<U10'), 
                                ('wavelength', '<f8'), ('intensity', '<f8')])
    return rec

def read_water_lines():
    '''
    Function to create a mask composed of water lines.

    Output parameters:
    - humid_mask    :   Mask containing only the water lines
    '''
    rec = read_hitran_linelist(filename)
    idx = np.where(rec['mol']=='h2o')
    humid_mask = rec[idx]
    return humid_mask['wavelength'], humid_mask['intensity']
    
def read_dry_lines():
    '''
    Function to create a mask composed of dry lines (not water).

    Output parameters:
    - dry_mask      :   Mask containing only the dry components lines.
    '''
    rec = read_hitran_linelist(filename)
    idx = np.where(rec['mol']!='h2o')
    dry_mask = rec[idx]
    return dry_mask['wavelength'], dry_mask['intensity']

'''Functions to read the skylines mask'''

def read_sky_lines(filename=None):
    if filename is None:
        raise Exception('read_sky_lines: No input file')
        # filename = paths.irap_tools_data_path+'list_OH_v2.0.dat'
    wvl, depths = np.loadtxt(filename, unpack=True, skiprows=30)
    return wvl/10, depths#*0+1

def read_perso_mask(filename=None):
    '''This function reads a line list mask for CCF uses. file must contain
    wvl in first column and depth is second column.'''
    if filename is None:
        raise Exception('read_perso_mask: No input file')
        # f = open(paths.irap_tools_data_path+'perso_mask', 'r')
    else:
        f = open(filename, 'r')
    wavelengths = []
    depths = []
    for i,line in enumerate(f.readlines()):
        if line.strip()[0]=='#': 
            continue
        elif len(line.split())>=2:
            wavelengths.append(line.split()[0].strip())
            depths.append(line.split()[1].strip())
    return np.array(wavelengths, dtype=float), np.array(depths, dtype=float)

def read_vald_lines(filename=None, teff=None, wave_bounds=[0, np.inf]):

    ## Read the first line of all the files to short the data:
    filenames = files("asap.support_data").joinpath(f'vald-lines_*')
    fnames = glob.glob(str(filenames))
    data_files = {}
    teffs = []; wave_ranges = []
    for fname in fnames:
        with open(fname, 'r') as f:
            ## First line of the file contains wave boundaries
            line = f.readline()
            sl = line.split(',')
            wave_start = float(sl[0]); wave_end = float(sl[1])
            nlines = int(sl[2]) ## Number of lines
            ## Get unit and medium:
            line = next(line for n, line in enumerate(f, 1) if n == 2)
            if 'WL_vac' in line:
                units=line.split('WL_vac(')[1][0].lower()
                medium='vac'
            elif 'WL_air':
                units=line.split('WL_air(')[1][0].lower()
                medium='air'
            if units=='a': ## To nanometers
                wave_start = wave_start/10
                wave_end = wave_end/10
            if medium=='air': ## To vacuum
                wave_start = tls.convert_lambda_in_vacuum(wave_start*10)/10 
                wave_end = tls.convert_lambda_in_vacuum(wave_end*10)/10 
            ## Get the model atmosphere used
            line = next(line for n, line in enumerate(f, 1) if n == (nlines+1))
            if '_T' in line:
                _t = int(line.strip().split('_T')[1].split('G')[0]) ## Temperature
            elif line.replace("'", "").strip()[0]=='p':
                _t = int(line.replace("'", "").strip()[1:].split('_')[0])
            else:
                raise Exception(f'Could not read vald file model: {line}')
            teffs.append(_t); wave_ranges.append([wave_start, wave_end])
    teffs = np.array(teffs); wave_ranges = np.array(wave_ranges)
    ## Now select the correct file

    wavelengths_in_angstrom = False
    
    absres = np.abs(teffs-teff)
    idx = np.where(absres==np.min(absres))[0] ## Can have multiple values
    fnames_keep = []
    for i in idx:
        if (wave_ranges[i][1]>wave_bounds[0]>wave_ranges[i][0])\
           | (wave_ranges[i][1]>wave_bounds[1]>wave_ranges[i][0]):
           fnames_keep.append(fnames[i])

    # if filename is None:
    #     if teff is None: 
    #         filename = files("asap.support_data").joinpath("vald-lines")
    #     else:
    #         try: 
    #             teff=int(teff); 
    #         except: 
    #             raise Exception('mask_tools.read_vald_line() teff must be int or None.')
    #         filename = files("asap.support_data").joinpath(f"vald-lines_300-900_{teff}_short.txt")
    #         wavelengths_in_angstrom = True
    #     # f = open(paths.irap_tools_data_path+'vald-lines', 'r')
    # # else:
    elements = []
    wavelengths = []
    depths = []
    for filename in fnames_keep:
        f = open(filename, 'r')
        wavelength_medium='unknown'
        for i,line in enumerate(f.readlines()):
            if i<=2:
                if 'WL_vac' in line: 
                    wavelength_medium='vac'
                    units=line.split('WL_vac(')[1][0].lower()
                elif 'WL_air' in line: 
                    wavelength_medium='air'
                    units=line.split('WL_air(')[1][0].lower()
                continue
            elif len(line.split(','))>2:
                _w = float(line.split(',')[1].strip())
                if units=='a': _w=_w/10 ## To nanometers
                ## Check medium
                if wavelength_medium=='unknown': 
                    raise Exception('Could not detect if WL are air or vac')
                elif wavelength_medium=='air': ## To vacuum
                    _w = tls.convert_lambda_in_vacuum(_w*10)/10 
                    wavelength_medium='vac'
                ## Check if the value is in the requested bounds
                if (_w>wave_bounds[0])&(_w<wave_bounds[1]):
                    elements.append(line.split(',')[0].strip()[1:-1])
                    wavelengths.append(_w)
                    depths.append(line.split(',')[9].strip())
            elif len(line.split(','))<=5:
                break
    if len(wavelengths)==0:
        return None, None
    w = np.array(wavelengths)
    return w, np.array(depths, dtype=float)

'''Fucntions used to generate masks from wavelength grid and line lists'''

@jit(nopython=True, cache=True)
def gen_mask_(ref_wvls, wvls, depth):
    M = np.zeros(len(ref_wvls))
    for i in range(len(ref_wvls)):
        sum_vals = 0
        count = 0
        for j in range(len(wvls)):
            if round(ref_wvls[i],3)==round(wvls[j],3):
                sum_vals += depth[j]
                count += 1
        if count==0:
            count = 1
        M[i] = sum_vals / count
    return M

@jit(nopython=True, cache=True)
def gen_mask(ref_wvls, wvls, depth, dspeed=1):
    M = np.zeros(len(ref_wvls))
    for i in range(len(ref_wvls)):
        sum_vals = 0
        count = 0
        val = 0
        for j in range(len(wvls)):
            # if round(ref_wvls[i],3)==round(wvls[j],3):
            dwvl = wvls[j] * dspeed * 1000 / 3e8 # 1km/s
            if (wvls[j]>(ref_wvls[i]-dwvl)) & (wvls[j]<(ref_wvls[i]+dwvl)):
                sum_vals += depth[j]
                if val < abs(depth[j]):
                    val = depth[j]
                    # print(depth[j])
                count += 1
        if count==0:
            count = 1
        M[i] = sum_vals / count # This could be replaced by a max abs. depth? 
        # M[i] = val # This could be replaced by a max abs. depth?
        # print(sum_vals)
    return M

@jit(nopython=True, cache=True)
def gen_mask_gauss(ref_wvls, wvls, depth, dspeed=1):
    '''Like gen_mask but with ponderation of the wavelengths'''
    M = np.zeros(len(ref_wvls))
    mastercount = 0
    for i in range(len(ref_wvls)):
        sum_vals = 0
        count = 0
        val = 0
        for j in range(len(wvls)):
            # if round(ref_wvls[i],3)==round(wvls[j],3):
            dwvl = wvls[j] * dspeed * 1000 / 3e8 # 1km/s
            if (wvls[j]>(ref_wvls[i]-dwvl)) & (wvls[j]<(ref_wvls[i]+dwvl)):
                # weight = 1/dspeed*np.exp(-((wvls[j] - ref_wvls[i])/ref_wvls[i]* 1000 / 3e8)**2/(2*(dspeed)**2))
                deltawvl = abs(wvls[j] - ref_wvls[i])
                deltaspeed = deltawvl / ref_wvls[i] * 3*1e5
                weight = 1 - deltaspeed / (2*dspeed)
                sum_vals += depth[j]*weight
                if val < abs(depth[j]):
                    val = depth[j]
                    # print(depth[j])
                count += 1
                mastercount += 1
        if count==0:
            count = 1
        M[i] = sum_vals / count # This could be replaced by a max abs. depth? 
        # M[i] = val # This could be replaced by a max abs. depth?
        # print(sum_vals)
    return M

@jit(nopython=True, cache=True)
def gen_mask_gauss_2d(ref_wvls, wvls, depth, dspeed=1):
    '''Like gen_mask but with ponderation of the wavelengths'''
    mask = np.zeros(ref_wvls.shape)

    for order in range(len(mask)):
        mask[order] = gen_mask_gauss(ref_wvls[order], wvls, depth, 
                                     dspeed=dspeed)

    return mask

@jit(nopython=True, cache=True)
def gen_mask_2d(ref_wvls, wvls, depth, dspeed=1):
    M = np.zeros(ref_wvls.shape)
    for r in range(len(ref_wvls)):
        m = gen_mask(ref_wvls[r], wvls, depth, dspeed)
        M[r] = m
    return M

def gen_mask_deriv(spectrum):
    '''Function to build mask from derivative of inputted spectrum'''
    mdrydiff = np.diff(spectrum)
    mdrydiff[mdrydiff<0] = -1
    mdrydiff[mdrydiff>0] = +1
    mdrydiff = np.diff(mdrydiff)
    idx = np.where(mdrydiff==-2)
    idx = idx[0]
    idx += 1
    mm = np.zeros(spectrum.shape)
    mm[(idx)] = 1
    mask = spectrum * mm
    return mask

def gen_mask_deriv2d(spectrum):
    mask      = np.zeros(spectrum.shape)
    for r in range(len(spectrum)):
        mask[r] = gen_mask_deriv(spectrum[r])
    return mask
