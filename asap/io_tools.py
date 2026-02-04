from astropy.io import fits
import os

def interpret_file_format(filename):
    '''Function to load a template observation (version built by P. I. 
    Cristofari) from fits file.
    Input:
    - filename      :   [string] absolute path of the file to load.'''

    ## Open fits file
    mode = 'unknown'
    ## Resoving file type:
    # if "p.fits" in filename:
    #     ## Check that the file is compatible with the file format
    ispfits = True
    with fits.open(filename) as hdu:
        if len(hdu)>=9:
            keys = [hdu[i].name for i in range(len(hdu))]
            for name in ['PRIMARY', 'Pol', 'PolErr', 'StokesI', 'StokesIErr',
                            'Null1', 'Null2', 'WaveAB', 'BlazeAB']:
                if name not in keys:
                    ## This is not a p.fits
                    ispfits = False
        else:
            ispfits = False
    ispoloformat = False
    if not ispfits:
        with fits.open(filename) as hdu:
            if (len(hdu)==4) | (len(hdu)==5):
                shape0 = hdu[1].data.shape
                for i in range(2, len(hdu)):
                    if hdu[i].shape!=shape0:
                        ispoloformat = False
                        break
                    else:
                        ispoloformat = True

    if ispfits:
        mode='p.fits'
    elif ispoloformat:
        mode='poloformat' 

    return mode

import glob
def interactive_list_file(_pathtodata):
    if _pathtodata[-1]!='/': _pathtodata+='/'
    listfiles = glob.glob(_pathtodata+'*.fits')
    list_valid_files = []
    modes = []
    for name in listfiles:
        ## Check if file is valid:
        mode = interpret_file_format(name)
        if mode!='unknown':
            list_valid_files.append(name)
            modes.append(mode)
    nbfiles = len(list_valid_files)
    print(f'------------------------')
    print(f'Found {nbfiles} files:')
    for i in range(nbfiles):
        print(f'{i} {list_valid_files[i]} - mode: {modes[i]}')
    choice = input(f"\n Select file (0-{nbfiles}) or 'q' to quit: ").strip()
    the_file = None
    if choice.isnumeric():
        choice = int(choice)
        if (choice>=0) & (choice<nbfiles):
            the_file = list_valid_files[choice]
            print(f"Running for {the_file}")
    if the_file is None:
        print('No file found')
    return the_file