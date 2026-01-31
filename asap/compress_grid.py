import numpy as np
from asap import analysis_tools as tls
import h5py
from astropy.io import fits

class Compressor:

    def __init__(self):
        self.resampleVel = False
        self.bs = 0.
        self.teffs = np.arange(3000, 9000, 100)
        self.loggs = np.arange(3.0, 6.0, 0.5)
        self.mhs = np.arange(-1.5, 2.0, 0.25)
        self.alphas = np.arange(-1.0, 1.5, 0.25)
        self.message = ""
        self.file_struc = '{}g{:0.1f}z{:0.2f}a{:.2f}b{:04.0f}p{:0.1f}'\
                          +'rot{:0.2f}beta{:0.2f}.hdf5'

    def interpret_grid_dimensions(self, pathtogrid):
        '''Function to automatically read the grid limits available
        The function should allow me to obtain arrays of uneven teffs, 
        loggs, etc.'''
        import glob
        filelist = glob.glob(pathtogrid+'*.hdf5')
        if len(filelist)==0:
            return None, None, None, None ## No file found in this format
        ## I need a list of all teffs, loggs, and mhs
        teffs=[]; loggs=[]; mhs=[]; alphas=[]; bvals=[]
        for file in filelist:
            if 'wave.hdf5' in file: continue
            variables = self.parse_filename(file.split('/')[-1])
            if variables['teff'] not in teffs: teffs.append(variables['teff'])
            if variables['logg'] not in loggs: loggs.append(variables['logg'])
            if variables['mh'] not in mhs: mhs.append(variables['mh'])
            if variables['alpha'] not in alphas: 
                alphas.append(variables['alpha'])
            if variables['bval'] not in bvals: bvals.append(variables['bval'])
        teffs.sort(); loggs.sort(); mhs.sort(); alphas.sort(); bvals.sort()
        ## CAUTION: This will break if the grid is not "square"
        teffs=np.array(teffs); loggs=np.array(loggs); 
        mhs=np.array(mhs); alphas=np.array(alphas); bvals=np.array(bvals)
        return teffs, loggs, mhs, alphas, bvals

    def get_grid_dims(self):
        '''Compute the dimensions of the grid used to store the data.
        Dimensions labeled from d1 to d7 are respectively the number of Teff,
        number of log(g), number of [M/H], number of [a/Fe], number of B,
        number of regions (estimated from input region file, initialized to 0),
        and the number of data points in a region.'''
        self.d1 = len(self.teffs); self.d2 = len(self.loggs); 
        self.d3 = len(self.mhs); self.d4 = len(self.alphas); 
        # self.d5 = len(self.bs);
        self.d7 = 0 ## Initializization
        self.griDims = (self.d1, self.d2, self.d3, self.d4, self.d7)
        #,                 self.d5, self.d7)
    
    def parse_filename(self, fname):
        # '{}g{:0.1f}z{:0.2f}a{:.2f}b{:04.0f}p{:0.1f}rot{:0.2f}beta{:0.2f}.hdf5'
        variables = {}
        tmpname = fname.split('g')
        variables['teff'] = float(tmpname[0])
        tmpname = fname.split('g')[1].split('z')
        variables['logg'] = float(tmpname[0])
        tmpname = fname.split('z')[1].split('a')
        variables['mh'] = float(tmpname[0])
        tmpname = fname.split('a')[1].split('b')
        variables['alpha'] = float(tmpname[0])
        tmpname = fname.split('b')[1].split('p')
        variables['bval'] = float(tmpname[0])
        tmpname = fname.split('p')[1].split('rot')
        variables['phase'] = float(tmpname[0])
        tmpname = fname.split('rot')[1].split('beta')
        variables['rot'] = float(tmpname[0])
        tmpname = fname.split('beta')[1].split('.hdf5')
        variables['beta'] = float(tmpname[0])
        return variables

    ###################################
    #### ---- LOAD MODEL GRID ---- ####
    ###################################
    def load_grid(self, pathtogrid, bval):
        '''Load a grid of models for all mag field strengths'''

        B = bval

        ## Ensures the path ends with a '/'
        if pathtogrid[-1]!='/': pathtogrid+='/'
        _t,_l,_m,_a,_b = self.interpret_grid_dimensions(pathtogrid)
        ## Adjust arrays so that we take the true values in the grid 
        ## and only use the min and max
        # _tl = self.teffs[0]; _th = self.teffs[-1]
        # _ll = self.loggs[0]; _lh = self.loggs[-1]
        # _ml = self.mhs[0]; _mh = self.mhs[-1]
        # _al = self.alphas[0]; _ah = self.alphas[-1]
        ## Just take the whole grid
        _tl = -np.inf; _th = np.inf
        _ll = -np.inf; _lh = np.inf
        _ml = -np.inf; _mh = np.inf
        _al = -np.inf; _ah = np.inf

        idx = (_t>=_tl) & (_t<=_th)
        self.teffs = _t[idx]
        idx = (_l>=_ll) & (_l<=_lh)
        self.loggs = _l[idx]
        idx = (_m>=_ml) & (_m<=_mh)
        self.mhs = _m[idx]
        idx = (_a>=_al) & (_a<=_ah)
        self.alphas = _a[idx]
        message_line = "/!\ Grid dimensions were adjusted: " \
                       +"bypassing user requested grid"
        self.message += message_line
        print(message_line)
        ## Make sure you get the new grid dimensions:
        self.get_grid_dims()
        ## Read this grid
        print('Loading grid')
        wgrid = np.zeros((self.d1, self.d2, self.d3, 
                          self.d4)).tolist()
        # wvls = np.zeros((1)).tolist()
        grid = np.zeros([self.d1, self.d2, self.d3, 
                          self.d4]).tolist()
        
        ntot = self.d1*self.d2*self.d3*self.d4
        n = 0
        teffs_int = np.array(self.teffs, dtype=int)
        for it, teff in enumerate(teffs_int):
            for il, logg in enumerate(self.loggs):
                for im, mh in enumerate(self.mhs):
                    for ia, alpha in enumerate(self.alphas):
                        # for ib, B in enumerate(self.bs):
                        try:
                            ## Try to read the hdf5 file:
                            ## HARCODE phase and rot and beta are hardcoded
                            ## TODO: REMOVE HARDCODE
                            ztfile_struct = self.file_struc
                            phase = 0.0; rot=90.; beta=0.
                            ztfile = ztfile_struct.format(teff, logg, mh, 
                                                            alpha, B*1000, 
                                                            phase, rot, beta)
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
                        # We add an option to resample the grid on a 
                        # wavelength solution constant in speed.
                        if self.resampleVel:
                            w, s = \
                            tls.resample_vel_interp(w, s, kind='cubic')
                        #
                        wgrid[it][il][im][ia] = w
                        grid[it][il][im][ia] = s
                        wvls = w
                        n += 1
                        stat = n / ntot * 100
                        print("Reading... {:0.2f} %".format(stat), 
                                                        end='\r')

        grid = np.array(grid, dtype=float)
        self.nwvls = wgrid
        print('Done grid')
        return wgrid, grid, self.teffs, self.loggs, self.mhs, self.alphas