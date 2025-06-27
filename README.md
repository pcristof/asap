# A Spectral Analysis Pipeline (ASAP) version 0.1
#### Author: [Paul I. Cristofari](https://paulcristofari.github.io/) <br/> Contact: cristofari@strw.leidenuniv.nl

If you use this package, please cite [Cristofari et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.1342C/abstract)

Some versions of this package were also used in:
- [Cristofari et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.1893C/abstract)
- [Cristofari et al. (2022b)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.3802C/abstract)
- [Cristofari et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.1342C/abstract)
- [Cristofari et al. (2023b)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.5648C/abstract)

## INSTALATION

### Notes:
1. I highly recommend using anaconda[https://www.anaconda.com/download] or miniconda[https://www.anaconda.com/docs/getting-started/miniconda/main] to keep a clean and self-contained environment. <br/>
2. If you are unfamiliar with conda, take the time to read about the basic usage. <br/>
3. I have put some effort in streamlining the installation process, which relies on a setup.py file to ensure installation of dependencies. The process was tested on a macbook M4 Pro running MacOS 15.5. <br/>

### Step-by-step installation instructions:
1. Create a new conda environement [recommended]: `conda create --name asap` <br/>
2. Go to the directory containing the source code, e.g.: `cd ~/usr/softs/asap_v0.1` <br/>
3. Activate the environment you created: `conda activate asap` <br/>
*NB: You will need to have the environment 'asap' activated everytime you want to run ASAP.* <br/>
4. Make sure you have installed pip **in the conda environment** [recommended].  <br/>  &emsp;&nbsp;&nbsp; You will likely need to run `conda install pip` <br/>
5. Install the packages and dependencies by running: `pip install .` <br/>

***You're all set !***  

## QUICK START GUIDE

### Notes:
I put a lot of effort in making this program easy to run, which is no easy task given the quantity of options we need to implement. The core of the program is object-oriented, bundled into the package, and runs with a script accessible with the command `python -m asap`. Once the package is properly installed and the user activated the proper conda environment, the user can run the program from anywhere on their machine. The analysis options are controlled with a single `config.ini` file, which needs to be placed in the working directory of the anlysis to run.

I add a working example to this guide to present the main parts of the program.

### Working example:
The package includes a working example, minus the necessary data and models. These can be requested by contacting the author.

##### Obtaining the config.ini file
1.  Move to your favorite working directory. <br/>
2. Obtain the example configuration file by running `asap.configure` <br/>

This will create a `config.ini` file. Take a moment to look at this example file. It contains everything that the program needs to know to perform the analysis. You will notice different blocks. 
- The first block "MAIN TRIGGERS", is used to initialize some parameters and decide what will be fit. Most variable names are self-explanatory.  <br/>
*NB1: the filling factors correspond to magnetic field steps of 2kG.*  <br/>
*NB2: If a parameters (e.g. fitFields) is set to False, then the value associated (e.g. fillFactors) is assumed to be exact.* <br/>
*NB3: To avoid a crash of the program due to incorrect user inputs, if the sum of the fillFactors is not equal to 1, the program will default the values*
- The "ATMOSPHERIC PARAMETERS" block controls which atmospheric parameters to fit and the initial guess / default values. The arrays (e.g. teffArray) define the dimensions of your precomputed grid. Be sure to have these values set properly, as the program will fail if they are not properly set.
- The "MCMC OPTIONS" block contains the options related to the MCMC run, such as the number of walkers (nbWalkers) and the number of steps (nbSteps). If "parallel" is set to "True", then the program will run the MCMC in parallel using "nbCores" threads.
- If "saveBackend" is set to True, the program will store the chain in a "backend.h5" file. This allows to continue the chain (say you ran for 1000 steps and want to run for 1000 more). Note that the "backend.h5" file can become heavy, and that changing the configuration file before continuing the chain may result in program crashes.
- The "PATHS" block defines the paths that are necessary for the program to run. They can be relative or absolute [recommended].
- The "OPTIONAL TRIGGERS" are advanced options. They typically should not be changed, unless you are debugging the code.

##### Preparing config.ini for your analysis
3. Modify the config.ini file so that the 'pathToGrid' and 'pathToData' point to the correct directories on your local computer. 

##### Running the analysis
4. You can now run the anlysis by simply typing `python3 -m asap dotau`. This will create a new direcotry 'output_dotau' containing the results of the analysis and a copy of the configuration file.

### Understanding your results files
You should now see a new directory 'output_dotau'. The directory will contain a number of figures whose names and should be self-explanatory:
- a0-b.pdf
- b-distrib.pdf
- b_histogram.pdf
- corner.pdf
- samples.pdf
- samples_postburn.pdf

Besides the figures, you get several data files:
- fit-data.fits: contains the spectrum and fits for optimal parameters obtained by the program. Makes it easy to make quick plots (see below) !
- log_prob_walkers_noflat.npy: contains the log_prob for each walkers
- samples.npy: contains the sample chain for all walkers
- tau.npy: sampler data used to compute autocorrelation time.
- weights.npy: should currently be ignored - used for current developments in preparation of the next version of ASAP.

Finally, you get text files containing the parameters:
- results_raw.txt: contains all the parameters obtained from posterior distributions of the MCMC.
- factors.txt: should be ignore -- depricated, will be removed in the next version of ASAP.

#### Using the fit-data.fits file

The fit-data.fits file contains multiple HDU. The WVL, FLUX, FLUXFIT and ERROR cards contain the observation data in a 2D matrix. Note that the first dimension does not represent the orders, but the regions that we used for the fit. The FIT and FITNOMAG contain the obtained model and corresponding non-magnetic model, broadening and rebinned on the WVL solution. IDXTOFIT contains indexing of the bins that were used to obtain the fit. 

Example:
```
>>  hdu.info()
Filename: fit-data.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU       6   ()      
  1  WVL           1 ImageHDU         8   (401, 30)   float64   
  2  FLUX          1 ImageHDU         8   (401, 30)   float64   
  3  FLUXFIT       1 ImageHDU         8   (401, 30)   float64   
  4  ERROR         1 ImageHDU         8   (401, 30)   float64   
  5  FIT           1 ImageHDU         8   (401, 30)   float64   
  6  FITNOMAG      1 ImageHDU         8   (401, 30)   float64   
  7  IDXTOFIT      1 ImageHDU         8   (3286, 2)   int64   
```

With python, one can obtain a very simple plot of these different variables with something like:
```
from astropy.io import fits
import matplotlib.pyplot as plt
hdu = fits.open('fit-data.fits')

plt.figure()
plt.plot(hdu['WVL'].data.T, hdu['FLUX'].data.T, color='black')
plt.plot(hdu['WVL'].data.T, hdu['FLUXFIT'].data.T, color='red')
plt.plot(hdu['WVL'].data.T, hdu['FIT'].data.T, color='green')
idxtofit = tuple(hdu['IDXTOFIT'].data)
plt.plot(hdu['WVL'].data[idxtofit], hdu['FIT'].data[idxtofit], color='purple')
plt.show()
```

#### Reading the results_raw.txt file

The results_raw.txt file contains results extracted from the posterior distributions. They are not in an easy-to-read format. Lets break it down line by line:

1. Contains the filling factors (including 0kG).
2. Contains the error bars on the filling factors
3. Contains Teff log(g) [M/H] and [a/Fe]
4. Contains the *formal* error bars on Teff log(g) [M/H] and [a/Fe]
5. The "mean" magnetic field obtained by looking at the posterior distribution total magnetic fields of the walkers (this is not obtained by computing a0*S0+a2*S2...). First value after the column is \<B\>, second is the error
6. "average" magnetic field obtained by computing a0*S0+a2*S2...; first value is  \<B\>, second is the propagated error.
7. vb: *additional* gaussian broadening. Depricated.
8. GussRV: Guess or the radial velocity obtained with a fast and approximate cross correlation at the beginning of ASAP. Used to recenter the observation spectra and avoid "out of bounds" problems.
9. RV: Radial velocity obtained from the posterior distribution *relative* to GussRV. With this one, the synthetic spectra are shifted.
10. vsini and error on vsini
11. vmac and error.
12. min. chi2 obtained **after** renormalization of the error bars (see line 15).
13. same as line 12 but setting taking a non-magnetic spectrum for the same atmospheric and broadening parameters.
14. Nb. of points used to perform the fit
15. Normalization factor used to rescale the error bars. This factor was computed at the end of the previous run**.
16. veiling factors ***
17. errors on veiling
18. placeholder for bolometric luminosity
19. placeholder for absolute K band magnitude
20. placeholder for distance
21. placeholder (ignore)
22. placeholder (ignore)
23. placeholder (ignore)
24. placeholder (ignore)
25. placeholder (ignore)
26. placeholder (ignore)
27. placeholder (ignore)
28. Error type mode (depricated, for advanced users and debugging)
29. vinstru: instrumentation width provided by the user or defaulted.

*****Notes on normFactor:*** Remember to run the program *at least* twice to obtain use renormalized error bars.

******Notes on Veiling:*** Implementation of veiling with multiple band is not trivial. Veiling currently implemented by assuming constant veiling factors throughout bands. Bands are defined by a central wavelength and a FWHM (currently hardcoded, sorry). ASAP currently handles bands I, Y, J, H, K and L. transition between bands are currently accounted for by interpolating the veiling factors between the edges of consecutive bands. To handle edges, the current implementation allows to provide "nan" to the bands at the edges of the considered domain. If nan is provided, the code will impose the value of the band at the edge to be that of the closest band. This allowed to fix issues with, e.g., very few lines selected in a band leading to bad veiling estimates.

### Understanding my input data and format

#### Synthetic spectra
ASAP reads grids of ZeeTurbo (or other!) spectra in a very specific format. The choice of hdf5 format was motivated by 1- the easy link to a unique wavelength solution file reducing disk space usage and 2- the highly efficient IO that h5py offers. The working example comes with a very minimal grid for storage reasons.
 
#### Observation data
ASAP reads spectra store in a FITS file. Note that the program was designed to handle spectra with multiple orders.
Your input FITS files should contain a HDUList with 3 *ordered* cards **after** the PrimaryHDU, containing the 2D wavelength, normalized fluxes, and errors. Currently, no other format is adequately supported (see example below).

```
>> hdu.info()
Filename: dotau_templates.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU       4   ()      
  1  WVL           1 ImageHDU         8   (4088, 49)   float64   
  2  TEMPLATE      1 ImageHDU         8   (4088, 49)   float64   
  3  ERR           1 ImageHDU         8   (4088, 49)   float64   
```
