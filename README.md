# A Spectral Analysis Pipeline (ASAP) version 0.1
#### Author: Paul I. Cristofari <br/> Contact: paul.ivan.cristofari@gmail.com

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
I put a lot of effort in making this program easy to run, which is no easy task given the quantity of options we need to implement. The core of the program is object-oriented, buddle into the package, and runs with a script accessible with the command `asap.run_analysis.py`. Once the package is properly installed and the user activated the proper conda environment, the user can run the program from anywhere on their machine. The analysis options are controlled in a single file `config.ini` that needs to be placed in the working directory of the anlysis to run.

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

##### Run the analysis
4. You can now run the anlysis by simply typing `python3 -m asap dotau`. This will create a new direcotry 'output_dotau' containing the results of the analysis and a copy of the configuration file.

## UNDERSTANDING YOUR RESULT FILES