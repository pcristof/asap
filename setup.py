from setuptools import setup, find_packages

setup(
    name='asap',
    version='0.1',    # Initial version
    author='Paul I. Cristofari',
    author_email='paul.ivan.cristofari@gmail.com',
    description='A Spectra Analysis Pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',  # Replace with your repo URL
    packages=find_packages(),  # Automatically find package directories
    include_package_data=True,  # Include package data files
    package_data={
        "asap": ["resources/config.ini", 
                 "support_data/*",
                 "support_data/blaze_data/*",
                 "support_data/ref_params/*",],  # Specify the file(s) to include
    },
    python_requires='>=3.6',  # Specify the Python version
    install_requires=[  # Dependencies that will be installed automatically
        "numpy",
        "astropy",
        "matplotlib",
        "numba",
        "emcee",
        "ipython",
        "astroquery",
        "h5py",
        "dynesty",
        "ultranest",
        "corner",
        "tqdm",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'asap.configure=asap.helper_tools:configure',  # Command-line utility
            # 'asap.run_analysis=asap.scripts.run_analysis:main',  # Command-line utility
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
