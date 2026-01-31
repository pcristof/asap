from setuptools import setup, find_packages

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="asap.c_tools.linsolve",
        sources=["asap/c_tools/linsolve.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="asap.c_tools.interpolate_4d",
        sources=["asap/c_tools/interpolate_4d.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="asap.c_tools.normalization_tools",
        sources=["asap/c_tools/normalization_tools.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="asap.c_tools.effects",
        sources=["asap/c_tools/effects.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="asap.c_tools.spectral_broadening",
        sources=["asap/c_tools/spectral_broadening.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="asap.c_tools.disk_integration",
        sources=["asap/c_tools/disk_integration.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name='asap',
    version='0.1',    # Initial version
    author='Paul I. Cristofari',
    author_email='paul.ivan.cristofari@gmail.com',
    description='A Spectra Analysis Pipeline',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pcristof/asap',  # Replace with your repo URL
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
        "PyAstronomy"
    ],
    entry_points={
        'console_scripts': [
            'asap.configure=asap.helper_tools:configure',  # Command-line utility
            'asap.gen_synth_obs=asap.scripts.gen_synth_obs:main',  # Command-line utility
            'asap.convert_ov_observations=asap.scripts.convert_ov_observations:main',
            'asap.plot_corner=asap.scripts.plot_corner:main',
            'asap.pca_compress_zeeturbo_grid='\
            +'asap.scripts.pca_compress_zeeturbo_grid:main',
            # 'asap.run_analysis=asap.scripts.run_analysis:main',  # Command-line utility
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
