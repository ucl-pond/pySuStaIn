pySuStaIn
============

**Su**btype and **St**age **In**ference, or SuStaIn, is an algorithm for discovery of data-driven groups or "subtypes" in chronic disorders. This repository is the Python implementation of SuStaIn, with the option to describe the subtype progression patterns using either the event-based model, the piecewise linear z-score model or the scored events model.

Acknowledgement
================
If you use pySuStaIn, please cite the following core papers:
1. [The original SuStaIn paper](https://doi.org/10.1038/s41467-018-05892-0)
2. [The pySuStaIn software paper](https://doi.org/10.1016/j.softx.2021.100811)

Please also cite the corresponding progression pattern model you use:
1. [The piecewise linear z-score model (i.e. ZscoreSustain)](https://doi.org/10.1038/s41467-018-05892-0)
2. [The event-based model (i.e. MixtureSustain)](https://doi.org/10.1016/j.neuroimage.2012.01.062) 
   with [Gaussian mixture modelling](https://doi.org/10.1093/brain/awu176) 
   or [kernel density estimation](https://doi.org/10.1002/alz.12083)).
3. [The scored events model (i.e. OrdinalSustain)](https://doi.org/10.3389/frai.2021.613261)   
   
Thanks a lot for supporting this project.

Installation
============
## Install option 1 (for installing the pySuStaIn code in a chosen directory): clone repository, install locally

1) [Clone this repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)

2) Navigate to the main pySuStaIn directory (where you see setup.py, README.txt, LICENSE.txt, and all subfolders), then run:

       pip install .

    Alternatively, you can do `pip install -e .` where the `-e` flag allows you to make edits to the code without reinstalling.

Either way, it will install everything listed in `requirements.txt`, including the [awkde](https://github.com/mennthor/awkde) package (used for mixture modelling). During the installation of `awkde`, an error may appear, but then the installation should continue and be successful. Note that you need `pip` version 18.1+ for this installation to work.

## Install option 2 (for simply using pySuStaIn as a package): direct install from repository

1) Run the following command to directly install pySuStaIn:

       pip install git+https://github.com/ucl-pond/pySuStaIn

Note that if you must already have numpy (1.18+) installed to do this. To create a new environment, follow the instructions in the [Troubleshooting](#troubleshooting) section below.

Troubleshooting
============
If the above install breaks, you may have some interfering packages installed. One way around this would be to create a new [Anaconda](https://www.anaconda.com) environment that uses Python 3.7+, then activate it and repeat the installation steps above. To do this, download and install Anaconda/Miniconda, then run:

```
conda create  --name sustain_env python=3.7
conda activate sustain_env
conda install numpy
```

To create an environment named `sustain_env` and install numpy. Then, follow the installation instructions as normal.



Dependencies
============
- Python >= 3.7 
- [NumPy >= 1.18](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Scikit-learn](https://scikit-learn.org) for cross-validation
- [kde_ebm](https://github.com/noxtoby/kde_ebm_open) for mixture modelling (KDE and GMM included)
- [pathos](https://github.com/uqfoundation/pathos) for parallelization
- [awkde](https://github.com/mennthor/awkde) for KDE mixture modelling

Testing
===============
If you want to check that the installation was successful, you can run the end-to-end tests. For this, you will need to navigate to the `tests/` subfolder (wherever pySuStaIn has been installed on your system). Then, you can use the following command to run all SuStaIn variants (this may take a bit of time!):

```
python validation.py -f
```

For a quicker run (using just `MixtureSustain`), just use:
```
python validation.py
```
instead. Testing of single classes is possible using the `-c` flag, e.g. `python validation.py -c ordinal`. To see all options, run `python validation.py --help`.


Parallelization
===============
- Added parallelized startpoints

Running different SuStaIn implementations
===============
sustainType can be set to:
  - `mixture_GMM` : SuStaIn with an event-based model progression pattern, with Gaussian mixture modelling of normal/abnormal.
  - `mixture_KDE`:  SuStaIn with an event-based model progression pattern, with Kernel Density Estimation (KDE) mixture modelling of normal/abnormal.
  - `zscore`:       SuStaIn with a piecewise linear z-score model progression pattern.
  
 See `simrun.py` for examples of how to run these different implementations.

SuStaIn Tutorial
===============  
See the jupyter notebook in the notebooks folder for a tutorial on how to use SuStaIn using simulated data.
We also have a set of tutorial videos on YouTube, which you can find [here](https://www.youtube.com/watch?v=5CFsfFcVzEc&list=PL25fUWY3exLxYSPOnEe60kSEh0JRdVVPB).

Papers
============
Methods:
- The SuStaIn algorithm: [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0) 
- The pySuStaIn software paper: [Aksman, Wijeratne et al. 2021](https://doi.org/10.1016/j.softx.2021.100811)
- The event-based model: [Fonteijn et al. 2012](https://doi.org/10.1016/j.neuroimage.2012.01.062), (with Gaussian mixture modelling [Young et al. 2014](https://doi.org/10.1093/brain/awu176) or non-parametric kernel density estimation [Firth et al. 2020](https://doi.org/10.1002/alz.12083))
- The piecewise linear z-score model: [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0) 
- The scored events model ('Ordinal SuStaIn'): [Young et al. 2021](https://doi.org/10.3389/frai.2021.613261)  


Applications:
- Multiple sclerosis (predicting treatment response): [Eshaghi et al. 2021](https://doi.org/10.1038/s41467-021-22265-2). The trained model is available [here](https://github.com/armaneshaghi/trained_models_MS_SuStaIn). 
- Tau PET data in Alzheimer's disease: [Vogel et al. 2021](https://doi.org/10.1038/s41591-021-01309-6)
- COPD: [Young and Bragman et al. 2020](https://doi.org/10.1164/rccm.201908-1600OC)
- Frontotemporal dementia: [Young et al. 2021](https://doi.org/10.1212/WNL.0000000000012410)

Funding
================
This project has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under Grant Agreements 666992. Application of SuStaIn to multiple sclerosis was supported by the International Progressive MS Alliance (IPMSA, award reference number PA-1603-08175).

Quotes
============
> _(The authors) have also persuaded me that (SuStaIn is) as clever as e.g. Heiko Braak's brain, (and) can infer longitudinal trajectories based on cross-sectional observations._
> - Anonymous reviewer
