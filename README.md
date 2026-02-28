# pySuStaIn

**Su**btype and **St**age **In**ference, or SuStaIn, is an algorithm for discovery of data-driven groups or "subtypes" in chronic disorders. This repository is the Python implementation of SuStaIn, with the option to describe the subtype progression patterns using either the event-based model, the piecewise linear z-score model or the scored events model.


---

> _(The authors) have also persuaded me that (SuStaIn is) as clever as e.g. Heiko Braak's brain, (and) can infer longitudinal trajectories based on cross-sectional observations._
> - Anonymous reviewer


---

# Acknowledgement: how to cite SuStaIn and pySuStaIn

If you use pySuStaIn, please cite the following core papers:

1. The original SuStaIn paper: [Young, et al., Nature Communications 2018](https://doi.org/10.1038/s41467-018-05892-0)
2. The pySuStaIn software paper: [Askman and Wijeratne, et al., SoftwareX 2021](https://doi.org/10.1016/j.softx.2021.100811)

Please also cite the corresponding disease progression model you use:

1. **ZscoreSustain**: piecewise linear z-score model - [Young, et al., Nature Communications 2018](https://doi.org/10.1038/s41467-018-05892-0)
  - **ZScoreSustainMissingData**: piecewise linear z-score model with missing data handling - [Estarellas, et al., Brain Communications 2024](https://doi.org/10.1093/braincomms/fcae219)
2. **MixtureSustain**: the event-based model - [Fonteijn, et al., NeuroImage 2012](https://doi.org/10.1016/j.neuroimage.2012.01.062)
  - with Gaussian mixture modelling: [Young, et al., Brain 2014](https://doi.org/10.1093/brain/awu176)
  - or Kernel Density Estimation mixture modelling: [Firth, et al., Alzheimer's & Dementia 2020](https://doi.org/10.1002/alz.12083)
3. **OrdinalSustain**: the scored events model - [Young, et al., Frontiers AI 2021](https://doi.org/10.3389/frai.2021.613261)

Thanks in advance for acknowledging our work correctly.

---

# Installation

We strongly recommend using a virtual environment manager like conda (see [Troubleshooting](#Troubleshooting) below).

## Install option 1 (for installing the pySuStaIn code in a chosen directory): clone repository, install locally

1. Clone this repo ([how to clone a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))
2. Navigate to the main pySuStaIn directory (where you see README.md, LICENSE.txt, and all subfolders), then run:
```
pip install .
```
or
```
pip install '.[kdeebm]'
```
if you wish to also install the [`kde_ebm`](https://github.com/ucl-pond/kde_ebm) and [`awkde`](https://github.com/mennthor/awkde) packages for event-based modelling.

Alternatively, you can do `pip install -e .` where the `-e` flag allows you to make edits to the code without reinstalling.

Either way, it will install everything listed in `pyproject.toml`.


### Possible errors during installation

- In previous versions, an error/warning may occur during the installation of `awkde` but then the installation _should_ continue and be successful.

## Install option 2 (for simply using pySuStaIn as a package): direct install from repository

1. Run the following command to directly install pySuStaIn:
```
pip install git+https://github.com/ucl-pond/pySuStaIn
```

---

# Troubleshooting

There may exist clashes between python packages. A sensible solution is to use a virtual environment such as [Anaconda](https://www.anaconda.com) to manage a clean set of working dependencies. Previously pySuStaIn versions required python version 3.7 or greater, but the current version works for python 3.14.

We have included a conda environment file to assist with this: `py314sustain.yml`. Tested on an M3 Mac in March 2026.

To do this, download and install Anaconda/Miniconda, then run:
```
conda env create -f py314sustain.yml
conda activate py314sustain
```
then install `pySuStaIn` (with or without `kde_ebm`) as described above.


---

# Dependencies

Based on our testing in March 2026, pySuStaIn requires the following packages:

- Python >= 3.14, might work on versions as old as 3.7
- [NumPy](https://github.com/numpy/numpy) >= 2.4.2
- [SciPy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib) >= 3.10.8
- [scikit-learn](https://scikit-learn.org) for cross-validation
- [pybind11](https://github.com/pybind/pybind11) for C++ / python integration
- [setuptoools](https://pypi.org/project/setuptools/)

- [kde_ebm](https://github.com/ucl-pond/kde_ebm) for mixture modelling (KDE and GMM included)
   - [awkde](https://github.com/mennthor/awkde) for KDE mixture modelling

The example/tutorial notebooks require (among others, possibly):

- [pandas](https://pandas.pydata.org) >=3.0.0
- [tqdm](https://tqdm.github.io)


---

# Testing

If you want to check that the installation was successful, you can run the end-to-end tests. For this, you will need to navigate to the `tests/` subfolder (wherever pySuStaIn has been installed on your system). Then, you can use the following command to run all SuStaIn variants (this may take a bit of time!):
```
python create_validation.py
```

Testing of single classes is possible using the `-c` flag, e.g. `python create_validation.py -c ordinal`. To see all options, run `python create_validation.py --help`.

---

# Examples

We have provided Jupyter notebooks to demonstrate how to run pySuStaIn in a few limited scenarios. You can also look at `simrun.py` for examples of how to run the different implementations:

- `mixture_GMM` : SuStaIn with an event-based model progression pattern, with Gaussian mixture modelling of normal/abnormal.
- `mixture_KDE`:  SuStaIn with an event-based model progression pattern, with Kernel Density Estimation (KDE) mixture modelling of normal/abnormal.
- `zscore`:       SuStaIn with a piecewise linear z-score model progression pattern.

We also have a set of tutorial videos on YouTube, which you can find [here](https://www.youtube.com/watch?v=5CFsfFcVzEc&list=PL25fUWY3exLxYSPOnEe60kSEh0JRdVVPB).

---

# Papers

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

---

# Funding

The original project received funding from the European Union's Horizon 2020 Research and Innovation Programme under Grant Agreement number 666992.

Ongoing maintenance, updates, bug-fixes, etc. are largely done out of the goodness of our hearts.

