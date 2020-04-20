pySuStaIn
============

SuStaIn algorithm in Python, with the option to describe the subtype progression patterns using either the event-based model or the linear z-score model.

SuStaIn papers
============
Please cite [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0) if you use SuStaIn for your research.

Dependencies
============
- Python >= 3.5 
- [NumPy >= 1.18](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [Scikit-learn](https://scikit-learn.org) for cross-validation
- [kde_ebm](https://github.com/noxtoby/kde_ebm_open) for mixture modelling (KDE and GMM included)
- [pathos](https://github.com/uqfoundation/pathos) for parallelization

Parallelisation
===============
- Added parallelized startpoints

Running different SuStaIn implementations
===============
sustainType can be set to:
  - "mixture_GMM" : SuStaIn with an event-based model progression pattern, with Gaussian mixture modelling of normal/abnormal.
  - "mixture_KDE":  SuStaIn with an event-based model progression pattern, with Kernel Density Estimation (KDE) mixture modelling of normal/abnormal.
  - "zscore":       SuStaIn with a linear z-score model progression pattern.
 See simrun.py for examples of how to run these different implementations.

SuStaIn Tutorial
===============  
See the jupyter notebook for a tutorial on how to use SuStaIn using simulated data.
