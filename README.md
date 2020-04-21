pySuStaIn
============

SuStaIn algorithm in Python, with the option to describe the subtype progression patterns using either the event-based model or the piecewise linear z-score model.

Papers
============
- The SuStaIn algorithm: [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0) 
- The event-based model: [Fonteijn et al. 2012](https://doi.org/10.1016/j.neuroimage.2012.01.062), (with Gaussian mixture modelling [Young et al. 2014](https://doi.org/10.1093/brain/awu176))
- The piecewise linear z-score model: [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0) 

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
  - "zscore":       SuStaIn with a piecewise linear z-score model progression pattern.
  
 See simrun.py for examples of how to run these different implementations.

SuStaIn Tutorial
===============  
See the jupyter notebook for a tutorial on how to use SuStaIn using simulated data.
