LOOpySuStaIn: Leon's Object Oriented pySuStaIn
============

SuStaIn algorithm in Python, with a combination of mixture style (i.e. EBM style) and z-score style SuStaIn implementations.

SuStaIn papers
============
- [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0)

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
In simrun.py sustainType can be set to:
  - "mixture_GMM" : mixture model style SuStaIn with Gaussian mixture modelling of normal/abnormal.
  - "mixture_KDE":  mixture model style SuStaIn with Kernel Density Estimation (KDE) mixture modelling of normal/abnormal.
  - "zscore":       z-score style SuStaIn with three events for each biomarker (1,2,3 std. devs. from normality)

