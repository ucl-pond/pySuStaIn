pySuStaIn
============

[![Build Status](https://travis-ci.org/ucl-pond/pySuStaIn.svg?branch=dev)](https://travis-ci.org/ucl-pond/pySuStaIn)

Subtyping and Staging Inference (SuStaIn) algorithm implementation in Python. SuStaIn is a model-based and data-driven machine learning method to 
discover subtypes of patients who share similar patterns of progression and stage them across the trajectory of a given disease.

![SuSTaIn implemented on imaging markers in multiple scelrosis](https://raw.githubusercontent.com/ucl-pond/pySuStaIn/dev/sustain.png){:height="50%" width="50%"}

This image shows two data-driven subtypes discovered by SuStaIn. The horizontal axis is the sequence through which subjects progress. The vertical axis shows different imaging variables fed to SuStaIn. Three differnt colours show the extent of deviation from healthy-control population (mild, moderate or severe abnormality according to Z-score distribution). 

SuStaIn papers
============

If you use SuStaIn please cite the following paper:

- [Young et al. 2018](https://doi.org/10.1038/s41467-018-05892-0)

Dependencies
============
- Python >= 3.5 
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

Parallelisation
===============
- pySuStaIn uses Python's `multiprocessing` module to parallelise expectation maximisation algorithms.
