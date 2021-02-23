# coding: utf8

"""
Part of awkde package
---------------------

Helper tools for standardizing a data sample.
"""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as _np


def standardize_nd_sample(sam, mean=None, cov=None,
                          cholesky=True, ret_stats=False, diag=False):
    """
    Standardizes a n-dimensional sample using the Mahalanobis distance.

    .. math:: x' = \Sigma^{-1/2} (x - y)

    The resulting sample :math:`x'` has a zero mean vector and an identity
    covariance.

    Parameters
    ----------
    sam : array-like, shape (n_samples, n_features)
        Data points in the sample, each column is a feature, each row a point.
    mean : array-like, shape (n_features), optional
        If explicitely given, use this mean vector for the transformation. If
        None, the estimated mean from data is used. (default: None)
    cov : array-like, shape (n_features, n_features), optional
        If explicitely given, use this covariance matrix for the transformation.
        If None, the estimated cov from data is used. (default: None)
    cholesky : bool, optional
        If true, use fast Cholesky decomposition to calculate the sqrt of the
        inverse covariance matrix. Else use eigenvalue decomposition (Can be
        numerically unstable, not recommended). (default: True)
    ret_stats : bool, optional
        If True, the mean vector and covariance matrix of the input sample are
        returned, too. (default: False)
    diag : bool
        If True, only scale by variance, diagonal cov matrix. (default: False)

    Returns
    -------
    stand_sam : array-like, shape (n_samples, n_features)
        Standardized sample, with mean = [0., ..., 0.] and cov = identity.

    Optional Returns
    ----------------
    mean : array-like, shape(n_features)
        Mean vector of the input data, only if ret_stats is True.
    cov : array-like, shape(n_features, n_features)
        Covariance matrix of the input data, only if ret_stats is True.

    Example
    -------
    >>> mean = [10, -0.01, 1]
    >>> cov = [[14, -.2, 0], [-.2, .1, -0.1], [0, -0.1, 1]]
    >>> sam = np.random.multivariate_normal(mean, cov, size=1000)
    >>> std_sam = standardize_nd_sample(sam)
    >>> print(np.mean(std_sam, axis=0))
    >>> print(np.cov(std_sam, rowvar=False))
    """
    if len(sam.shape) != 2:
        raise ValueError("Shape of `sam` must be (n_samples, n_features).")
    if mean is None and cov is None:
        # Mean and cov over the first axis
        mean = _np.mean(sam, axis=0)
        cov = _np.atleast_2d(_np.cov(sam, rowvar=False))
    elif mean is not None and cov is not None:
        mean = _np.atleast_1d(mean)
        cov = _np.atleast_2d(cov)
        if len(mean) != sam.shape[1]:
            raise ValueError("Dimensions of mean and sample don't match.")
        if cov.shape[0] != sam.shape[1]:
            raise ValueError("Dimensions of cov and sample don't match.")

    if diag:
        cov = _np.diag(cov) * _np.eye(cov.shape[0])

    if cholesky:
        # Cholesky produces a tridiagonal matrix from A with: L L^T = A
        # To get the correct trafo, we need to transpose the returned L:
        #   L.L^t
        sqrtinvcov = _np.linalg.cholesky(_np.linalg.inv(cov)).T
    else:
        # The naive sqrt of eigenvalues. Is (at least) instable for > 3d
        # A = Q lam Q^-1. If A is symmetric: A = Q lam Q^T
        lam, Q = _np.linalg.eig(_np.linalg.inv(cov))
        sqrtlam = _np.sqrt(lam)
        sqrtinvcov = _np.dot(sqrtlam * Q, Q.T)

    # Transform each sample point and reshape result (n_samples, n_features)
    stand_sam = _np.dot(sqrtinvcov, (sam - mean).T).T

    if ret_stats:
        return stand_sam, mean, cov
    else:
        return stand_sam


def shift_and_scale_nd_sample(sam, mean, cov, cholesky=True):
    """
    Shift and scale a nD sample by given mean and covariance matrix.

    This is the inverse operation of `standardize_nd_sample`. If a
    standardized sample :math:`x'` with zero mean vector and identity covariance
    matrix is given, it is rescaled and shifted using

    .. math:: x = (\Sigma^{1/2} x) + y

    then having a mean vector `mean` and a covariance matrix `cov`.

    Parameters
    ----------
    sam : array-like, shape (n_samples, n_features)
        Data points in the sample, each column is a feature, each row a point.
    mean : array-like, shape (n_features)
        Mean vector used for the transformation.
    cov : array-like, shape (n_features, n_features)
        Covariance matrix used for the transformation.

    Returns
    -------
    scaled_sam : array-like, shape (n_samples, n_features)
        Scaled sample using the transformation with the given mean and cov.
    """
    if len(sam.shape) != 2:
        raise ValueError("Shape of `sam` must be (n_samples, n_features).")
    mean = _np.atleast_1d(mean)
    cov = _np.atleast_2d(cov)
    if len(mean) != sam.shape[1]:
        raise ValueError("Dimensions of mean and sample don't match.")
    if cov.shape[0] != sam.shape[1]:
        raise ValueError("Dimensions of cov and sample don't match.")

    # Transformation matrix: inverse of original trafo
    sqrtinvcov = _np.linalg.cholesky(_np.linalg.inv(cov)).T
    sqrtcov = _np.linalg.inv(sqrtinvcov)

    return _np.dot(sqrtcov, sam.T).T + mean
