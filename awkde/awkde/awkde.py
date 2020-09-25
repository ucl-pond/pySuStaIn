# coding: utf-8

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from builtins import int, open
from future import standard_library
standard_library.install_aliases()

import os
import sys
import json
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from awkde.tools import standardize_nd_sample, shift_and_scale_nd_sample
import awkde.backend as backend


class GaussianKDE(BaseEstimator):
    """
    GaussianKDE

    Kernel denstiy estimate using gaussian kernels and a local kernel bandwidth.
    Implements the ``sklearn.BaseEstimator`` class and can be used in a cross-
    validation gridsearch (``sklearn.model_selection``).

    Parameters
    ----------
    glob_bw : float or str, optional
        The global bandwidth of the kernel, must be a float ``> 0`` or one of
        ``['silverman'|'scott']``. If ``alpha`` is not ``None``, this is the
        bandwidth used for the first estimate KDE from which the local bandwidth
        is calculated. ``If ['silverman'|'scott']`` a rule of thumb is used to
        estimate the global bandwidth. (default: 'silverman')
    alpha : float or None, optional
        If ``None``, only the global bandwidth ``glob_bw`` is used. If
        ``0 <= alpha <= 1``, an adaptive local kernel bandwith is used as
        described in [1]_. (default: 0.5)
    diag_cov : bool, optional
        If ``True``, scale fit sample by variance only, which means using a
        diagonal covariance matrix. (default: False)

    Notes
    -----
    The unweighted kernel density estimator is defined as

    .. math:

      \hat{f}(x) = \sum_i \frac{1}{h\lambda_i}\cdot
                     K\left(\frac{x - X_i}{h\lambda_i}\right)


    where the product :math:`h\lambda_i` takes the role of a local
    variance :math`\sigma_i^2`.

    The kernel bandwith is choosen locally to account for variations in the
    density of the data.
    Areas with large density gets smaller kernels and vice versa.
    This smoothes the tails and gets high resolution in high statistics regions.
    The local bandwidth parameter is defined as

    .. math: \lambda_i = (\hat{f}(X_i) / g)^{-\alpha}

    where :math:`\log g = n^{-1}\sum_i \log\hat{f}(X_i)` is some normalization
    and :math:`\hat{f}(X_i)` the KDE estimate at the data point :math:`X_i`.
    The local bandwidth is multiplied to the global bandwidth for each kernel.

    Furthermore different scales in data is accounted for by scaling it via its
    covariance matrix to an equal spread.
    First a global kernel bandwidth is applied to the transformed data and then
    based on that density a local bandwidth parameter is applied.

    All credit for the method goes to [1]_ and to S. Schoenen and L. Raedel for
    huge parts of the implementation :+1:.
    For information on Silverman or Scott rule, see [2]_ or [3]_.

    References
    ----------
    .. [1] B. Wang and X. Wang, "Bandwidth Selection for Weighted Kernel Density
           Estimation", Sep. 2007, DOI: 10.1214/154957804100000000.
    .. [2] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [3] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    """
    def __init__(self, glob_bw="silverman", alpha=0.5, diag_cov=False):
        if type(glob_bw) is str:
            if glob_bw not in ["silverman", "scott"]:
                raise ValueError("glob_bw can be one of ['silverman'|'scott'].")
            pass
        elif glob_bw <= 0:
            raise ValueError("Global bandwidth must be > 0.")

        # List class attributes. Setup indicating that no fit was done yet
        self._n_kernels = None
        self._n_features = None
        self._std_X = None
        self._mean = None
        self._cov = None
        self._kde_values = None
        self._inv_loc_bw = None
        self._adaptive = None

        self.alpha = alpha
        self._glob_bw = glob_bw
        self._diag_cov = diag_cov

        return

    # Properties
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """
        The adaptive width can easily be changed after the model has been fit,
        because the computation only needs the cached ``_kde_values``.
        """
        if alpha is None:
            self._adaptive = False
        else:
            if alpha < 0 or alpha > 1:
                raise ValueError("alpha must be in [0, 1]")
            self._adaptive = True

        self._alpha = alpha

        if self._std_X is not None and self._adaptive:
            # Recalculate local bandwidth if we already have a fitted model
            self._calc_local_bandwidth()

    @property
    def glob_bw(self):
        return self._glob_bw

    @property
    def diag_cov(self):
        return self._diag_cov

    # Public Methods
    def __call__(self, X):
        # Does the same as `predict`, only copy docstring here
        self.__call__.__func__.__doc__ = self.predict.__doc__
        return self.predict(X)

    def fit(self, X, bounds=None, weights=None):
        """
        Prepare KDE to describe the data.

        Data is transformed via global covariance matrix to equalize scales in
        different features.
        Then a symmetric kernel with cov = diag(1) is used to describe the pdf
        at each point.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points defining each kernel position. Each row is a point, each
            column is a feature.
        bounds : array-like, shape (n_features, 2)
            Boundary condition for each dimension. The method of mirrored points
            is used to improve prediction close to bounds. If no bound shall be
            given in a specific direction use ``None``, eg.
            ``[[0, None], ...]``. If ``bounds`` is ``None`` no bounds are used
            in any direction. (default: ``None``)
        weights : array-like, shape (n_samples), optional
            Per event weights to consider for ``X``. If ``None`` all weights are
            set to one. (default: ``None``)

        Returns
        -------
        mean : array-like, shape (n_features)
            The (weighted) mean of the given data.
        cov : array-like, shape (n_features, n_features)
            The (weighted) covariance matrix of the given data.

        Raises
        ------
        ``NotImplementedError`` if ``bounds`` or ``weights`` are not ``None``.
        """
        if bounds is not None:
            # TODO: Use mirroring of points near boundary regions and then
            #       constrain KDE to values inside Region but taking all kernels
            #       into account. (only neccessary on hard cuts)
            raise NotImplementedError("TODO: Boundary conditions.")
        if weights is not None:
            # TODO: Implement weighted statitistics
            raise NotImplementedError("TODO: Implement weighted statistics.")

        if len(X.shape) != 2:
            raise ValueError("`X` must have shape (n_samples, n_features).")

        # Transform sample to zero mean and unity covariance matrix
        self._n_kernels, self._n_features = X.shape
        self._std_X, self._mean, self._cov = standardize_nd_sample(
            X, cholesky=True, ret_stats=True, diag=self._diag_cov)

        # Get global bandwidth number
        self._glob_bw = self._get_glob_bw(self._glob_bw)

        # Build local bandwidth parameter if alpha is set
        if self._adaptive:
            self._kde_values = self._evaluate(self._std_X, adaptive=False)
            self._calc_local_bandwidth()

        return self._mean, self._cov

    def predict(self, X):
        """
        Evaluate KDE at given points X.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points we want to evaluate the KDE at. Each row is a point,
            each column is a feature.

        Returns
        -------
        prob : array-like, shape (len(X))
            The probability from the KDE pdf for each point in X.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        X = np.atleast_2d(X)
        _, n_feat = X.shape
        if n_feat != self._n_features:
            raise ValueError("Dimensions of given points and KDE don't match.")

        # Standardize given points to be in the same space as the KDE
        X = standardize_nd_sample(X, mean=self._mean, cov=self._cov,
                                  cholesky=True, ret_stats=False,
                                  diag=self._diag_cov)

        # No need to backtransform, because we only return y-values
        return self._evaluate(X, adaptive=self._adaptive)

    def sample(self, n_samples, random_state=None):
        """
        Get random samples from the KDE model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. (default: 1)
        random_state : RandomState, optional
            Turn seed into a `np.random.RandomState` instance. Method from
            `sklearn.utils`. Can be None, int or RndState. (default: None)

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            Generated samples from the fitted model.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        rndgen = check_random_state(random_state)

        # Select randomly all kernels to sample from
        idx = rndgen.randint(0, self._n_kernels, size=n_samples)

        # Because we scaled to standard normal dist, we can draw uncorrelated
        # and the cov is only the inverse bandwidth of each kernel.
        means = self._std_X[idx]
        invbw = np.ones(n_samples) / self._glob_bw
        if self._adaptive:
            invbw *= self._inv_loc_bw[idx]
        invbw = invbw.reshape(n_samples, 1)

        # Retransform to original space
        sample = np.atleast_2d(rndgen.normal(means, 1. / invbw))
        return shift_and_scale_nd_sample(sample, self._mean, self._cov)

    def score(self, X):
        """
        Compute the total ln-probability of points X under the KDE model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points included in the score calculation. Each row is a point,
            each column is a feature.

        Returns
        -------
        lnprob : float
            Total ln-likelihood of the data ``X`` given the KDE model.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        X = np.atleast_2d(X)
        _, n_feat = X.shape
        if n_feat != self._n_features:
            raise ValueError("Dimensions of given points and KDE don't match.")

        probs = self.predict(np.atleast_2d(X))
        if np.any(probs <= 0):
            return -np.inf
        else:
            return np.sum(np.log(probs))

    def to_json(self, fpath):
        """
        Write out the relevant parameters for the KDE model as a JSON file,
        which can be used to reconstruct the whole model with ``from_json``.

        Parameters
        ----------
        fpath : string
            File path where to save the JSON dump.
        """
        if self._std_X is None:
            raise ValueError("KDE has not been fitted to data yet.")

        out = self.get_params()  # From implementing sklearnBaseEstimator
        out["kde_X_std"] = [list(Xi) for Xi in self._std_X]
        out["kde_X_mean"] = list(self._mean)
        out["kde_X_cov"] = [list(Xi) for Xi in self._cov]

        if self._kde_values is not None:
            out["kde_Y"] = list(self._kde_values)
        else:
            out["kde_Y"] = None

        # json seems to behave differently in py2 vs py3 ...
        if sys.version_info[0] < 3:
            mode = "wb"
        else:
            mode = "w"
        with open(os.path.abspath(fpath), mode) as f:
            json.dump(obj=out, fp=f, indent=2)

        return

    @classmethod
    def from_json(cls, fpath, verb=False):
        """
        Build a awKDE object from a JSON dict with the needed parts.

        Parameters
        ----------
        fpath : string
            Path to the JSON file. Must have keys:

            - 'alpha', 'diag_cov', 'glob_bw': See GaussianKDE docstring.
            - 'kde_Y': KDE function values at points 'kde_X_std' used for the
                       adaptive kernel computation.
            - 'kde_X_std': Standardized sample in shape
                           ``(nsamples, nfeatures)``.
            - 'kde_X_mean': Mean vector of the standardized sample.
            - 'kde_X_cov': Covariance matrix of the stadardized sample.
        verb : bool, optional
            If ``True`` print model summary. (default: ``False``)

        Returns
        -------
        kde : KDE.GaussianKDE
            KDE object in fitted state, ready to evaluate or sample from.
        """
        with open(os.path.abspath(fpath), "rb") as f:
            d = json.load(f)

        kde = cls(glob_bw=d["glob_bw"], alpha=d["alpha"],
                  diag_cov=d["diag_cov"])

        # Reconstruct all internals without using fit again
        kde._std_X = np.atleast_2d(d["kde_X_std"])
        kde._n_kernels, kde._n_features = kde._std_X.shape
        kde._mean = np.atleast_1d(d["kde_X_mean"])
        kde._cov = np.atleast_2d(d["kde_X_cov"])

        if len(kde._mean) != kde._n_features:
            raise ValueError("'kde_X_mean' has not the same dimension " +
                             "as the X values.")
        if kde._cov.shape != (kde._n_features, kde._n_features):
            raise ValueError("'kde_X_cov' has not shape " +
                             "(n_features, n_features).")

        if d["kde_Y"] is not None:
            if d["alpha"] is None:
                raise ValueError("Saved 'alpha' is None, but 'kde_Y' is not.")
            # Set kde values and alpha to restore inverse bandwidth internally
            kde._kde_values = np.atleast_1d(d["kde_Y"])
            kde.alpha = d["alpha"]

        if len(kde._kde_values) != kde._n_kernels:
            raise ValueError("'kde_Y' has not the same length as 'kde_X_std'.")

        if verb:
            print("Loaded KDE model from {}".format(fpath))
            print("- glob_bw         : {:.3f}".format(kde._glob_bw))
            print("- alpha           : {:.3f}".format(kde._alpha))
            print("- adaptive        : {}".format(kde._adaptive))
            print("- Nr. of kernels  : {:d}".format(kde._n_kernels))
            print("- Nr. of data dim : {:d}".format(kde._n_features))

        return kde

    # Private Methods
    def _evaluate(self, X, adaptive):
        """
        Evaluate KDE at given points, returning the log-probability.

        Parameters
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points we want to evaluate the KDE at. Each row is a point,
            each column is a feature.
        adaptive : bool, optional
            Wether to evaluate with fixed or with adaptive kernel.
            (default: True)

        Returns
        -------
        prob : array-like, shape (len(X))
            The probability from the KDE PDF for each point in X.
        """
        n = self._n_kernels
        d = self._n_features

        # Get fixed or adaptive bandwidth
        invbw = np.ones(n) / self._glob_bw
        if adaptive:
            invbw *= self._inv_loc_bw

        # Total norm, including gaussian kernel norm with data covariance
        norm = invbw**d / np.sqrt(np.linalg.det(2 * np.pi * self._cov)) / n

        return backend.kernel_sum(self._std_X, X, invbw, norm)

    def _get_glob_bw(self, glob_bw):
        """Simple wrapper to handle string args given for global bw."""
        dim = self._n_features
        nsam = self._n_kernels
        if glob_bw == "silverman":
            return np.power(nsam * (dim + 2.0) / 4.0, -1. / (dim + 4))
        elif glob_bw == "scott":
            return np.power(nsam, -1. / (dim + 4))
        else:
            return self._glob_bw

    def _calc_local_bandwidth(self):
        """ Build the local bandwidth from cached ``_kde_values``."""
        # Get local bandwidth from local "density" g
        g = (np.exp(np.sum(np.log(self._kde_values)) / self._n_kernels))
        # Needed inverted so use power of (+alpha), shape (n_samples)
        self._inv_loc_bw = (self._kde_values / g)**(self._alpha)
        return
