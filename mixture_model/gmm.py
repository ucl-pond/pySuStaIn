# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from scipy import optimize
import numpy as np


class ParametricMM():
    """Wraps up two distributions and the mixture parameter.

    Attributes
    ----------
    dModel : distribution
        Distribution object to use for the diseased data.
    hModel : distribution
        Distribution object to use for the healthy data.
    mix : float
        Mixing fraction, as percent of healthy patients.
    """
    def __init__(self, cn_comp=None, ad_comp=None, mixture=None):
        """Initiate new GMM object

        Parameters
        ----------
        healthyModel : distribution, optional
            Distribution object to use for the healthy data.
        diseaseModel : TYdistributionPE, optional
            Distribution object to use for the diseased data.
        mixture : float, optional
            Mixing fraction, as percent of healthy patients.
        """
        self.cn_comp = cn_comp
        self.ad_comp = ad_comp
        self.mix = mixture
        self.theta = None

    def pdf(self, theta, X):
        """Summary

        Parameters
        ----------
        theta : array-like, shape(5,)
            List containing the parameters required for a mixture model.
            [hModelMu, hModelSig, dModelMu, dModelSig, mixture]
        inData : array-like, shape(numPatients,)
            Biomarker measurements for patients.

        Returns
        -------
        TYPE
            Description
        """
        if theta is None:
            theta = self.theta
        if np.isnan(X.sum()):
            raise ValueError('NaN in likelihood')
        if np.isnan(theta.sum()):
            out = np.empty(X.shape[0])
            out[:] = np.nan
            return out, out
        n_cn_params = self.cn_comp.n_params
        n_ad_params = self.ad_comp.n_params
        cn_theta = theta[:n_cn_params]
        ad_theta = theta[n_cn_params:n_cn_params+n_ad_params]
        mixture = theta[-1]

        self.cn_comp.set_theta(cn_theta)
        self.ad_comp.set_theta(ad_theta)

        cn_pdf = self.cn_comp.pdf(X)*mixture
        ad_pdf = self.ad_comp.pdf(X)*(1-mixture)
        return cn_pdf, ad_pdf

    def likelihood(self, theta, X):
        """"Calculates the likelihood of the data given the model
        parameters scored in theta. theta should contain normal mean,
        normal standard deviation, abnormal mean, abnormal standard
        deviation and the fraction of the data that is normal

        Parameters
        ----------
        theta : array-like, shape(5,)
            List containing the parameters required for a mixture model.
            [hModelMu, hModelSig, dModelMu, dModelSig, mixture]
        inData : array-like, shape(numPatients,)
            Biomarker measurements for patients.

        Returns
        -------
        likelihood : float
            Negative log likelihood of the data given the parameters theta.
        """
        # thetaNums allows us to use other distributions with a varying
        # number of paramters. Not included in this version of the code.
        cn_pdf, ad_pdf = self.pdf(theta, X)
        data_likelihood = cn_pdf + ad_pdf
        data_likelihood[data_likelihood == 0] = np.finfo(float).eps
        data_likelihood = np.log(data_likelihood)
        return -1*np.sum(data_likelihood)

    def fixed_cn_likelihood(self, ad_theta, X):
        theta = np.concatenate((self.cn_comp.get_theta(), ad_theta))
        return self.likelihood(theta, X)

    def fixed_ad_likelihood(self, ad_theta, X):
        raise NotImplementedError('Fixed ad component not yet needed')

    def probability(self, X):
        theta = self.theta
        controls_score, patholog_score = self.pdf(theta, X)
        return controls_score / (controls_score+patholog_score)

    def fit(self, X, y):
        """This will fit a mixture model to some given data. Labelled data
        is used to derive starting conditions for the optimize function,
        labels are 0 for normal and 1 for abnormal. The model type corresponds
        to the type of distributions used, currently there is normal and
        uniform distributions. Be careful when chosing distributions as the
        optimiser can throw out NaNs.

        Parameters
        ----------
        X : array-like, shape(numPatients,)
            Biomarker measurements for patients.
        y : array-like, shape(numPatients,)
            Diagnosis labels for each of the patients.

        Returns
        -------
        mixInfoOutput : array-like, shape(5,)
            List containing the parameters required for a mixture model.
            [hModelMu, hModelSig, dModelMu, dModelSig, mixture]
        """
        event_sign = np.nanmean(X[y == 0]) < np.nanmean(X[y == 1])
        opt_bounds = []
        opt_bounds += self.cn_comp.get_bounds(X, X[y == 0], event_sign)
        opt_bounds += self.ad_comp.get_bounds(X, X[y == 1], not event_sign)
        # Magic number
        opt_bounds += [(0.1, 0.9)]
        init_params = []
        init_params += self.cn_comp.estimate_params(X[y == 0])
        init_params += self.ad_comp.estimate_params(X[y == 1])
        # Magic number
        init_params += [0.5]
        res = optimize.minimize(self.likelihood,
                                init_params, args=(X[~np.isnan(X)],),
                                bounds=opt_bounds,
                                method='SLSQP')
        res = res.x
        if np.isnan(res.sum()):
            res = optimize.minimize(self.likelihood,
                                    init_params, args=(X[~np.isnan(X)],),
                                    bounds=opt_bounds)
            res = res.x
        n_cn_params = self.cn_comp.n_params
        n_ad_params = self.ad_comp.n_params
        self.cn_comp.set_theta(res[:n_cn_params])
        self.ad_comp.set_theta(res[n_cn_params:n_cn_params+n_ad_params])
        self.mix = res[-1]
        self.theta = res
        return res

    def fit_constrained(self, X, y, fixed_component=None):
        if fixed_component is not None:
            raise NotImplementedError('Only cn can be fixed currently')

        event_sign = np.nanmean(X[y == 0]) < np.nanmean(X[y == 1])

        cn_est = self.cn_comp.estimate_params(X[y == 0])
        self.cn_comp.set_theta(cn_est)

        opt_bounds = self.ad_comp.get_bounds(X, X[y == 1], not event_sign)
        # magic number
        opt_bounds += [(0.1, 0.9)]
        init_params = self.ad_comp.estimate_params(X[y == 1])
        # magic number
        init_params += [0.5]
        self.fixed_cn_likelihood(init_params, X[~np.isnan(X)])
        res = optimize.minimize(self.fixed_cn_likelihood,
                                init_params, args=(X[~np.isnan(X)],),
                                bounds=opt_bounds,
                                method='SLSQP')
        res = res.x
        if np.isnan(res.sum()):
            res = optimize.minimize(self.fixed_cn_likelihood,
                                    init_params, args=(X[~np.isnan(X)],),
                                    bounds=opt_bounds)
            res = res.x
        self.ad_comp.set_theta(res[:-1])
        self.mix = res[-1]
        return res
