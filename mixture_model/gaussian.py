# Authors: Nicholas C. Firth <ncfirth87@gmail.com>
# License: TBC
from scipy import stats
import numpy as np


class Gaussian():
    """Wrapper for distributions to be used in the mixture
    modelling. Addition distributions can be added here.
    Attributes
    ----------
    scipDist : scipy.stats.object
        scipy distribution to be used for pdf calculations.
    theta : array-like, shape(2)
        Array of the mean and standard deviation of the normal distribution
    """
    def __init__(self, mu=None, sigma=None):
        """Constructor for the Gaussian class.
        Parameters
        ----------
        theta : array-like, shape(2), optional
            An array of the parameters for this distribution.
        """
        self.n_params = 2
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm

    def pdf(self, X):
        """Summary
        Parameters
        ----------
        X : array-like, shape(n_participants)
            Array of biomarker data for patients. Should not contain
            NaN values.
        Returns
        -------
        name : array-like, shape(n_participants)
            The probability distribution function of each of the values
            from X.
        """
        return self.dist.pdf(X, loc=self.mu, scale=self.sigma)

    def set_params(self, mu=None, sigma=None):
        """Set's the theta values for this instance of the class.
        Parameters
        ----------
        mu : None, optional
            Description
        sigma : None, optional
            Description
        """
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma

    def set_theta(self, theta):
        self.set_params(mu=theta[0], sigma=theta[1])

    def get_theta(self):
        return [self.mu, self.sigma]

    def get_bounds(self, X_mix, X_comp, event_sign):
        """Get the bounds be used in the minimisation of the mixture model.
        Parameters
        ----------
        X_mix : array-like, shape(n_participants)
            All patient data for this biomarker
        X_comp : array-like, shape(n_subpopulation_participants)
            Sample of the patient data used to create distribution.
            This is usually either controls or AD diagnosed data.
        event_sign : bool
            1 if this sample mean is greater than the mean of the other
            component in the mixture model, 0 otherwise.
        Returns
        -------
        name : array-like, shape(2, 2)
            (upper-bound, lower-bound) Pairs for each of the parameters in
            theta, i.e. mean and standard deviation.
        """
        if event_sign:
            return [(np.nanmin(X_mix), np.nanmean(X_comp)),
                    (0.05*np.nanstd(X_comp), np.nanstd(X_comp))]
        else:
            return [(np.nanmean(X_comp), np.nanmax(X_comp)),
                    (0.05*np.nanstd(X_comp), np.nanstd(X_comp))]

    def estimate_params(self, X_comp):
        """Gets values for the start point of the optimisation.
        Parameters
        ----------
        X_comp : array-like, shape(n_participantsSample)
            Sample of the patient data used to create distribution.
            This is usually either controls or AD diagnosed data.
        Returns
        -------
        name : array-like, shape(2)
            Initial values of parameters in theta for optimisation.
        """
        return [np.nanmean(X_comp), np.nanstd(X_comp)]

    def __repr__(self,):
        return "Gaussian(mu=%r,sigma=%r)" % (self.mu, self.sigma)

    def __str__(self):
        return self.__repr__()