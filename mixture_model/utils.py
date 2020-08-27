import numpy as np
from ..distributions.gaussian import Gaussian
from .gmm import ParametricMM
from .kde import KDEMM


def get_prob_mat(X, mixture_models):
    """Gives the matrix of probabilities that a patient has normal/abnormal
    measurements for each of the biomarkers. Output is number of patients x
    number of biomarkers x 2.

    Parameters
    ----------
    X : array-like, shape(numPatients, numBiomarkers)
        All patient-all biomarker measurements.
    y : array-like, shape(numPatients,)
        Diagnosis labels for each of the patients.
    mixtureModels : array-like, shape(numBiomarkers,)
        List of fit mixture models for each of the biomarkers.

    Returns
    -------
    outProbs : array-like, shape(numPatients, numBioMarkers, 2)
        Probability for a normal/abnormal measurement in all biomarkers
        for all patients (and controls).
    """

    n_particp, n_biomarkers = X.shape
    prob_mat = np.zeros((n_particp, n_biomarkers, 2))
    for i in range(n_biomarkers):
        nan_mask = ~np.isnan(X[:, i])
        probs = mixture_models[i].probability(X[nan_mask, i])
        prob_mat[nan_mask, i, 0] = probs
        prob_mat[~nan_mask, i, 0] = 0.5
    prob_mat[:, :, 1] = 1-prob_mat[:, :, 0]
    return prob_mat


def fit_all_gmm_models(X, y, implement_fixed_controls=False):
    #* Extract only the first two diagnoses
    msk = np.where(y<2)[0]
    X = X[msk]
    y = y[msk]
    
    n_particp, n_biomarkers = X.shape
    mixture_models = []
    for i in range(n_biomarkers):
        bio_y = y[~np.isnan(X[:, i])]
        bio_X = X[~np.isnan(X[:, i]), i]
        cn_comp = Gaussian()
        ad_comp = Gaussian()
        mm = ParametricMM(cn_comp, ad_comp)
        mm.fit(bio_X, bio_y)
        mixture_models.append(mm)
    return mixture_models


def fit_all_kde_models(X, y, implement_fixed_controls=False):
    #* Extract only the first two diagnoses
    msk = np.where(y<2)[0]
    X = X[msk]
    y = y[msk]
    
    n_particp, n_biomarkers = X.shape
    kde_mixtures = []
    for i in range(n_biomarkers):
        bio_X = X[:, i]
        bio_y = y[~np.isnan(bio_X)]
        bio_X = bio_X[~np.isnan(bio_X)]
        # print('utils:fit_all_kde_models() \n  - range(np.isnan(bio_X[y=0/1])) = [{0},{1}],[{2},{3}]'.format(
        #     min((bio_X[bio_y==0])),max((bio_X[bio_y==0])),
        #     min((bio_X[bio_y==1])),max((bio_X[bio_y==1]))
        #     )
        # )
        kde = KDEMM()
        kde.fit(bio_X, bio_y,implement_fixed_controls)
        kde_mixtures.append(kde)
    return kde_mixtures