###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Authors: Peter Wijeratne (p.wijeratne@ucl.ac.uk) and Leon Aksman (l.aksman@ucl.ac.uk)
# Contributors: Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
###
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook as cbook

import os

import pandas as pd

from simfuncs import *

from mixture_model import fit_all_gmm_models, fit_all_kde_models
import plotting

import warnings
warnings.filterwarnings("ignore",category=cbook.mplDeprecation)

from pySuStaIn.ZscoreSustain  import ZscoreSustain
from pySuStaIn.MixtureSustain import MixtureSustain

import sklearn.model_selection

import pylab

def main():

    #***************** parameters for generating the ground-truth subtypes
    N                       = 5         # number of biomarkers
    M                       = 800       # number of observations ( e.g. subjects )
    N_S_ground_truth        = 3         # number of ground truth subtypes

    # the fractions of the total number of subjects (M) belonging to each subtype
    ground_truth_fractions = np.array([0.5, 0.30, 0.20])

    #***************** parameters for SuStaIn-based inference of subtypes
    use_parallel_startpoints = True

    # number of starting points
    N_startpoints           = 25
    # maximum number of inferred subtypes - note that this could differ from N_S_ground_truth
    N_S_max                 = 3
    N_iterations_MCMC       = int(1e5)  #For speed - you should use int(1e6) normally, but it can take a long time

    # cross-validation
    validate                = True
    N_folds                 = 10

    #either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
    sustainType             = 'zscore'

    assert sustainType in ("mixture_GMM", "mixture_KDE", "zscore"), "sustainType should be either mixture_GMM, mixture_KDE or zscore"


    SuStaInLabels           = []
    for i in range(N):
        SuStaInLabels.append('Biomarker ' + str(i))  # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]


    #****************** generate the ground-truth sequences and groud-truth data (i.e. subjects' biomarker measures)
    dataset_name            = 'sim'
    output_folder           = os.path.join(os.getcwd(), dataset_name + '_' + sustainType)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if      sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":

        np.random.seed(5)

        ground_truth_subj_ids   = list(np.arange(1, M+1).astype('str'))
        ground_truth_sequences  = generate_random_mixture_sustain_model(N, N_S_ground_truth)

        ground_truth_subtypes   = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)

        N_stages                = N

        ground_truth_stages_control = np.zeros((int(np.round(M * 0.25)), 1))
        ground_truth_stages_other   = np.random.randint(1, N_stages+1, (int(np.round(M * 0.75)), 1))
        ground_truth_stages         = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

        data, data_denoised     = generate_data_mixture_sustain(ground_truth_subtypes, ground_truth_stages, ground_truth_sequences, sustainType)

        # choose which subjects will be cases and which will be controls
        MIN_CASE_STAGE          = np.round((N + 1) * 0.8)
        index_case              = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
        index_control           = np.where(ground_truth_stages ==  0)[0]

        labels                  = 2 * np.ones(data.shape[0], dtype=int)     # 2 - intermediate value, not used in mixture model fitting
        labels[index_case]      = 1                                         # 1 - cases
        labels[index_control]   = 0                                         # 0 - controls

        data_case_control       = data[labels != 2, :]
        labels_case_control     = labels[labels != 2]

        if sustainType == "mixture_GMM":
            mixtures            = fit_all_gmm_models(data, labels)
        elif sustainType == "mixture_KDE":
            mixtures            = fit_all_kde_models(data, labels)

        fig, ax                 = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels, plotting_font_size=20)
        fig.show()
        fig.savefig(os.path.join(output_folder, 'kde_fits.png'))

        L_yes                   = np.zeros(data.shape)
        L_no                    = np.zeros(data.shape)
        for i in range(N):
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif sustainType   == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

        sustain                 = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints)

    elif    sustainType == 'zscore':

        np.random.seed(10)

        Z_vals                  = np.array([[1, 2, 3]] * N)  # define the Z-score based events for each biomarker
        Z_max                   = np.array([5] * N)  # maximum z-score for each biomarker

        ground_truth_subj_ids   = list(np.arange(1, M+1).astype('str'))

        # generate the ground truth sequence for each subtype
        ground_truth_sequences  = generate_random_Zscore_sustain_model(Z_vals, N_S_ground_truth)

        # randomly generate the ground truth subtype and stage assignment for every one of the M subjects
        ground_truth_subtypes   = np.random.choice(range(N_S_ground_truth), M, replace=True, p=ground_truth_fractions).astype(int)

        N_stages                = np.sum(Z_vals > 0) + 1

        #1/4 of all subjects are assigned stage zero, the rest a random number from 1 to number of stages
        ground_truth_stages_control = np.zeros((int(np.round(M * 0.25)), 1))
        ground_truth_stages_other   = np.random.randint(1, N_stages+1, (int(np.round(M * 0.75)), 1))
        ground_truth_stages         = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

        data, data_denoised, stage_value = generate_data_Zscore_sustain( ground_truth_subtypes,
                                                                         ground_truth_stages,
                                                                         ground_truth_sequences,
                                                                         Z_vals,
                                                                         Z_max)

        # choose which subjects will be cases and which will be controls
        MIN_CASE_STAGE          = int(np.round((N_stages + 1) * 0.8))
        index_case              = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
        index_control           = np.where(ground_truth_stages ==  0)[0]


        labels                  = 2 * np.ones(data.shape[0], dtype=int)  # 2 - MCI, default assignment here
        labels[index_case]      = 0
        labels[index_control]   = 1

        sustain                 = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints)


    #****** plot the ground truth sequences
    ground_truth_sequences  = np.expand_dims(ground_truth_sequences, axis=2)
    ground_truth_fractions  = np.expand_dims(ground_truth_fractions, axis=1)
    ground_truth_nsamples   = np.inf
    fig, ax                 = sustain._plot_sustain_model(ground_truth_sequences, ground_truth_fractions, ground_truth_nsamples, title_font_size=12)
    fig.savefig(os.path.join(output_folder, 'PVD_true.png'))
    fig.show()

    #************* run SuStaIn to infer subtype sequences and subjects' subtypes/stages estimates
    samples_sequence,   \
    samples_f,          \
    ml_subtype,         \
    prob_ml_subtype,    \
    ml_stage,           \
    prob_ml_stage,      \
    prob_subtype_stage      = sustain.run_sustain_algorithm()

    #save the most likely subtype, the associated subtype probability,
    # the most likely stage and the associated stage probability for each subject
    df                      = pd.DataFrame()
    df['subj_id']           = ground_truth_subj_ids
    df['ml_subtype']        = ml_subtype
    df['prob_ml_subtype']   = prob_ml_subtype
    df['ml_stage']          = ml_stage
    df['prob_ml_stage']     = prob_ml_stage
    df.to_csv(os.path.join(output_folder, 'Subject_subtype_stage_estimates.csv'), index=False)

    FONT_SIZE               = 15

    #plot the inferred subtypes as histograms binned by true subtype
    plt.style.use('seaborn-deep')
    bins                    = np.arange(0, N_S_ground_truth+1)
    X_hist                  = list()
    labels_hist             = list()
    for i in range(N_S_max):
        X_hist.append(ml_subtype[ground_truth_subtypes==i])
        labels_hist.append('Subtype ' + str(i+1))
    plt.hist(X_hist, bins, label=labels_hist)
    plt.xticks(np.arange(0, N_S_ground_truth)+0.5, np.arange(1, N_S_ground_truth+1))
    plt.xlabel('Estimated subtype', fontsize=FONT_SIZE)
    plt.title('')
    plt.legend(loc='upper right', fontsize=FONT_SIZE)
    plt.savefig(os.path.join(output_folder, 'Subtype_estimate_histograms.png'))
    plt.show()

    #plot the inferred stages as boxplots binned by true stage
    df_boxplot                  = pd.DataFrame()
    df_boxplot['subtypes_true'] = ground_truth_subtypes
    df_boxplot['subtypes_est']  = ml_subtype
    df_boxplot['stages_true']   = ground_truth_stages
    df_boxplot['stages_est']    = ml_stage
    df_boxplot.boxplot(column='stages_est', by='stages_true', grid=False, fontsize=FONT_SIZE)
    plt.xlabel('True stages',       fontsize=FONT_SIZE)
    plt.ylabel('Estimated stages',  fontsize=FONT_SIZE)
    plt.title('')
    plt.savefig(os.path.join(output_folder, 'Stage_estimate_boxplots.png'))
    plt.show()

    print('Maximum likelihood model finished. Saved figures and output files in ' + output_folder + ' folder.')

    # ************* cross-validation: splits the whole dataset into a number of folds (N_folds),
    #               runs SuStaIn on the training data of each fold and evaluates out-of-sample log likelihood of fold's test data,
    #               also evaluates cross-validation information criterion (CVIC) and displays cross-validated positional variance diagram (PVD)
    if validate:

        print('Running cross-validation. This may take an hour or longer. Set validate = False to disable.')

        test_idxs              = []

        cv                     = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
        cv_it                  = cv.split(data, labels)

        for train, test in cv_it:
            test_idxs.append(test)
        test_idxs              = np.array(test_idxs)

        CVIC, loglike_matrix   = sustain.cross_validate_sustain_model(test_idxs)

        #output CV folds' out-of-sample log likelihoods
        df_loglike              = pd.DataFrame(data = loglike_matrix, columns = ["Subtype " + str(i+1) for i in range(N_S_max)])
        df_loglike.to_csv(os.path.join(output_folder, 'Log_likelihoods_cv_folds.csv'), index=False)

        #this part estimates cross-validated positional variance diagrams
        for i in range(N_S_max):
            sustain.combine_cross_validated_sequences(i+1, N_folds)

    plt.show()

    print('Cross-validation finished. Saved figures and output files in ' + output_folder + ' folder.')

if __name__ == '__main__':
    np.random.seed(42)
    main()
