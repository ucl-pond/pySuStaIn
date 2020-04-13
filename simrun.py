###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk)
###
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook as cbook

from simfuncs import generate_random_sustain_model, generate_data_sustain

#from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models
#from kde_ebm import plotting

import warnings
warnings.filterwarnings("ignore",category=cbook.mplDeprecation)

from ZscoreSustain  import ZscoreSustain
from MixtureSustain import MixtureSustain

import sklearn.model_selection

def main():
    # cross-validation
    validate                = True
    N_folds                 = 10

    N                       = 5         # number of biomarkers
    M                       = 500       # number of observations ( e.g. subjects )
    N_S_gt                  = 3         # number of ground truth subtypes

    # number of starting points
    N_startpoints           = 25
    # maximum number of subtypes
    N_S_max                 = 3
    N_iterations_MCMC       = int(1e5)  #int(1e6)

    #either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
    sustainType             = 'zscore'

    assert sustainType in ("mixture_GMM", "mixture_KDE", "zscore"), "sustainType should be either mixture_GMM, mixture_KDE or zscore"

    dataset_name            = 'sim'
    output_folder           = dataset_name + '_' + sustainType

    Z_vals                  = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
    Z_max                   = np.array([5]*N)           # maximum z-score

    stage_zscore            = np.array([y for x in Z_vals.T for y in x])
    stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))

    IX_vals                 = np.array([[x for x in range(N)]]*3).T
    stage_biomarker_index   = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

    min_biomarker_zscore    = [0]*N
    max_biomarker_zscore    = Z_max
    std_biomarker_zscore    = [1]*N

    SuStaInLabels           = []
    SuStaInStageLabels      = []
    # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
    for i in range(N):
        SuStaInLabels.append( 'Biomarker '+str(i))
    for i in range(len(stage_zscore)):
        SuStaInStageLabels.append('B'+str(stage_biomarker_index[i])+' - Z'+str(stage_zscore[i]))

    gt_f                    = [1+0.5*x for x in range(N_S_gt)]
    gt_f                    = [x/sum(gt_f) for x in gt_f][::-1]

    # ground truth sequence for each subtype
    gt_sequence             = generate_random_sustain_model(stage_zscore,stage_biomarker_index,N_S_gt)

    N_k_gt                  = np.array(stage_zscore).shape[1]+1
    subtypes                = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
    stages                  = np.ceil(np.random.rand(M,1)*(N_k_gt+1))-1


    data, data_denoised, stage_value = generate_data_sustain(subtypes,
                                                             stages,
                                                             gt_sequence,
                                                             min_biomarker_zscore,
                                                             max_biomarker_zscore,
                                                             std_biomarker_zscore,
                                                             stage_zscore,
                                                             stage_biomarker_index)

    # choose which subjects will be cases and which will be controls
    index_case              = np.where(data[:, 0] < 1)
    index_control           = np.where(data[:, 4] > 3)

    labels                  = 2 * np.ones(data.shape[0], dtype=int)  # 2 - MCI, default assignment here
    labels[index_case]      = 0
    labels[index_control]   = 1

    if      sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":

        data_case_control   = data[labels != 2, :]
        labels_case_control = labels[labels != 2]

        if sustainType == "mixture_GMM":
            mixtures        = fit_all_gmm_models(data, labels)
#        elif sustainType == "mixture_KDE":
#            mixtures        = fit_all_kde_models(data, labels)

#        fig, ax, _          = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels)
#        fig.show()
        # fig.savefig(os.path.join(outDir, 'kde_fits.png'))

        L_yes               = np.zeros(data.shape)
        L_no                = np.zeros(data.shape)
        for i in range(N):
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif sustainType   == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

        sustain             = MixtureSustain(L_yes, L_no,        SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name)


    elif    sustainType == 'zscore':

        sustain             = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, False)


    samples_sequence, samples_f, _,_,_,_ = sustain.run_sustain_algorithm()

    if validate:
        test_idxs           = []

        cv                  = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
        cv_it               = cv.split(data, labels)

        for train, test in cv_it:
            test_idxs.append(test)
        test_idxs           = np.array(test_idxs)

        CVIC_matrix         = sustain.cross_validate_sustain_model(test_idxs)

        #this part estimates cross-validated positional variance diagrams
        for i in range(N_S_max):
            sustain.combine_cross_validated_sequences(i+1, N_folds)


    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    main()
