###
# pySuStaIn: a Python implementation of the Subtype and Stage Inference (SuStaIn) algorithm
#
# If you use pySuStaIn, please cite the following core papers:
# 1. The original SuStaIn paper:    https://doi.org/10.1038/s41467-018-05892-0
# 2. The pySuStaIn software paper:  https://doi.org/10.1016/j.softx.2021.100811
#
# Please also cite the corresponding progression pattern model you use:
# 1. The piece-wise linear z-score model (i.e. ZscoreSustain):  https://doi.org/10.1038/s41467-018-05892-0
# 2. The event-based model (i.e. MixtureSustain):               https://doi.org/10.1016/j.neuroimage.2012.01.062
#    with Gaussian mixture modeling (i.e. 'mixture_gmm'):       https://doi.org/10.1093/brain/awu176
#    or kernel density estimation (i.e. 'mixture_kde'):         https://doi.org/10.1002/alz.12083
# 3. The model for discrete ordinal data (i.e. OrdinalSustain): https://doi.org/10.3389/frai.2021.613261
#
# Thanks a lot for supporting this project.
#
# Authors:      Peter Wijeratne (p.wijeratne@ucl.ac.uk) and Leon Aksman (leon.aksman@loni.usc.edu)
# Contributors: Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk), Cameron Shand (c.shand@ucl.ac.uk)
###
import numpy as np
from scipy.stats import norm

#generate the Z-score based event sequences for the desired number of subtypes (N_S)
def generate_random_Zscore_sustain_model(Z_vals, N_S):

    B                                   = Z_vals.shape[0]
    stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
    stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

    IX_select                           = stage_zscore > 0
    stage_zscore                        = stage_zscore[IX_select]
    stage_zscore                        = stage_zscore.reshape(1, len(stage_zscore))

    num_zscores                         = Z_vals.shape[1]
    IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))
    stage_biomarker_index               = stage_biomarker_index[IX_select]
    stage_biomarker_index               = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

    N                                   = np.array(stage_zscore).shape[1]
    S                                   = np.zeros((N_S, N))
    for s in range(N_S):
        for i in range(N):
            IS_min_stage_zscore         = np.array([False] * N)
            possible_biomarkers         = np.unique(stage_biomarker_index)

            for j in range(len(possible_biomarkers)):
                IS_unselected           = [False] * N

                for k in set(range(N)) - set(S[s][:i]):
                    IS_unselected[k]    = True

                this_biomarkers         = np.array([(np.array(stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore = 0
                else:
                    this_min_stage_zscore = min(stage_zscore[this_biomarkers])
                if (this_min_stage_zscore):
                    temp                = ((this_biomarkers.astype(int) + (stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                    temp                = temp.reshape(len(temp), )
                    IS_min_stage_zscore[temp] = True

            events                      = np.array(range(N))
            possible_events             = np.array(events[IS_min_stage_zscore])
            this_index                  = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
            S[s][i]                     = possible_events[int(this_index)]

    return S


#generate the mixture model based event sequences for the desired number of subtypes (N_S)
def generate_random_mixture_sustain_model(N_biomarkers, N_S):

    S                                   = np.zeros((N_S, N_biomarkers))

    for i in range(30): #try 30 times to find a unique sequence for each subtype

        matched_others                  = False
        for s in range(N_S):
            S[s, :]                     = np.random.permutation(N_biomarkers)

            #compare to all previous sequences
            for i in range(s):
                if np.all(S[s, :] == S[i, :]):
                    matched_others      = True

        #all subtype sequences are unique, so break
        if not matched_others:
            break

    if matched_others:
        print('WARNING: Iterated 30 times and could not find unique sequences for all subtypes.')

    return S

#generate the Z-score based ideal and noisy data for subjects with given subtypes, stages and ground truth ordering
def generate_data_Zscore_sustain(subtypes, stages, gt_ordering, Z_vals, Z_max):

    B                                   = Z_vals.shape[0]
    stage_zscore                        = np.array([y for x in Z_vals.T for y in x])
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))
    IX_select                           = stage_zscore>0
    stage_zscore                        = stage_zscore[IX_select]
    stage_zscore                        = stage_zscore.reshape(1,len(stage_zscore))

    num_zscores                         = Z_vals.shape[1]
    IX_vals                             = np.array([[x for x in range(B)]] * num_zscores).T
    stage_biomarker_index               = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    stage_biomarker_index               = stage_biomarker_index[IX_select]
    stage_biomarker_index               = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

    min_biomarker_zscore                = [0]*B
    max_biomarker_zscore                = Z_max
    std_biomarker_zscore                = [1]*B

    N                                   = stage_biomarker_index.shape[1]
    N_S                                 = gt_ordering.shape[0]

    possible_biomarkers                 = np.unique(stage_biomarker_index)
    stage_value                         = np.zeros((B,N+2,N_S))

    for s in range(N_S):
        S                               = gt_ordering[s,:]
        S_inv                           = np.array([0]*N)
        S_inv[S.astype(int)]            = np.arange(N)
        for i in range(B):
            b                           = possible_biomarkers[i]
            event_location              = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
            event_value                 = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

            for j in range(len(event_location)-1):

                if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                    index               = np.arange(event_location[j],event_location[j+1]+2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                else:
                    index               = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

    M                                   = stages.shape[0]
    data_denoised                       = np.zeros((M,B))
    for m in range(M):
        data_denoised[m,:]              = stage_value[:,int(stages[m]),subtypes[m]]
    data                                = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

    return data, data_denoised, stage_value



#generate the mixture model based ideal and noisy data for subjects with given subtypes, stages and ground truth ordering
def generate_data_mixture_sustain(subtypes, stages, gt_ordering, mixture_style):

    #N_subtypes                          = gt_ordering.shape[0]
    N_biomarkers                        = gt_ordering.shape[1]

    N_subjects                          = len(subtypes)

    #controls are always drawn from N(0, 1) distribution
    mean_controls                       = np.array([0]   * N_biomarkers)
    std_controls                        = np.array([0.25] * N_biomarkers)

    #mean and variance for cases
    #if using mixture_GMM, use normal distribution with mean 1 and std. devs sampled from a range
    if mixture_style == 'mixture_GMM':
        mean_cases                      = np.array([1.5] * N_biomarkers)
        std_cases                       = np.random.uniform(0.25, 0.50, N_biomarkers)
    #if using mixture_KDE, use log normal with mean 0.5 and std devs sampled from a range
    elif mixture_style == 'mixture_KDE':
        mean_cases                      = np.array([0.5] * N_biomarkers)
        std_cases                       = np.random.uniform(0.2, 0.5, N_biomarkers)

    data                                = np.zeros((N_subjects, N_biomarkers))
    data_denoised                       = np.zeros((N_subjects, N_biomarkers))

    stages                              = stages.astype(int)

    #loop over all subjects, creating measurment for each biomarker based on what subtype and stage they're in
    for i in range(N_subjects):

        S_i                             = gt_ordering[subtypes[i], :].astype(int)
        stage_i                         = stages[i].item()

        #fill in with ABNORMAL values up to the subject's stage
        for j in range(stage_i):

            if      mixture_style == 'mixture_KDE':
                sample_j                = np.random.lognormal(mean_cases[S_i[j]], std_cases[S_i[j]])
            elif    mixture_style == 'mixture_GMM':
                sample_j                = np.random.normal(mean_cases[S_i[j]], std_cases[S_i[j]])

            data[i, S_i[j]]             = sample_j
            data_denoised[i, S_i[j]]    = mean_cases[S_i[j]]

        # fill in with NORMAL values from the subject's stage+1 to last stage
        for j in range(stage_i, N_biomarkers):
            data[i, S_i[j]]             = np.random.normal(mean_controls[S_i[j]], std_controls[S_i[j]])
            data_denoised[i, S_i[j]]    = mean_controls[S_i[j]]


    return data, data_denoised #, stage_value
