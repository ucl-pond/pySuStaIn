###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk)
###
import numpy as np
from matplotlib import pyplot as plt
from simfuncs import generate_random_sustain_model, generate_data_sustain
from funcs import run_sustain_algorithm, cross_validate_sustain_model

import math

def main():
    # cross-validation
    validate = False
    
    N = 5  # number of biomarkers
    M = 100 # number of observations ( e.g. subjects )
    N_S_gt = 3 # number of ground truth subtypes
    Z_vals = np.array([[1,2,3]]*N) # Z-scores for each biomarker
    IX_vals = np.array([[x for x in range(N)]]*3).T
    Z_max = np.array([5]*N) # maximum z-score
    stage_zscore = np.array([y for x in Z_vals.T for y in x])
    stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
    stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    min_biomarker_zscore = [0]*N;
    max_biomarker_zscore = Z_max;
    std_biomarker_zscore = [1]*N;

    SuStaInLabels = []
    SuStaInStageLabels = []
    # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
    for i in range(N):
        SuStaInLabels.append( 'Biomarker '+str(i))
    for i in range(len(stage_zscore)):
        SuStaInStageLabels.append('B'+str(stage_biomarker_index[i])+' - Z'+str(stage_zscore[i]))

    gt_f = [1+0.5*x for x in range(N_S_gt)]
    gt_f = [x/sum(gt_f) for x in gt_f][::-1]
    # ground truth sequence for each subtype
    gt_sequence = generate_random_sustain_model(stage_zscore,stage_biomarker_index,N_S_gt)

    N_k_gt = np.array(stage_zscore).shape[1]+1
    subtypes = np.random.choice(range(N_S_gt),M,replace=True,p=gt_f)
    stages = np.ceil(np.random.rand(M,1)*(N_k_gt+1))-1
    data, data_denoised, stage_value = generate_data_sustain(subtypes,
                                                             stages,
                                                             gt_sequence,
                                                             min_biomarker_zscore,
                                                             max_biomarker_zscore,
                                                             std_biomarker_zscore,
                                                             stage_zscore,
                                                             stage_biomarker_index)
    # number of starting points 
    N_startpoints = 25 
    # maximum number of subtypes 
    N_S_max = 3
    N_iterations_MCMC = int(1e6)
    
    likelihood_flag = 'Approx'
    output_folder = 'test3'
    dataset_name = 'test'
    
    samples_sequence, samples_f = run_sustain_algorithm(data,
                                                        min_biomarker_zscore,
                                                        max_biomarker_zscore,
                                                        std_biomarker_zscore,
                                                        stage_zscore,
                                                        stage_biomarker_index,
                                                        N_startpoints,
                                                        N_S_max,
                                                        N_iterations_MCMC,
                                                        likelihood_flag,
                                                        output_folder,
                                                        dataset_name)

    if validate:
        ### USER DEFINED INPUT: START
        # test_idxs: indices corresponding to 'data' for test set, with shape (N_folds, data.shape[0]/N_folds)
        # select_fold: index of a single fold from 'test_idxs'. For use if this code was to be run on multiple processors
        # target: stratification is done based on the labels provided here. For use with sklearn method 'StratifiedKFold'
        test_idxs = []
        select_fold = []
        target = []
        ### USER DEFINED INPUT: END
                
        cross_validate_sustain_model(data,
                                     test_idxs,
                                     min_biomarker_zscore,
                                     max_biomarker_zscore,
                                     std_biomarker_zscore,
                                     stage_zscore,
                                     stage_biomarker_index,
                                     N_startpoints,
                                     N_S_max,
                                     N_iterations_MCMC,
                                     likelihood_flag,
                                     output_folder,
                                     dataset_name,
                                     select_fold,
                                     target)

    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    main()
