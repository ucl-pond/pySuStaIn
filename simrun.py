###
# pySuStaIn: Python translation of Matlab version of SuStaIn algorithm (https://www.nature.com/articles/s41467-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
###
import numpy as np
from matplotlib import pyplot as plt
from simfuncs import generate_random_sustain_model, generate_data_sustain
from funcs import run_sustain_algorithm, cross_validate_sustain_model

import math

def main():

    validate = False
    
    N = 5
    M = 20
    N_S_gt = 3
    Z_vals = np.array([[1,2,3]]*N)
    IX_vals = np.array([[x for x in range(N)]]*3).T
    Z_max = np.array([5]*N)
    
    stage_zscore = np.array([y for x in Z_vals.T for y in x])
    stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
    stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
    stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
    
    min_biomarker_zscore = [0]*N;
    max_biomarker_zscore = Z_max;
    std_biomarker_zscore = [1]*N;
    SuStaInLabels = []
    SuStaInStageLabels = []
    for i in range(N):
        SuStaInLabels.append('Biomarker '+str(i))
    for i in range(len(stage_zscore)):
        SuStaInStageLabels.append('B'+str(stage_biomarker_index[i])+' - Z'+str(stage_zscore[i]))

    gt_f = [1] + [0.5*x for x in range(N_S_gt-1)]
    gt_f = [x/sum(gt_f) for x in gt_f][::-1]
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
    """
    numY, numX = (int(math.ceil(math.sqrt(data.shape[1]))),
                  int(round(math.sqrt(data.shape[1]))))
    fig, ax = plt.subplots(numX, numY)
    for i in range(data.shape[1]):
        ax[int(math.floor((i)/numY)), int(i % numY)].hist(data[:,i])
    print np.min(data),np.max(data)
    plt.show()
    """
    #    N_startpoints = 25
    #    N_S_max = 3
    #    N_iterations_MCMC = int(1e6)
    N_startpoints = 2
    N_S_max = 3
    N_iterations_MCMC = int(1e2)
    
    likelihood_flag = 'Exact'
    output_folder = 'simulateddataResults'
    dataset_name = 'simulateddata'
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
