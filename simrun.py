###
# PySustain: Python translation of Matlab version of SuStaIn algorithm (doi:10.1101/236604)
###
import numpy as np
from simfuncs import generate_random_sustain_model, generate_data_sustain
from funcs import run_sustain_algorithm

def main():

    N = 10
    M = 500
    N_S_gt = 1
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

    data, data_denoised, stage_value = generate_data_sustain(subtypes,stages,gt_sequence,
                                                             min_biomarker_zscore,max_biomarker_zscore,
                                                             std_biomarker_zscore,stage_zscore,
                                                             stage_biomarker_index)
    N_startpoints = 25
    N_S_max = 3
    N_iterations_MCMC = int(1e6)
    likelihood_flag = 'Exact'
    output_folder = 'simulateddataResults'
    dataset_name = 'simulateddata'
    run_sustain_algorithm(data,
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

if __name__ == '__main__':
    np.random.seed(42)
    main()
