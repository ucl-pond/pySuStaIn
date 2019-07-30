###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk)
###
import numpy as np
from scipy.stats import norm, multivariate_normal
from matplotlib import pyplot as plt
import csv
import os
from sklearn.model_selection import KFold, StratifiedKFold
import pickle
from pathlib import Path
import multiprocessing
from functools import partial


def prepare_data(pat_data, hc_data):
    '''
    prepares the data by normalising tha patient data, checking signs, and missing data
    for further modelling
    inputs:
    =======
    pat_data: an MxN Numpy array that corresponds to M observations for N biomarkers for patients
    hc_data: an LxN Numpy array from healhty population that has L observations for N biomarkers
    
    Outputs:
    =======
    z_score_array: an MxN numpy array of the Z-scores

    '''
    z_score_array = np.zeros( pat_data.shape )
    if ((np.nunsum( pat_data ) > 0 ) or (np.nunsum(hc_data) > 0 )):
        raise Exception("Patient or healthy control data have NaNs. Please remove or impute." )
    #calculate z-score
    for column_number, (sample_mean, sample_std) in enumerate( zip( np.mean( hc_data, axis = 0),
                                                                    np.std( hc_data, axis = 0 ))):
        z_score_array[ :, column_number - 1 ] = ( pat_data[:,column_number - 1] - sample_mean)/ sample_std
        #WIP: check signs needs testing
        if sample_mean > np.mean( pat_data[:, column_number - 1] ):
            print('WARNING: flipping sign for column number '  + str( column_number - 1 ))
            z_score_array[ :, column_number - 1 ] =  -1 * z_score_array[ :, column_number - 1 ]
    return z_score_array

def run_sustain_algorithm(data,
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
                          n_mcmc_opt_its,
                          num_cores,
                          covar = False):
    '''
    Runs the sustain algorithm
    Inputs:
    ======
    data: Z-score normalised data
    min_biomarker_zscore:
    max_biomarker_zscore:
    std_biomarker_zscore:
    stage_zscore:
    stage_biomarker_index:
    N_startpoints: number of starting points, currently 25 is default
    N_S_max: maximum number of subtypes
    N_iterations_MCMC: number of iterations (default: 1e6)
    likelihood_flag: Approximate or exact likelihood 
    output_folder: folder to dump results
    dataset_name: name that will be used in outputs
    num_cores: number of cpu cores 
    covar: covariance matrix between variables from healthy control data
    
    Outputs
    =======

    '''

    ml_sequence_prev_EM = []
    ml_f_prev_EM = []

    fig0, ax0 = plt.subplots()
    for s in range(N_S_max):

        pickle_filename_s = output_folder + '/' + dataset_name + '_subtype' + str(s) + '.pickle'
        pickle_filepath = Path(pickle_filename_s)
        # FIXME: for the moment we will always run the code, to avoid confusion
        #        if pickle_filepath.exists():
        if False:
            print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s) + " subtype.")
            pickle_file = open(pickle_filename_s, 'rb')

            save_variables = pickle.load(pickle_file)

            samples_sequence = save_variables["samples_sequence"]
            samples_f = save_variables["samples_f"]
            biomarker_labels = save_variables["biomarker_labels"]
            stage_zscore = save_variables["stage_zscore"]
            stage_biomarker_index = save_variables["stage_biomarker_index"]
            N_S_max = save_variables["N_S_max"]
            samples_likelihood = save_variables["samples_likelihood"]
            ml_sequence_EM          = save_variables["ml_sequence_EM"]
            ml_sequence_prev_EM     = save_variables["ml_sequence_prev_EM"]
            ml_f_EM                 = save_variables["ml_f_EM"]
            ml_f_prev_EM            = save_variables["ml_f_prev_EM"]

            pickle_file.close()
        else:
            print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s) + " subtype.")

            ml_sequence_EM, ml_f_EM, ml_likelihood_EM, ml_sequence_mat_EM, ml_f_mat_EM, ml_likelihood_mat_EM = estimate_ml_sustain_model_nplus1_clusters(
                data,
                min_biomarker_zscore,
                max_biomarker_zscore,
                std_biomarker_zscore,
                stage_zscore,
                stage_biomarker_index,
                ml_sequence_prev_EM,
                ml_f_prev_EM,
                N_startpoints,
                likelihood_flag,
                num_cores,
                covar)
            seq_init = ml_sequence_EM
            f_init = ml_f_EM
            ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = estimate_uncertainty_sustain_model(
                data,
                min_biomarker_zscore,
                max_biomarker_zscore,
                std_biomarker_zscore,
                stage_zscore,
                stage_biomarker_index,
                seq_init,
                f_init,
                N_iterations_MCMC,
                likelihood_flag,
                n_mcmc_opt_its)
            ml_sequence_prev_EM = ml_sequence_EM
            ml_f_prev_EM = ml_f_EM

        # write results if they don't exist
        biomarker_labels = np.array([str(x) for x in range(data.shape[1])])
        # max like subtype and stage / subject
        ml_subtype, ml_stage = subtype_and_stage_individuals_selectsubtypethenstage(data,
                                                                                    samples_sequence,
                                                                                    samples_f,
                                                                                    min_biomarker_zscore,
                                                                                    max_biomarker_zscore,
                                                                                    std_biomarker_zscore,
                                                                                    stage_zscore,
                                                                                    stage_biomarker_index,
                                                                                    likelihood_flag,
                                                                                    1000,
                                                                                    s,
                                                                                    output_folder,
                                                                                    dataset_name)
        if not pickle_filepath.exists():

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            save_variables = {}
            save_variables["samples_sequence"] = samples_sequence
            save_variables["samples_f"] = samples_f
            save_variables["biomarker_labels"] = biomarker_labels
            save_variables["stage_zscore"] = stage_zscore
            save_variables["stage_biomarker_index"] = stage_biomarker_index
            save_variables["N_S_max"] = N_S_max
            save_variables["samples_likelihood"] = samples_likelihood
            save_variables["ml_subtype"] = ml_subtype
            save_variables["ml_stage"] = ml_stage

            pickle_file = open(pickle_filename_s, 'wb')
            pickle_output = pickle.dump(save_variables, pickle_file)
            pickle_file.close()
        # plot results
        fig, ax = plot_sustain_model(samples_sequence,
                                     samples_f,
                                     biomarker_labels,
                                     stage_zscore,
                                     stage_biomarker_index,
                                     output_folder,
                                     dataset_name,
                                     s,
                                     samples_likelihood)

        ax0.plot(range(N_iterations_MCMC),samples_likelihood)
        plt.savefig( 'MCMC_likelihood' + str(N_iterations_MCMC) + '.png', bbox_inches = 'tight')

    return samples_sequence, samples_f

def estimate_ml_sustain_model_nplus1_clusters(data,
                                              min_biomarker_zscore,
                                              max_biomarker_zscore,
                                              std_biomarker_zscore,
                                              stage_zscore,
                                              stage_biomarker_index,
                                              ml_sequence_prev,
                                              ml_f_prev,
                                              N_startpoints,
                                              likelihood_flag, 
                                              num_cores,
                                              covar):
    '''
    Given the previous SuStaIn model, estimate the next model in the
    hierarchy (i.e. number of subtypes goes from N to N+1)
    
    Inputs: 
    =======

    data: *Positive* z-scores matrix (subjects x number of biomarkers) 
    min_biomarker_zscore: a minimum z-score for each biomarker (usually zero
    for all markers)
    dim: 1 x number of biomarkers

    max_biomarker_zscore - a maximum z-score for each biomarker - reached at
    the final stage of the linear z-score model
      dim: 1 x number of biomarkers

    std_biomarker_zscore - the standard devation of each biomarker z-score
    (should be 1 for all markers)
      dim: 1 x number of biomarkers

    stage_zscore and stage_biomarker_index give the different z-score stages
    for the linear z-score model, i.e. the index of the different z-scores
    for each biomarker

    stage_zscore - the different z-scores of the model
      dim: 1 x number of z-score stages

    stage_biomarker_index - the index of the biomarker that the corresponding
    entry of stage_zscore is referring to - !important! ensure biomarkers are
    indexed s.t. they correspond to columns 1 to number of biomarkers in your
    data
      dim: 1 x number of z-score stages

    ml_sequence_prev - the ordering of the stages for each subtype from the
    previous SuStaIn model
      dim: number of subtypes x number of z-score stages

    ml_f_prev - the proportion of individuals belonging to each subtype from
    the previous SuStaIn model
      dim: number of subtypes x 1

    N_startpoints: the number of start points for the fitting

    likelihood_flag: whether to use an exact method of inference - when set
    to 'Exact', the exact method is used, the approximate method is used for
    all other settings
   
    num_cores: number of cores to use in multiprocessing to parallelise
    Outputs:
    =======

    ml_sequence: the ordering of the stages for each subtype for the next
    SuStaIn model in the hierarchy

    ml_f: the most probable proportion of individuals belonging to each
    subtype for the next SuStaIn model in the hierarchy

    ml_likelihood: the likelihood of the most probable SuStaIn model for the
    next SuStaIn model in the hierarchy

    previous outputs _mat: same as before but for each start point

    '''

    N_S = len(ml_sequence_prev) + 1
    if N_S == 1:
        # If the number of subtypes is 1, fit a single linear z-score model 
        print( 'Finding ML solution to 1 cluster problem' )
        ml_sequence,ml_f,ml_likelihood,ml_sequence_mat,ml_f_mat,ml_likelihood_mat = find_ml_linearzscoremodel(data,
                                                                                                              min_biomarker_zscore,
                                                                                                              max_biomarker_zscore,
                                                                                                              std_biomarker_zscore,
                                                                                                              stage_zscore,
                                                                                                              stage_biomarker_index,
                                                                                                              N_startpoints,
                                                                                                              likelihood_flag, 
                                                                                                              num_cores  )
        print('Overall ML likelihood is',ml_likelihood)
    else:
        # If the number of subtypes is greater than 1, go through each subtype
        # in turn and try splitting into two subtypes
        _, _, _, p_sequence, _ = calculate_likelihood_mixturelinearzscoremodels(data,
                                                                                min_biomarker_zscore,
                                                                                max_biomarker_zscore,
                                                                                std_biomarker_zscore,
                                                                                stage_zscore,
                                                                                stage_biomarker_index,
                                                                                ml_sequence_prev,
                                                                                ml_f_prev,
                                                                                likelihood_flag)
        ml_sequence_prev = ml_sequence_prev.reshape(ml_sequence_prev.shape[0], ml_sequence_prev.shape[1])
        p_sequence = p_sequence.reshape(p_sequence.shape[0], N_S - 1)
        p_sequence_norm = p_sequence / np.tile(np.sum(p_sequence, 1).reshape(len(p_sequence), 1), (N_S - 1))
        # Assign individuals to a subtype (cluster) based on the previous model
        ml_cluster_subj = np.zeros((len(data), 1))
        for m in range(len(data)):
            ix = np.argmax(p_sequence_norm[m, :]) + 1
            ml_cluster_subj[
                m] = ix  # FIXME: should check this always works, as it differs to the Matlab code, which treats ix as an array

        ml_likelihood = -np.inf
        for ix_cluster_split in range(N_S - 1):
            this_N_cluster = sum(ml_cluster_subj == int(ix_cluster_split + 1))
            if this_N_cluster > 1:
                # Take the data from the individuals belonging to a particular
                # cluster and fit a two subtype model
                print('Splitting cluster', ix_cluster_split + 1, 'of', N_S - 1)
                data_split = data[(ml_cluster_subj == int(ix_cluster_split + 1)).reshape(len(data), )]
                print(' + Resolving 2 cluster problem')
                this_ml_sequence_split, _, _, _, _, _ = find_ml_mixture2linearzscoremodels(data_split,
                                                                                           min_biomarker_zscore,
                                                                                           max_biomarker_zscore,
                                                                                           std_biomarker_zscore,
                                                                                           stage_zscore,
                                                                                           stage_biomarker_index,
                                                                                           N_startpoints,
                                                                                           likelihood_flag, 
                                                                                           num_cores,
                                                                                           covar)
                # Use the two subtype model combined with the other subtypes to
                # inititialise the fitting of the next SuStaIn model in the
                # hierarchy
                this_seq_init = ml_sequence_prev.copy()  # have to copy or changes will be passed to ml_sequence_prev
                this_seq_init[ix_cluster_split] = (this_ml_sequence_split[0]).reshape(this_ml_sequence_split.shape[1])
                this_seq_init = np.hstack((this_seq_init.T, this_ml_sequence_split[1])).T
                this_f_init = np.array([1.] * N_S) / float(N_S)
                print(' + Finding ML solution from hierarchical initialisation')
                this_ml_sequence, this_ml_f, this_ml_likelihood, this_ml_sequence_mat, this_ml_f_mat, this_ml_likelihood_mat = find_ml_mixturelinearzscoremodels(data,
                                                                                                                                                                 min_biomarker_zscore,
                                                                                                                                                                 max_biomarker_zscore,
                                                                                                                                                                 std_biomarker_zscore,
                                                                                                                                                                 stage_zscore,
                                                                                                                                                                 stage_biomarker_index,
                                                                                                                                                                 this_seq_init,
                                                                                                                                                                 this_f_init,
                                                                                                                                                                 N_startpoints,
                                                                                                                                                                 likelihood_flag,
                                                                                                                                                                 num_cores)
                # Choose the most probable SuStaIn model from the different
                # possible SuStaIn models initialised by splitting each subtype
                # in turn
                # FIXME: these arrays have an unnecessary additional axis with size = N_startpoints - remove it further upstream
                if this_ml_likelihood[0] > ml_likelihood:
                    ml_likelihood = this_ml_likelihood[0]
                    ml_sequence = this_ml_sequence[:, :, 0]
                    ml_f = this_ml_f[:, 0]
                    ml_likelihood_mat = this_ml_likelihood_mat[0]
                    ml_sequence_mat = this_ml_sequence_mat[:, :, 0]
                    ml_f_mat = this_ml_f_mat[:, 0]
                print('- ML likelihood is', this_ml_likelihood[0])
            else:
                print('Cluster', ix_cluster_split + 1, 'of', N_S - 1, 'too small for subdivision')
        print('Overall ML likelihood is', ml_likelihood)

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

def find_ml_linearzscoremodel(data,
                              min_biomarker_zscore,
                              max_biomarker_zscore,
                              std_biomarker_zscore,
                              stage_zscore,
                              stage_biomarker_index,
                              N_startpoints,
                              likelihood_flag,
                              num_cores
                              ):
    ''' 
    Fit a linear z-score model
    
    INPUTS: 
    ======

     data: !important! needs to be (positive) z-scores! 
       dim: number of subjects x number of biomarkers

     min_biomarker_zscore: a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers

     max_biomarker_zscore - a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers
     std_biomarker_zscore - the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers
     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker
     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages
     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages
     N_startpoints - the number of start points for the fitting
     likelihood_flag - whether to use an exact method of inference - when set
     to 'Exact', the exact method is used, the approximate method is used for
     all other settings
    
    OUTPUTS:
    ======== 

     ml_sequence - the ordering of the stages for each subtype
     ml_f - the most probable proportion of individuals belonging to each
     subtype
     ml_likelihood - the likelihood of the most probable SuStaIn model
     previous outputs _mat - same as before but for each start point

    '''
    ml_sequence_mat = np.zeros((1,stage_zscore.shape[1],N_startpoints))
    ml_f_mat = np.zeros((1,N_startpoints))
    ml_likelihood_mat = np.zeros(N_startpoints)

    pool = multiprocessing.Pool(num_cores)
    # instantiate function as class to pass to pool.map
    # first calculate array of sequences - do this first or same random number will be used simultaneously on each processor
    copier = partial(initialise_sequence_linearzscoremodel,
                     stage_zscore=stage_zscore,
                     stage_biomarker_index=stage_biomarker_index)
    # will return shape (N_startpoints, 1)
    seq_mat = np.array(pool.map(copier, range(N_startpoints)))
    # now optimise
    copier = partial(perform_em_mixturelinearzscoremodels,
                     data=data,
                     min_biomarker_zscore=min_biomarker_zscore,
                     max_biomarker_zscore=max_biomarker_zscore,
                     std_biomarker_zscore=std_biomarker_zscore,
                     stage_zscore=stage_zscore,
                     stage_biomarker_index=stage_biomarker_index,
                     current_sequence=seq_mat,
                     current_f=[1],
                     likelihood_flag=likelihood_flag,
                     covar)
    # will return shape (N_startpoints, 6)
    par_mat = np.array(pool.map(copier, range(N_startpoints)))
    # distribute to local matrices
    for i in range(N_startpoints):
        ml_sequence_mat[:, :, i] = par_mat[i, 0]
        ml_f_mat[:, i] = par_mat[i, 1]
        ml_likelihood_mat[i] = par_mat[i, 2]    
    ix = np.argmax(ml_likelihood_mat)
    ml_sequence = ml_sequence_mat[:, :, ix]
    ml_f = ml_f_mat[:, ix]
    ml_likelihood = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

def initialise_sequence_linearzscoremodel(dummy,
                                          stage_zscore,
                                          stage_biomarker_index):
    '''
     Randomly initialises a linear z-score model ensuring that the biomarkers
     are monotonically increasing
    
    Inputs:
    =======

     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker
     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages
     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages
    
    Outputs:
    =======

     S - a random linear z-score model under the condition that each biomarker
     is monotonically increasing

    '''

    # ensure randomness across parallel processes by seeding numpy with process id
    current = multiprocessing.current_process()
    if current._identity:
        np.random.seed()
    
    N = np.array(stage_zscore).shape[1]
    S = np.zeros(N)
    for i in range(N):
        IS_min_stage_zscore = np.array([False] * N)
        possible_biomarkers = np.unique(stage_biomarker_index)
        for j in range(len(possible_biomarkers)):
            IS_unselected = [False] * N
            for k in set(range(N)) - set(S[:i]):
                IS_unselected[k] = True
            this_biomarkers = np.array([(np.array(stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) + (
                        np.array(IS_unselected) == 1).astype(int)]) == 2
            if not np.any(this_biomarkers):
                this_min_stage_zscore = 0
            else:
                this_min_stage_zscore = min(stage_zscore[this_biomarkers])
            if (this_min_stage_zscore):
                temp = ((this_biomarkers.astype(int) + (stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                temp = temp.reshape(len(temp), )
                IS_min_stage_zscore[temp] = True
        events = np.array(range(N))
        possible_events = np.array(events[IS_min_stage_zscore])
        this_index = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
        S[i] = possible_events[int(this_index)]
    S = S.reshape(1, len(S))
    return S

def perform_em_mixturelinearzscoremodels(dummy,
                                         data,
                                         min_biomarker_zscore,
                                         max_biomarker_zscore,
                                         std_biomarker_zscore,
                                         stage_zscore,
                                         stage_biomarker_index,
                                         current_sequence,
                                         current_f,
                                         likelihood_flag,
                                         covar=False):
    '''
    Performs an E-M procedure to estimate parameters of the SuStaIn model.
    
    Inputs
    ====== 
     
    Output
    ======

    ml_sequence:

    ml_f:

    ml_likelihood:

    samples_sequence:

    samples_f:

    samples_likelihood:

    covar: covariance matrix

    ''' 

    # ensure randomness across parallel processes by seeding numpy with process id
    current = multiprocessing.current_process()
    if current._identity:
        np.random.seed()
        # this should only be the case if called in parallel - then there will be a current_sequence for each startpoint
        if len(current_sequence.shape)==3:
            current_sequence = current_sequence[dummy]
        if len(np.array(current_f).shape)==2:
            current_f = current_f[dummy]


    MaxIter = 100

    N = stage_zscore.shape[1]
    N_S = current_sequence.shape[0]
    current_likelihood, _, _, _, _ = calculate_likelihood_mixturelinearzscoremodels(data,
                                                                                    min_biomarker_zscore,
                                                                                    max_biomarker_zscore,
                                                                                    std_biomarker_zscore,
                                                                                    stage_zscore,
                                                                                    stage_biomarker_index,
                                                                                    current_sequence,
                                                                                    current_f,
                                                                                    likelihood_flag)
    terminate = 0
    iteration = 0
    convergence_threshold = 1e-6

    samples_sequence = np.nan * np.ones(( MaxIter, N,N_S ))
    samples_f = np.nan * np.ones(( MaxIter, N_S ))
    samples_likelihood = np.nan * np.ones((MaxIter,1))
    
    samples_sequence[ 0, :, : ] = current_sequence.reshape(current_sequence.shape[1],current_sequence.shape[0])
    current_f = np.array( current_f ).reshape(len(current_f))
    samples_f[ 0, : ] = current_f
    samples_likelihood[0] = current_likelihood
    while terminate==0:
        print('++ iteration',iteration)
        candidate_sequence,candidate_f,candidate_likelihood = optimise_parameters_mixturelinearzscoremodels(data,
                                                                                                             min_biomarker_zscore,
                                                                                                             max_biomarker_zscore,
                                                                                                             std_biomarker_zscore,
                                                                                                             stage_zscore,
                                                                                                             stage_biomarker_index,
                                                                                                             current_sequence,
                                                                                                             current_f,
                                                                                                             likelihood_flag,
                                                                                                             covar)
        HAS_converged = np.fabs((candidate_likelihood-current_likelihood)/max(candidate_likelihood,current_likelihood)) < convergence_threshold
        if HAS_converged:
            print('EM converged in', iteration + 1, 'iterations')
            terminate = 1
        else:
            if candidate_likelihood > current_likelihood:
                current_sequence = candidate_sequence
                current_f = candidate_f
                current_likelihood = candidate_likelihood
        samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
        samples_f[iteration, :] = current_f
        samples_likelihood[iteration] = current_likelihood

        if iteration == (MaxIter - 1):
            terminate = 1
        iteration = iteration + 1
    ml_sequence = current_sequence
    ml_f = current_f
    ml_likelihood = current_likelihood
    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

def calculate_likelihood_mixturelinearzscoremodels(data,
                                                   min_biomarker_zscore,
                                                   max_biomarker_zscore,
                                                   std_biomarker_zscore,
                                                   stage_zscore,
                                                   stage_biomarker_index,
                                                   S,
                                                   f,
                                                   likelihood_flag):
    '''
     Computes the likelihood of a mixture of linear Z-score models using either
     an approximate method (faster, default setting) or an exact method
    
    Inputs: 
    ======

     data - !important! needs to be (positive) z-scores! 
       dim: number of subjects x number of biomarkers

     min_biomarker_zscore - a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers

     max_biomarker_zscore - a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers

     std_biomarker_zscore - the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers

     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker

     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages

     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages

     S - the current ordering of the z-score stages for each subtype
       dim: number of subtypes x number of z-score stages

     f - the current proportion of individuals belonging to each subtype
       dim: number of subtypes x 1

     likelihood_flag - whether to use an exact method of inference - when set
     to 'Exact', the exact method is used, the approximate method is used for
     all other settings
    
    Outputs:
    ========

     loglike - the log-likelihood of the current model
     total_prob_subj - the total probability of the current SuStaIn model for
     each subject
     total_prob_stage - the total probability of each stage in the current
     SuStaIn model
     total_prob_cluster - the total probability of each subtype in the current
     SuStaIn model
     p_perm_k - the probability of each subjects data at each stage of each
     subtype in the current SuStaIn model
    '''

    M = data.shape[0]
    N_S = S.shape[0]
    N = stage_zscore.shape[1]
    f = np.array(f).reshape(N_S, 1, 1)
    f_val_mat = np.tile(f, (1, N + 1, M))
    f_val_mat = np.transpose(f_val_mat, (2, 1, 0))

    p_perm_k = np.zeros((M, N + 1, N_S))

    for s in range(N_S):
        if likelihood_flag=='Approx':
            p_perm_k[:,:,s] = calculate_likelihood_stage_linearzscoremodel_approx(data,
                                                                                  min_biomarker_zscore,
                                                                                  max_biomarker_zscore,
                                                                                  std_biomarker_zscore,
                                                                                  stage_zscore,
                                                                                  stage_biomarker_index,
                                                                                  S[s],
                                                                                  covar)
        else:
            # FIXME: test this function
            p_perm_k[:,:,s] = calculate_likelihood_stage_linearzscoremodel(data,
                                                                           min_biomarker_zscore,
                                                                           max_biomarker_zscore,
                                                                           std_biomarker_zscore,
                                                                           stage_zscore,
                                                                           stage_biomarker_index,
                                                                           S[s])
    total_prob_cluster = np.squeeze(np.sum(p_perm_k*f_val_mat,1))
    total_prob_stage = np.sum(p_perm_k*f_val_mat,2)
    total_prob_subj = np.sum(total_prob_stage,1)
    loglike = sum(np.log(total_prob_subj + 1e-250))

    return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

def calculate_likelihood_stage_linearzscoremodel_approx(data,
                                                        min_biomarker_zscore,
                                                        max_biomarker_zscore,
                                                        std_biomarker_zscore,
                                                        stage_zscore,
                                                        stage_biomarker_index,
                                                        S,
                                                        covar=False
                                                        ):
    '''
     Computes the likelihood of a single linear z-score model using an
     approximation method (faster)
    Inputs:
    =======
     data - !important! needs to be (positive) z-scores!
       dim: number of subjects x number of biomarkers
     min_biomarker_zscore - a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers
     max_biomarker_zscore - a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers
     std_biomarker_zscore - the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers
     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker
     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages
     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages
     S - the current ordering of the z-score stages for a particular subtype
       dim: 1 x number of z-score stages
     covar: covariance matrix calculated from the healthy control data

    Outputs:
    ========

     p_perm_k - the probability of each subjects data at each stage of a particular subtype
     in the SuStaIn model

    '''
    N = stage_biomarker_index.shape[1]
    S_inv = np.array([ 0 ] * N)
    S_inv[S.astype(int)] = np.arange(N)
    possible_biomarkers = np.unique(stage_biomarker_index)
    B = len(possible_biomarkers)
    point_value = np.zeros((B, N + 2))

    # all the arange you'll need below
    arange_N = np.arange(N + 2)

    for i in range(B):
        b = possible_biomarkers[i]
        event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
        event_value = np.concatenate(
            [[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])
        for j in range(len(event_location) - 1):

            if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                temp = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                N_j = event_location[j + 1] - event_location[j] + 2
                point_value[i, temp] = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])
            else:
                temp = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                N_j = event_location[j + 1] - event_location[j] + 1
                point_value[i, temp] = linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

    stage_value = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]
    M = data.shape[0]
    p_perm_k = np.zeros((M, N + 1))

    # optimised likelihood calc - take log and only call np.exp once after loop
    # FIXME: put this flag outside, or do an automated check
    if covar:
        # FIXME: this should be calculated from control data
        sigmat = np.tile(np.outer(std_biomarker_zscore, std_biomarker_zscore), (M, 1, 1))
    else:
        sigmat = np.tile(std_biomarker_zscore, (M, 1))    
        factor = np.log(1. / np.sqrt(np.pi * 2.0) * sigmat)
        coeff = np.log(1. / float(N + 1))

    stage_value_tiled = np.tile(stage_value, (M, 1))
    N_biomarkers = stage_value.shape[0]
    for j in range(N + 1):
        stage_value_tiled_j = stage_value_tiled[:, j].reshape(M, N_biomarkers)
        if covar:
            temp = []
            for i in range(stage_value_tiled_j.shape[0]):
                temp.append(1./(N+1.)*multivariate_normal.pdf(data[i,:],stage_value_tiled_j[i,:],sigmat[i]))
            p_perm_k[:, j] = temp
        else:
            x = (data - stage_value_tiled_j) / sigmat
            p_perm_k[:, j] = coeff + np.sum(factor - .5 * np.square(x), 1)        
        
    p_perm_k = np.exp(p_perm_k)

    return p_perm_k

def calculate_likelihood_stage_linearzscoremodel(data,
                                                 min_biomarker_zscore,
                                                 max_biomarker_zscore,
                                                 std_biomarker_zscore,
                                                 stage_zscore,
                                                 stage_biomarker_index,
                                                 S):
    ########################################################
    ########################################################
    # FIXME: this code hasn't been tested yet
    ########################################################
    ########################################################
    # Computes the likelihood of a single linear z-score model using an exact method (slower)
    #Inputs: 

    # data - !important! needs to be (positive) z-scores! 
    #   dim: number of subjects x number of biomarkers
    # min_biomarker_zscore - a minimum z-score for each biomarker (usually zero
    # for all markers)
    #   dim: 1 x number of biomarkers
    # max_biomarker_zscore - a maximum z-score for each biomarker - reached at
    # the final stage of the linear z-score model
    #   dim: 1 x number of biomarkers
    # std_biomarker_zscore - the standard devation of each biomarker z-score
    # (should be 1 for all markers)
    #   dim: 1 x number of biomarkers
    # stage_zscore and stage_biomarker_index give the different z-score stages
    # for the linear z-score model, i.e. the index of the different z-scores
    # for each biomarker
    # stage_zscore - the different z-scores of the model
    #   dim: 1 x number of z-score stages
    # stage_biomarker_index - the index of the biomarker that the corresponding
    # entry of stage_zscore is referring to - !important! ensure biomarkers are
    # indexed s.t. they correspond to columns 1 to number of biomarkers in your
    # data
    #   dim: 1 x number of z-score stages
    # S - the current ordering of the z-score stages for a particular subtype
    #   dim: 1 x number of z-score stages
    #
    # OUTPUTS:
    # p_perm_k - the probability of each subjects data at each stage of a
    # particular subtype in the SuStaIn model
    N = stage_biomarker_index.shape[1]
    S_inv = np.array([0] * N)
    S_inv[S.astype(int)] = np.arange(N)
    possible_biomarkers = np.unique(stage_biomarker_index)
    B = len(possible_biomarkers)
    tau_val = np.linspace(0, 1, N + 2)
    point_value = np.zeros(B, N + 2)
    for i in range(B):
        b = possible_biomarkers[i]
        event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])
        event_value = np.concatenate(
            [[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])
        for j in range(len(event_location) - 1):
            if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                temp = np.arange(event_location[j], event_location[j + 1] + 2)
                point_value[i, temp] = np.linspace(event_value[j], event_value[j + 1],
                                                   event_location[j + 1] - event_location[j] + 2)
            else:
                temp = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                point_value[i, temp] = np.linspace(event_value[j], event_value[j + 1],
                                                   event_location[j + 1] - event_location[j] + 1)
    stage_initial_value = point_value[:, point_value.shape[1] - 1]
    stage_final_value = point_value[:, 1:]

    stage_initial_tau = tau_val[len(tav_val) - 1]
    stage_final_tau = tau_val[1:]

    stage_a = (stage_final_value - stage_initial_value) / np.tile(stage_final_tau - stage_initial_tau, (B, 1))
    stage_b = stage_initial_value - stage_a * np.tile(stage_initial_tau, (B, 1))
    stage_std = np.tile(std_biomarker_zscore.T, (1, N + 1))

    M = data.shape[0]

    iterative_mean = (np.tile(data[:, 1], (1, N + 1)) - np.tile(stage_b[0, :], (M, 1))) / np.tile(stage_a[0, :], (M, 1))
    iterative_std = np.tile(stage_std[0, :], (M, 1)) / np.tile(stage_a[0, :], (M, 1))
    iterative_kappa = np.ones(M, N + 1)
    for b in range(1, B):
        mu1 = iterative_mean
        mu2 = (np.tile(data[:, b], (1, N + 1)) - np.tile(stage_b[b, :], (M, 1))) / np.tile(stage_a[b, :], (M, 1))
        std1 = iterative_std
        std2 = np.tile(stage_std[b, :], (M, 1)) / np.tile(stage_a[b, :], (M, 1))
        cov1 = std1 ** 2
        cov2 = std2 ** 2
        munew = (cov1 ** -1 + cov2 ** -1) ** -1 * (cov1 ** -1 * mu1 + cov2 ** -1 * mu2);
        covnew = (cov1 ** -1 + cov2 ** -1) ** -1
        kappaval = normPdf(mu1, mu2, np.sqrt(cov1 + cov2))
        iterative_mean = munew
        iterative_std = np.sqrt(covnew)
        iterative_kappa = iterative_kappa * kappaval
    iterative_const = np.tile(prod(1. / np.fabs(stage_a), 0), (M, 1))
    cdf_diff_val = norm.cdf(np.tile(stage_final_tau, (M, 1)), iterative_mean, iterative_std) - norm.cdf(
        np.tile(stage_initial_tau, (M, 1)), iterative_mean, iterative_std)

    if iterative_const.shape[0] != iterative_kappa.shape[0]:
        print('Error!')
    if iterative_const.shape[0] != cdf_diff_val.shape[0]:
        print('Error!')
    if iterative_const.shape[1] != iterative_kappa.shape[1]:
        print('Error!')
    if iterative_const.shape[1] != cdf_diff_val.shape[1]:
        print('Error!')

    return iterative_const * iterative_kappa * cdf_diff_val

def optimise_parameters_mixturelinearzscoremodels(data,
                                                  min_biomarker_zscore,
                                                  max_biomarker_zscore,
                                                  std_biomarker_zscore,
                                                  stage_zscore,
                                                  stage_biomarker_index,
                                                  S_init,
                                                  f_init,
                                                  likelihood_flag,
                                                  covar = False):
    '''
    Optimises the parameters of the SuStaIn model
    
    Inputs
    ======

    data:

    min_biomarker_zscore:

    max_biomarker_zscore:

    std_biomarker_zscore:

    stage_zscore:

    stage_biomarker_index:

    S_init:

    f_init:

    likelihood_flag:

    covar: covariance matrix

    Outputs
    =======

    S_opt:

    f_opt:
    
    likelihood_opt:

    '''

    # ensure randomness across parallel processes by seeding numpy with process id
    current = multiprocessing.current_process()
    if current._identity:
        np.random.seed()
    
    M = data.shape[0]
    N_S = S_init.shape[0]
    N = stage_zscore.shape[1]

    S_opt = S_init.copy()  # have to copy or changes will be passed to S_init
    f_opt = np.array(f_init).reshape(N_S, 1, 1)
    f_val_mat = np.tile(f_opt, (1, N + 1, M))
    f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
    p_perm_k = np.zeros((M, N + 1, N_S))

    for s in range(N_S):
        if likelihood_flag=='Approx':
            p_perm_k[:,:,s] = calculate_likelihood_stage_linearzscoremodel_approx(data,
                                                                                  min_biomarker_zscore,
                                                                                  max_biomarker_zscore,
                                                                                  std_biomarker_zscore,
                                                                                  stage_zscore,
                                                                                  stage_biomarker_index,
                                                                                  S_opt[s], 
                                                                                  covar )
        else:
            p_perm_k[:, :, s] = calculate_likelihood_stage_linearzscoremodel(data,
                                                                             min_biomarker_zscore,
                                                                             max_biomarker_zscore,
                                                                             std_biomarker_zscore,
                                                                             stage_zscore,
                                                                             stage_biomarker_index,
                                                                             S_opt[s])            
    p_perm_k_weighted = p_perm_k * f_val_mat
    p_perm_k_norm = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (
    1, N + 1, N_S))  # the second summation axis is different to Matlab version
    f_opt = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
    f_val_mat = np.tile(f_opt, (1, N + 1, M))
    f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
    order_seq = np.random.permutation(N_S)  # this will produce different random numbers to Matlab
    for s in order_seq:
        order_bio = np.random.permutation(N)  # this will produce different random numbers to Matlab
        for i in order_bio:
            current_sequence = S_opt[s]
            current_location = np.array([0]*len(current_sequence))
            current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))
            selected_event = i
            move_event_from = current_location[selected_event]
            this_stage_zscore = stage_zscore[0, selected_event]
            selected_biomarker = stage_biomarker_index[0, selected_event]
            possible_zscores_biomarker = stage_zscore[stage_biomarker_index == selected_biomarker]
            # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
            min_filter = possible_zscores_biomarker < this_stage_zscore
            max_filter = possible_zscores_biomarker > this_stage_zscore
            events = np.array(range(N))
            if np.any(min_filter):
                min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                min_zscore_bound_event = events[((stage_zscore[0] == min_zscore_bound).astype(int) + (
                            stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
            else:
                move_event_to_lower_bound = 0
            if np.any(max_filter):
                max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                max_zscore_bound_event = events[((stage_zscore[0] == max_zscore_bound).astype(int) + (
                            stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                move_event_to_upper_bound = current_location[max_zscore_bound_event]
            else:
                move_event_to_upper_bound = N
            # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
            if move_event_to_lower_bound == move_event_to_upper_bound:
                possible_positions = np.array([0])
            else:
                possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
            possible_sequences = np.zeros((len(possible_positions), N))
            possible_likelihood = np.zeros((len(possible_positions), 1))
            possible_p_perm_k = np.zeros((M, N + 1, len(possible_positions)))
            for index in range(len(possible_positions)):
                current_sequence = S_opt[s]
                move_event_to = possible_positions[index]
                current_sequence = np.delete(current_sequence,move_event_from,0) # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to,N-1)]])
                possible_sequences[index,:] = new_sequence
            
                if likelihood_flag=='Approx':
                    possible_p_perm_k[:,:,index] = calculate_likelihood_stage_linearzscoremodel_approx(data,
                                                                                                       min_biomarker_zscore,
                                                                                                       max_biomarker_zscore,
                                                                                                       std_biomarker_zscore,
                                                                                                       stage_zscore,
                                                                                                       stage_biomarker_index,
                                                                                                       new_sequence,
                                                                                                       covar)
                else:
                    possible_p_perm_k[:, :, index] = calculate_likelihood_stage_linearzscoremodel(data,
                                                                                                  min_biomarker_zscore,
                                                                                                  max_biomarker_zscore,
                                                                                                  std_biomarker_zscore,
                                                                                                  stage_zscore,
                                                                                                  stage_biomarker_index,
                                                                                                  new_sequence)
                p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                total_prob_stage = np.sum(p_perm_k * f_val_mat, 2)
                total_prob_subj = np.sum(total_prob_stage, 1)
                possible_likelihood[index] = sum(np.log(total_prob_subj + 1e-250))

            possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
            max_likelihood = max(possible_likelihood)
            this_S = possible_sequences[possible_likelihood == max_likelihood, :]
            this_S = this_S[0, :]
            S_opt[s] = this_S
            this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
            p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]
        S_opt[s] = this_S
    p_perm_k_weighted = p_perm_k * f_val_mat
    p_perm_k_norm = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
    f_opt = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
    f_val_mat = np.tile(f_opt, (1, N + 1, M))
    f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
    f_opt = f_opt.reshape(N_S)
    total_prob_stage = np.sum(p_perm_k * f_val_mat, 2)
    total_prob_subj = np.sum(total_prob_stage, 1)

    likelihood_opt = sum(np.log(total_prob_subj + 1e-250))

    return S_opt, f_opt, likelihood_opt

def split_clusters(thread,
                   data,
                   min_biomarker_zscore,
                   max_biomarker_zscore,
                   std_biomarker_zscore,
                   stage_zscore,
                   stage_biomarker_index,
                   likelihood_flag,
                   N_S,
                   covar = False):
    # ensure randomness across parallel processes by seeding numpy with process id
    current = multiprocessing.current_process()
    np.random.seed()
    
    min_N_cluster = 0
    while min_N_cluster == 0:
        cluster_assignment = np.array([np.ceil(x) for x in N_S * np.random.rand(data.shape[0])]).astype(int)
        temp_N_cluster = np.zeros(N_S)
        for s in range(1, N_S + 1):
            temp_N_cluster = np.sum((cluster_assignment == s).astype(int),0)  # FIXME? this means the last index always defines the sum...
        min_N_cluster = min([temp_N_cluster])
    # initialise the stages of the two linear z-score models by fitting a
    # single linear z-score model to each of the two sets of individuals
    seq_init = np.zeros((N_S, stage_zscore.shape[1]))
    for s in range(N_S):
        temp_data = data[cluster_assignment.reshape(cluster_assignment.shape[0], ) == (s + 1), :]
        temp_seq_init = initialise_sequence_linearzscoremodel(0,
                                                              stage_zscore,
                                                              stage_biomarker_index)
        seq_init[s, :], _, _, _, _, _ = perform_em_mixturelinearzscoremodels(0,
                                                                             temp_data,
                                                                             min_biomarker_zscore,
                                                                             max_biomarker_zscore,
                                                                             std_biomarker_zscore,
                                                                             stage_zscore,
                                                                             stage_biomarker_index,
                                                                             temp_seq_init,
                                                                             [1],
                                                                             likelihood_flag,
                                                                             covar)
    return seq_init

def find_ml_mixture2linearzscoremodels(data,
                                       min_biomarker_zscore,
                                       max_biomarker_zscore,
                                       std_biomarker_zscore,
                                       stage_zscore,
                                       stage_biomarker_index,
                                       N_startpoints,
                                       likelihood_flag, 
                                       num_cores,
                                       covar ):
    '''
    Fit a mixture of two linear z-score models
    
    Inputs: 
    ======

     data: !important! needs to be (positive) z-scores! 
       dim: number of subjects x number of biomarkers

     min_biomarker_zscore: a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers

     max_biomarker_zscore: a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers

     std_biomarker_zscore: the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers

     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker

     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages

     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages

     N_startpoints: the number of start points for the fitting
     likelihood_flag - whether to use an exact method of inference - when set
     to 'Exact', the exact method is used, the approximate method is used for
     all other settings
    
    Outputs:
    =======

     ml_sequence - the ordering of the stages for each subtype

     ml_f - the most probable proportion of individuals belonging to each
     subtype

     ml_likelihood - the likelihood of the most probable SuStaIn model
     previous outputs _mat - same as before but for each start point

    '''
    N_S = 2

    ml_sequence_mat = np.zeros((N_S, stage_zscore.shape[1], N_startpoints))
    ml_f_mat = np.zeros((N_S, N_startpoints))
    ml_likelihood_mat = np.zeros((N_startpoints, 1))    

    pool = multiprocessing.Pool(num_cores)
    # instantiate function as class to pass to pool.map
    # first calculate array of sequences - do this first or same random number will be used simultaneously on each processor
    copier = partial(split_clusters,
                     data=data,
                     min_biomarker_zscore=min_biomarker_zscore,
                     max_biomarker_zscore=max_biomarker_zscore,
                     std_biomarker_zscore=std_biomarker_zscore,
                     stage_zscore=stage_zscore,
                     stage_biomarker_index=stage_biomarker_index,
                     likelihood_flag=likelihood_flag,
                     N_S=N_S,
                     covar = covar )
    # will return shape (N_startpoints, 1)
    seq_mat = np.array(pool.map(copier, range(N_startpoints)))
    # now optimise
    copier = partial(perform_em_mixturelinearzscoremodels,
                     data=data,
                     min_biomarker_zscore=min_biomarker_zscore,
                     max_biomarker_zscore=max_biomarker_zscore,
                     std_biomarker_zscore=std_biomarker_zscore,
                     stage_zscore=stage_zscore,
                     stage_biomarker_index=stage_biomarker_index,
                     current_sequence=seq_mat,
                     current_f=np.array([1.] * N_S) / float(N_S),
                     likelihood_flag=likelihood_flag,
                     covar)
    # will return shape (N_startpoints, 6)
    par_mat = np.array(pool.map(copier, range(N_startpoints)))
    # distribute to local matrices
    for i in range(N_startpoints):
        ml_sequence_mat[:, :, i] = par_mat[i, 0]
        ml_f_mat[:, i] = par_mat[i, 1]
        ml_likelihood_mat[i] = par_mat[i, 2]
    ix = np.where(ml_likelihood_mat == max(ml_likelihood_mat))
    ix = ix[0]
    ml_sequence = ml_sequence_mat[:, :, ix]
    ml_f = ml_f_mat[:, ix]
    ml_likelihood = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

def find_ml_mixturelinearzscoremodels(data,
                                      min_biomarker_zscore,
                                      max_biomarker_zscore,
                                      std_biomarker_zscore,
                                      stage_zscore,
                                      stage_biomarker_index,
                                      seq_init,
                                      f_init,
                                      N_startpoints,
                                      likelihood_flag,
                                      num_cores):
    '''
     Fit a mixture of linear z-score models
    
    Inputs: 
    ======

     data: !important! needs to be (positive) z-scores! 
       dim: number of subjects x number of biomarkers

     min_biomarker_zscore: a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers

     max_biomarker_zscore: a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers

     std_biomarker_zscore: the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers

     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker

     stage_zscore: the different z-scores of the model
       dim: 1 x number of z-score stages

     stage_biomarker_index: the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages

     seq_init: intial ordering of the stages for each subtype

     f_init: initial proprtion of individuals belonging to each subtype

     N_startpoints - the number of start points for the fitting

     likelihood_flag - whether to use an exact method of inference - when set
     to 'Exact', the exact method is used, the approximate method is used for
     all other settings
    
    Outputs:
    =======

     ml_sequence - the ordering of the stages for each subtype for the next
     SuStaIn model in the hierarchy

     ml_f - the most probable proportion of individuals belonging to each
     subtype for the next SuStaIn model in the hierarchy

     ml_likelihood - the likelihood of the most probable SuStaIn model for the
     next SuStaIn model in the hierarchy

     previous outputs _mat - same as before but for each start point

    '''

    N_S = seq_init.shape[0]
    terminate = 0
    startpoint = 0

    ml_sequence_mat = np.zeros((N_S, stage_zscore.shape[1], N_startpoints))
    ml_f_mat = np.zeros((N_S, N_startpoints))
    ml_likelihood_mat = np.zeros((N_startpoints, 1))

    pool = multiprocessing.Pool(num_cores)
    # instantiate function as class to pass to pool.map
    seq_mat = []
    for i in range(N_startpoints):
        seq_mat.append(seq_init)
    seq_mat = np.array(seq_mat)
    f_mat = []
    for i in range(N_startpoints):
        f_mat.append(f_init)
    f_mat = np.array(f_mat)

    copier = partial(perform_em_mixturelinearzscoremodels,
                     data=data,
                     min_biomarker_zscore=min_biomarker_zscore,
                     max_biomarker_zscore=max_biomarker_zscore,
                     std_biomarker_zscore=std_biomarker_zscore,
                     stage_zscore=stage_zscore,
                     stage_biomarker_index=stage_biomarker_index,
                     current_sequence=seq_mat,
                     current_f=f_mat,
                     likelihood_flag=likelihood_flag,
                     covar)
    # will return shape (N_startpoints, 6)
    par_mat = np.array(pool.map(copier, range(N_startpoints)))
    # distribute to local matrices
    for i in range(N_startpoints):
        ml_sequence_mat[:, :, i] = par_mat[i, 0]
        ml_f_mat[:, i] = par_mat[i, 1]
        ml_likelihood_mat[i] = par_mat[i, 2]        
    ix = np.where(ml_likelihood_mat == max(ml_likelihood_mat))
    ix = ix[0]
    ml_sequence = ml_sequence_mat[:, :, ix]
    ml_f = ml_f_mat[:, ix]
    ml_likelihood = ml_likelihood_mat[ix]

    return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

def estimate_uncertainty_sustain_model(data,
                                       min_biomarker_zscore,
                                       max_biomarker_zscore,
                                       std_biomarker_zscore,
                                       stage_zscore,
                                       stage_biomarker_index,
                                       seq_init,
                                       f_init,
                                       N_iterations_MCMC,
                                       likelihood_flag,
                                       n_mcmc_opt_its):
    '''
     Estimate the uncertainty in the subtype progression patterns and
     proportion of individuals belonging to the SuStaIn model
    
    Inputs:
    ======

     data: !important! needs to be (positive) z-scores! 
       dim: number of subjects x number of biomarkers

     min_biomarker_zscore: a minimum z-score for each biomarker (usually zero
     for all markers)
       dim: 1 x number of biomarkers

     max_biomarker_zscore: a maximum z-score for each biomarker - reached at
     the final stage of the linear z-score model
       dim: 1 x number of biomarkers

     std_biomarker_zscore: the standard devation of each biomarker z-score
     (should be 1 for all markers)
       dim: 1 x number of biomarkers

     stage_zscore and stage_biomarker_index give the different z-score stages
     for the linear z-score model, i.e. the index of the different z-scores
     for each biomarker
     stage_zscore - the different z-scores of the model
       dim: 1 x number of z-score stages
     stage_biomarker_index - the index of the biomarker that the corresponding
     entry of stage_zscore is referring to - !important! ensure biomarkers are
     indexed s.t. they correspond to columns 1 to number of biomarkers in your
     data
       dim: 1 x number of z-score stages
     seq_init - the ordering of the stages for each subtype to initialise the
     MCMC estimation from
       dim: number of subtypes x number of z-score stages
     f_init - the proportion of individuals belonging to each subtype to
     intialise the MCMC esimation from
       dim: number of subtypes x 1
     N_iterations_MCMC - the number of MCMC samples to take
     likelihood_flag - whether to use an exact method of inference - when set
     to 'Exact', the exact method is used, the approximate method is used for
     all other settings
    
    Outputs:
    =======

     ml_sequence: the most probable ordering of the stages for each subtype
     found across MCMC samples

     ml_f: the most probable proportion of individuals belonging to each
     subtype found across MCMC samples

     ml_likelihood: the likelihood of the most probable SuStaIn model found
     across MCMC samples

     samples_sequence: samples of the ordering of the stages for each subtype
     obtained from MCMC sampling

     samples_f: samples of the proportion of individuals belonging to each
     subtype obtained from MCMC sampling

     samples_likeilhood: samples of the likelihood of each SuStaIn model
     sampled by the MCMC sampling
    
     Perform a few initial passes where the perturbation sizes of the MCMC
     unertainty estimation are tuned

    '''

    seq_sigma_opt,f_sigma_opt = optimise_mcmc_settings_mixturelinearzscoremodels(data,
                                                                                 min_biomarker_zscore,
                                                                                 max_biomarker_zscore,
                                                                                 std_biomarker_zscore,
                                                                                 stage_zscore,
                                                                                 stage_biomarker_index,
                                                                                 seq_init,
                                                                                 f_init,
                                                                                 likelihood_flag,
                                                                                 n_mcmc_opt_its)
    # Run the full MCMC algorithm to estimate the uncertainty
    ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood = perform_mcmc_mixturelinearzscoremodels(data,
                                                                                                                               min_biomarker_zscore,
                                                                                                                               max_biomarker_zscore,
                                                                                                                               std_biomarker_zscore,
                                                                                                                               stage_zscore,
                                                                                                                               stage_biomarker_index,
                                                                                                                               seq_init,
                                                                                                                               f_init,
                                                                                                                               N_iterations_MCMC,
                                                                                                                               seq_sigma_opt,
                                                                                                                               f_sigma_opt,
                                                                                                                               likelihood_flag)
    
    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

def optimise_mcmc_settings_mixturelinearzscoremodels(data,
                                                     min_biomarker_zscore,
                                                     max_biomarker_zscore,
                                                     std_biomarker_zscore,
                                                     stage_zscore,
                                                     stage_biomarker_index,
                                                     seq_init,
                                                     f_init,
                                                     likelihood_flag,
                                                     n_iterations_MCMC_optimisation):
    # Optimise the perturbation size for the MCMC algorithm
    #    n_iterations_MCMC_optimisation = int(1e4)  # FIXME: set externally

    n_passes_optimisation = 3

    seq_sigma_currentpass = 1
    f_sigma_currentpass = 0.01  # magic number

    N_S = seq_init.shape[0]

    for i in range(n_passes_optimisation):
        _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = perform_mcmc_mixturelinearzscoremodels(data,
                                                                                                                 min_biomarker_zscore,
                                                                                                                 max_biomarker_zscore,
                                                                                                                 std_biomarker_zscore,
                                                                                                                 stage_zscore,
                                                                                                                 stage_biomarker_index,
                                                                                                                 seq_init,
                                                                                                                 f_init,
                                                                                                                 n_iterations_MCMC_optimisation,
                                                                                                                 seq_sigma_currentpass,
                                                                                                                 f_sigma_currentpass,
                                                                                                                 likelihood_flag)
        samples_position_currentpass = np.zeros(samples_sequence_currentpass.shape)
        for s in range(N_S):
            for sample in range(n_iterations_MCMC_optimisation):
                temp_seq = samples_sequence_currentpass[s, :, sample]
                temp_inv = np.array([0] * samples_sequence_currentpass.shape[1])
                temp_inv[temp_seq.astype(int)] = np.arange(samples_sequence_currentpass.shape[1])
                samples_position_currentpass[s, :, sample] = temp_inv
        seq_sigma_currentpass = np.std(samples_position_currentpass, axis=2,
                                       ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default
        seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01  # magic number
        f_sigma_currentpass = np.std(samples_f_currentpass, axis=1,
                                     ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default

    seq_sigma_opt = seq_sigma_currentpass
    f_sigma_opt = f_sigma_currentpass

    return seq_sigma_opt, f_sigma_opt

def perform_mcmc_mixturelinearzscoremodels(data,
                                           min_biomarker_zscore,
                                           max_biomarker_zscore,
                                           std_biomarker_zscore,
                                           stage_zscore,
                                           stage_biomarker_index,
                                           seq_init,
                                           f_init,
                                           n_iterations,
                                           seq_sigma,
                                           f_sigma,
                                           likelihood_flag):
    '''
    Inputs:
    ========

    data:

    min_biomarker_zscore:

    max_biomarker_zscore:

    std_biomarker_zscore:

    stage_zscore:

    stage_biomarker_index:

    seq_init:

    f_init:

    n_iterations:

    seq_sigma:

    f_sigma:

    likelihood_flag:


    Outputs:
    ========

    ml_sequence:

    ml_f:

    ml_likelihood: the likelihood of the most probable SuStaIn model found
     across MCMC samples

    samples_sequence: posterior samples of the ordering of stages for each subtype
     obtained from the MCMC

    samples_f:  

    samples_likelihood:

    '''    
    # Take MCMC samples of the uncertainty in the SuStaIn model parameters
    N = stage_zscore.shape[1]
    N_S = seq_init.shape[0]

    if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
        f_sigma = np.array([f_sigma])

    samples_sequence = np.zeros((N_S, N, n_iterations))
    samples_f = np.zeros((N_S, n_iterations))
    samples_likelihood = np.zeros((n_iterations, 1))
    samples_sequence[:, :, 0] = seq_init  # don't need to copy as we don't write to 0 index
    samples_f[:, 0] = f_init

    for i in range(n_iterations):
        if i % (n_iterations / 10) == 0:
            print('Iteration', i, 'of', n_iterations, ',', int(float(i) / float(n_iterations) * 100.), '% complete')
        if i > 0:
            seq_order = np.random.permutation(N_S)  # this function returns different random numbers to Matlab
            for s in seq_order:
                move_event_from = int(np.ceil(N * np.random.rand())) - 1
                current_sequence = samples_sequence[s, :, i - 1]
                current_location = np.array([0] * N)
                current_location[current_sequence.astype(int)] = np.arange(N)
                selected_event = int(current_sequence[move_event_from])
                this_stage_zscore = stage_zscore[0, selected_event]
                selected_biomarker = stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker = stage_zscore[stage_biomarker_index == selected_biomarker]
                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter = possible_zscores_biomarker < this_stage_zscore
                max_filter = possible_zscores_biomarker > this_stage_zscore
                events = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event = events[((stage_zscore[0] == min_zscore_bound).astype(int) + (
                                stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event = events[((stage_zscore[0] == max_zscore_bound).astype(int) + (
                                stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N
                # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions = np.array([0])
                else:
                    possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                distance = possible_positions - move_event_from

                if isinstance(seq_sigma, int):  # FIXME: change to float
                    this_seq_sigma = seq_sigma
                else:
                    this_seq_sigma = seq_sigma[s, selected_event]
                # use own normal PDF because stats.norm is slow
                weight = calc_coeff(this_seq_sigma) * calc_exp(distance, 0., this_seq_sigma)
                weight /= np.sum(weight)
                index = np.random.choice(range(len(possible_positions)), 1, replace=True,
                                         p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                move_event_to = possible_positions[index]

                current_sequence = np.delete(current_sequence, move_event_from, 0)
                new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event],
                                               current_sequence[np.arange(move_event_to, N - 1)]])
                samples_sequence[s, :, i] = new_sequence

            new_f = samples_f[:, i - 1] + f_sigma * np.random.randn()
            new_f = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
            samples_f[:, i] = new_f
        S = samples_sequence[:, :, i]
        f = samples_f[:, i]
        likelihood_sample, _, _, _, _ = calculate_likelihood_mixturelinearzscoremodels(data,
                                                                                       min_biomarker_zscore,
                                                                                       max_biomarker_zscore,
                                                                                       std_biomarker_zscore,
                                                                                       stage_zscore,
                                                                                       stage_biomarker_index,
                                                                                       S,
                                                                                       f,
                                                                                       likelihood_flag)
        samples_likelihood[i] = likelihood_sample
        if i > 0:
            ratio = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
            if ratio < np.random.rand():
                samples_likelihood[i] = samples_likelihood[i - 1]
                samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
                samples_f[:, i] = samples_f[:, i - 1]

    perm_index = np.where(samples_likelihood == max(samples_likelihood))
    perm_index = perm_index[0]
    ml_likelihood = max(samples_likelihood)
    ml_sequence = samples_sequence[:, :, perm_index]
    ml_f = samples_f[:, perm_index]
    return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

def plot_sustain_model(samples_sequence,
                       samples_f,
                       biomarker_labels,
                       stage_zscore,
                       stage_biomarker_index,
                       output_folder,
                       dataset_name,
                       subtype,
                       samples_likelihood,
                       cval=False):
    colour_mat = np.array([[0,0,1],[1,0,1],[1,0,0]])
    temp_mean_f = np.mean(samples_f, 1)
    vals = np.sort(temp_mean_f)[::-1]
    vals = np.array([np.round(x * 100.) for x in vals]) / 100.
    ix = np.argsort(temp_mean_f)[::-1]
    N_S = samples_sequence.shape[0]
    N_bio = len(biomarker_labels)
    if N_S > 1:
        fig, ax = plt.subplots(1, N_S)
    else:
        fig, ax = plt.subplots()
    for i in range(N_S):
        this_samples_sequence = np.squeeze(samples_sequence[ix[i], :, :]).T
        markers = np.unique(stage_biomarker_index)
        N = this_samples_sequence.shape[1]
        confus_matrix = np.zeros((N, N))
        for j in range(N):
            confus_matrix[j, :] = sum(this_samples_sequence == j)
        confus_matrix /= float(max(this_samples_sequence.shape))
        zvalues = np.unique(stage_zscore)
        N_z                     = len(zvalues)
        confus_matrix_z = np.zeros((N_bio, N, N_z))
        for z in range(N_z):
            confus_matrix_z[stage_biomarker_index[stage_zscore == zvalues[z]], :, z] = confus_matrix[
                                                                                       (stage_zscore == zvalues[z])[0],
                                                                                       :]
        confus_matrix_c = np.ones((N_bio, N, 3))
        for z in range(N_z):
            this_confus_matrix = confus_matrix_z[:, :, z]
            this_colour = colour_mat[z, :]
            alter_level = this_colour == 0
            this_colour_matrix = np.zeros((N_bio, N, 3))
            this_colour_matrix[:, :, alter_level] = np.tile(this_confus_matrix[markers, :].reshape(N_bio, N, 1),
                                                            (1, 1, sum(alter_level)))
            confus_matrix_c = confus_matrix_c - this_colour_matrix
        # must be a smarter way of doing this, but subplots(1,1) doesn't produce an array...
        if N_S > 1:
            ax[i].imshow(confus_matrix_c, interpolation='nearest', cmap=plt.cm.Blues)
            ax[i].set_xticks(np.arange(N))
            ax[i].set_xticklabels(range(1, N+1), rotation=45) #, fontsize=15)
            ax[i].set_yticks(np.arange(N_bio))
            ax[i].set_yticklabels(np.array(biomarker_labels, dtype='object'), rotation=30, ha='right', rotation_mode='anchor') #, fontsize=15)
            for tick in ax[i].yaxis.get_major_ticks():
                tick.label.set_color('black')
            ax[i].set_ylabel('Biomarker name') #, fontsize=20)
            ax[i].set_xlabel('Event position') #, fontsize=20)
            ax[i].set_title('Group ' + str(i) + ' f=' + str(vals[i]))
        else:
            ax.imshow(confus_matrix_c, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_xticks(np.arange(N))
            ax.set_xticklabels(range(1, N+1), rotation=45) #, fontsize=15)
            ax.set_yticks(np.arange(N_bio))
            ax.set_yticklabels(np.array(biomarker_labels, dtype='object'), rotation=30, ha='right', rotation_mode='anchor') #, fontsize=15)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_color('black')
            ax.set_ylabel('Biomarker name') #, fontsize=20)
            ax.set_xlabel('Event position') #, fontsize=20)
            ax.set_title('Group ' + str(i) + ' f=' + str(vals[i]))
    plt.tight_layout()
    if cval:
        fig.suptitle('Cross validation')

    # write results
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # with open(output_folder+'/'+dataset_name+'_subtype'+str(subtype)+'.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(samples_sequence)
    #     writer.writerows(samples_f)
    #     writer.writerows(biomarker_labels.reshape(1,len(biomarker_labels)))
    #     writer.writerows(stage_zscore)
    #     writer.writerows(stage_biomarker_index)
    #     writer.writerow([N_z])
    #     writer.writerow(samples_likelihood)
    return fig, ax

def cross_validate_sustain_model(fold,
                                 data,
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
                                 target
                                 ):
    '''
    # Cross-validate the SuStaIn model by running the SuStaIn algorithm (E-M
    # and MCMC) on a training dataset and evaluating the model likelihood on a test
    # dataset. 'data_fold' should specify the membership of each data point to a
    # test fold. Use a specific index of variable 'select_fold' to just run for a
    # single fold (allows the cross-validation to be run in parallel), or leave
    # the variable 'select_fold' empty to iterate across folds sequentially.
    '''

    #    for fold in range(Nfolds):
    print(fold)
    # print('Cross-validating fold',fold,'of',Nfolds,'with index',test_idxs[fold])
    data_train = data[np.array([x for x in range(data.shape[0]) if x not in test_idxs[fold]])]
    data_test = data[test_idxs[fold]]
    ml_sequence_prev_EM = []
    ml_f_prev_EM = []
    samples_sequence_cval = []
    samples_f_cval = []
    for s in range(N_S_max):
        #expectation maximisation
        ml_sequence_EM,ml_f_EM,ml_likelihood_EM,ml_sequence_mat_EM,ml_f_mat_EM,ml_likelihood_mat_EM = estimate_ml_sustain_model_nplus1_clusters(data_train,
                                                                                                                                                min_biomarker_zscore,
                                                                                                                                                max_biomarker_zscore,
                                                                                                                                                std_biomarker_zscore,
                                                                                                                                                stage_zscore,
                                                                                                                                                stage_biomarker_index,
                                                                                                                                                ml_sequence_prev_EM,
                                                                                                                                                ml_f_prev_EM,
                                                                                                                                                N_startpoints,
                                                                                                                                                likelihood_flag, 
                                                                                                                                                num_cores)
        with open(output_folder+'/'+dataset_name+'_EM_'+str(s)+'_Seq_Fold'+str(fold)+'.csv', 'w') as f:
            print("\n"+ "Writing results to disk" + "\n")
            writer = csv.writer(f)
            writer.writerows(ml_sequence_EM)
            writer.writerow([ml_likelihood_EM])
            writer.writerows(ml_sequence_mat_EM)
            writer.writerow([ml_f_mat_EM])
            writer.writerow([ml_likelihood_mat_EM])
            writer.writerow([ml_f_EM])
        seq_init = ml_sequence_EM
        f_init = ml_f_EM
        #MCMC
        ml_sequence,ml_f,ml_likelihood,samples_sequence,samples_f,samples_likelihood = estimate_uncertainty_sustain_model(data_train,
                                                                                                                          min_biomarker_zscore,
                                                                                                                          max_biomarker_zscore,
                                                                                                                          std_biomarker_zscore,
                                                                                                                          stage_zscore,
                                                                                                                          stage_biomarker_index,
                                                                                                                          seq_init,
                                                                                                                          f_init,
                                                                                                                          N_iterations_MCMC,
                                                                                                                          likelihood_flag,
                                                                                                                          n_mcmc_opt_its)
        with open(output_folder+'/'+dataset_name+'_MCMC_'+str(s)+'_Seq_Fold'+str(fold)+'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(ml_sequence)
            writer.writerows(ml_f)
            writer.writerow([ml_likelihood])
            writer.writerows(samples_sequence)
            writer.writerows(samples_f)
            writer.writerows(samples_likelihood)

        samples_likelihood_subj_test = evaluate_likelihood_setofsamples_mixturelinearzscoremodels(data_test,
                                                                                                  samples_sequence,
                                                                                                  samples_f,
                                                                                                  min_biomarker_zscore,
                                                                                                  max_biomarker_zscore,
                                                                                                  std_biomarker_zscore,
                                                                                                  stage_zscore,
                                                                                                  stage_biomarker_index,
                                                                                                  likelihood_flag)
        with open(output_folder + '/' + dataset_name + '_OutOfSampleLikelihood_' + str(s) + '_Seq_Fold' + str(fold) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(samples_likelihood_subj_test)

        ml_sequence_prev_EM = ml_sequence_EM
        ml_f_prev_EM = ml_f_EM
        samples_sequence_cval += list(samples_sequence)
        samples_f_cval += list(samples_f)
    
    return None

def evaluate_likelihood_setofsamples_mixturelinearzscoremodels(data,
                                                               samples_sequence,
                                                               samples_f,
                                                               min_biomarker_zscore,
                                                               max_biomarker_zscore,
                                                               std_biomarker_zscore,
                                                               stage_zscore,
                                                               stage_biomarker_index,
                                                               likelihood_flag):
    '''

    Function to 

    '''

    # Take MCMC samples of the uncertainty in the SuStaIn model parameters
    M = data.shape[0]
    n_iterations = samples_sequence.shape[2]
    samples_likelihood_subj = np.zeros((M, n_iterations))
    for i in range(n_iterations):
        #        if i%(n_iterations/10)==0:
        #            print('Iteration',i,'of',n_iterations,',',float(i)/float(n_iterations)*100.,'% complete')
        S = samples_sequence[:,:,i]
        f = samples_f[:,i]
        _,likelihood_sample_subj,_,_,_ = calculate_likelihood_mixturelinearzscoremodels(data,
                                                                                        min_biomarker_zscore,max_biomarker_zscore,
                                                                                        std_biomarker_zscore,
                                                                                        stage_zscore,
                                                                                        stage_biomarker_index,
                                                                                        S,
                                                                                        f,
                                                                                        likelihood_flag)
        samples_likelihood_subj[:,i] = likelihood_sample_subj
        
    return samples_likelihood_subj


def subtype_and_stage_individuals_selectsubtypethenstage(data,
                                                         samples_sequence,
                                                         samples_f,
                                                         min_biomarker_zscore,
                                                         max_biomarker_zscore,
                                                         std_biomarker_zscore,
                                                         stage_zscore,
                                                         stage_biomarker_index,
                                                         likelihood_flag,
                                                         N_samples,
                                                         N_c,
                                                         output_folder,
                                                         dataset_name):
    M = data.shape[0]
    n_iterations_MCMC = samples_sequence.shape[2]
    select_samples = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples))
    N_S = samples_sequence.shape[0]
    temp_mean_f = np.mean(samples_f, axis=1)
    ix = np.argsort(temp_mean_f)[::-1]

    prob_subtype_stage = np.zeros((data.shape[0], stage_zscore.shape[1] + 1, N_S))
    prob_subtype = np.zeros((data.shape[0], N_S))
    prob_stage = np.zeros((data.shape[0], stage_zscore.shape[1] + 1))
    for i in range(N_samples):
        sample = int(select_samples[i])
        #        if i%(N_samples/10)==0:
        #            print('Staging: iteration',i,'of',N_samples,',',float(i)/N_samples*100.,'complete\n')
        this_S = samples_sequence[ix, :, sample]
        this_f = samples_f[ix, sample]

        _, _, total_prob_stage, total_prob_subtype, total_prob_subtype_stage = calculate_likelihood_mixturelinearzscoremodels(
            data,
            min_biomarker_zscore,
            max_biomarker_zscore,
            std_biomarker_zscore,
            stage_zscore,
            stage_biomarker_index,
            this_S,
            this_f,
            likelihood_flag)
        total_prob_subtype = total_prob_subtype.reshape(len(total_prob_subtype), N_S)
        total_prob_subtype_norm = total_prob_subtype / np.tile(
            np.sum(total_prob_subtype, 1).reshape(len(total_prob_subtype), 1), (1, N_S))
        total_prob_stage_norm = total_prob_stage / np.tile(
            np.sum(total_prob_stage, 1).reshape(len(total_prob_subtype), 1), (1, stage_zscore.shape[1] + 1))
        total_prob_subtype_stage_norm = total_prob_subtype_stage / np.tile(
            np.sum(np.sum(total_prob_subtype_stage, 1), 1).reshape(M, 1, 1), (1, stage_zscore.shape[1] + 1, N_S))

        prob_subtype_stage = (i / (i + 1.) * prob_subtype_stage) + (1. / (i + 1.) * total_prob_subtype_stage_norm)
        prob_subtype = (i / (i + 1.) * prob_subtype) + (1. / (i + 1.) * total_prob_subtype_norm)
        prob_stage = (i / (i + 1.) * prob_stage) + (1. / (i + 1.) * total_prob_stage_norm)

    ml_subtype = np.nan * np.ones((data.shape[0], 1))
    ml_stage = np.nan * np.ones((data.shape[0], 1))
    for i in range(data.shape[0]):
        this_prob_subtype = np.squeeze(prob_subtype[i, :])
        if (np.sum(np.isnan(this_prob_subtype)) == 0):
            this_subtype = np.where(this_prob_subtype == np.max(this_prob_subtype))
            ml_subtype[i] = this_subtype
        this_prob_stage = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])
        if (np.sum(np.isnan(this_prob_stage)) == 0):
            this_stage = np.where(this_prob_stage == np.max(this_prob_stage))
            ml_stage[i] = this_stage[0][0]

    return ml_subtype, ml_stage

# own functions for optimisation
def calc_coeff(sig):
    return 1. / np.sqrt(np.pi * 2.0) * sig

def calc_exp(x, mu, sig):
    x = (x - mu) / sig
    return np.exp(-.5 * x * x)

def normPdf(x, mu, sig):
    return calc_coeff * calc_exp

def linspace_local(a, b, N):
    return a + (b - a) / (N - 1.) * np.arange(N)

def linspace_local2(a, b, N, arange_N):
    return a + (b - a) / (N - 1.) * arange_N
