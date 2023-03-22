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
from abc import ABC, abstractmethod

from tqdm.auto import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import pickle
import csv
import os
import multiprocessing
from functools import partial, partialmethod

import time
import pathos

#*******************************************
#The data structure class for AbstractSustain. It has no data itself - the implementations of AbstractSustain need to define their own implementations of this class.
class AbstractSustainData(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getNumSamples(self):
        pass

    @abstractmethod
    def getNumBiomarkers(self):
        pass

    @abstractmethod
    def getNumStages(self):
        pass

    @abstractmethod
    def reindex(self, index):
        pass

#*******************************************
class AbstractSustain(ABC):

    def __init__(self,
                 sustainData,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the abstract class
        # Parameters:
        #   sustainData                 - an instance of an AbstractSustainData implementation
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        assert(isinstance(sustainData, AbstractSustainData))

        self.__sustainData              = sustainData

        self.N_startpoints              = N_startpoints
        self.N_S_max                    = N_S_max
        self.N_iterations_MCMC          = N_iterations_MCMC

        self.num_cores                  = multiprocessing.cpu_count()

        self.output_folder              = output_folder
        self.dataset_name               = dataset_name

        if isinstance(seed, int):
            self.seed = seed
        elif isinstance(seed, float):
            self.seed = int(seed)
        elif seed is None:
            # Select random seed if none given
            self.seed = np.random.default_rng().integers((2**32)-1)

        # Create global rng to create process-specific rngs
        self.global_rng = np.random.default_rng(self.seed)

        self.use_parallel_startpoints   = use_parallel_startpoints

        if self.use_parallel_startpoints:
            np_version                  = float(np.__version__.split('.')[0] + '.' + np.__version__.split('.')[1])
            assert np_version >= 1.18, "numpy version must be >= 1.18 for parallelization to work properly."

            self.pool                   = pathos.multiprocessing.ProcessingPool() #pathos.multiprocessing.ParallelPool()
            self.pool.ncpus             = multiprocessing.cpu_count()
        else:
            self.pool                   = pathos.serial.SerialPool()

    #********************* PUBLIC METHODS
    def run_sustain_algorithm(self, plot=False, plot_format="png", **kwargs):
        # Externally called method to start the SuStaIn algorithm after initializing the SuStaIn class object properly

        ml_sequence_prev_EM                 = []
        ml_f_prev_EM                        = []

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)
        if plot:
            fig0, ax0                           = plt.subplots()
        for s in range(self.N_S_max):

            pickle_filename_s               = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(s) + '.pickle')
            pickle_filepath                 = Path(pickle_filename_s)
            if pickle_filepath.exists():
                print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s) + " subtype.")

                pickle_file                 = open(pickle_filename_s, 'rb')

                loaded_variables            = pickle.load(pickle_file)

                #self.stage_zscore           = loaded_variables["stage_zscore"]
                #self.stage_biomarker_index  = loaded_variables["stage_biomarker_index"]
                #self.N_S_max                = loaded_variables["N_S_max"]

                samples_likelihood          = loaded_variables["samples_likelihood"]
                samples_sequence            = loaded_variables["samples_sequence"]
                samples_f                   = loaded_variables["samples_f"]

                ml_sequence_EM              = loaded_variables["ml_sequence_EM"]
                ml_sequence_prev_EM         = loaded_variables["ml_sequence_prev_EM"]
                ml_f_EM                     = loaded_variables["ml_f_EM"]
                ml_f_prev_EM                = loaded_variables["ml_f_prev_EM"]

                pickle_file.close()
            else:
                print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s) + " subtype.")

                ml_sequence_EM,     \
                ml_f_EM,            \
                ml_likelihood_EM,   \
                ml_sequence_mat_EM, \
                ml_f_mat_EM,        \
                ml_likelihood_mat_EM        = self._estimate_ml_sustain_model_nplus1_clusters(self.__sustainData, ml_sequence_prev_EM, ml_f_prev_EM) #self.__estimate_ml_sustain_model_nplus1_clusters(self.__data, ml_sequence_prev_EM, ml_f_prev_EM)

                seq_init                    = ml_sequence_EM
                f_init                      = ml_f_EM

                ml_sequence,        \
                ml_f,               \
                ml_likelihood,      \
                samples_sequence,   \
                samples_f,          \
                samples_likelihood          = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init)           #self.__estimate_uncertainty_sustain_model(self.__data, seq_init, f_init)
                ml_sequence_prev_EM         = ml_sequence_EM
                ml_f_prev_EM                = ml_f_EM

            # max like subtype and stage / subject
            N_samples                       = 1000
            ml_subtype,             \
            prob_ml_subtype,        \
            ml_stage,               \
            prob_ml_stage,          \
            prob_subtype,           \
            prob_stage,             \
            prob_subtype_stage               = self.subtype_and_stage_individuals(self.__sustainData, samples_sequence, samples_f, N_samples)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
            if not pickle_filepath.exists():

                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                save_variables                          = {}
                save_variables["samples_sequence"]      = samples_sequence
                save_variables["samples_f"]             = samples_f
                save_variables["samples_likelihood"]    = samples_likelihood

                save_variables["ml_subtype"]            = ml_subtype
                save_variables["prob_ml_subtype"]       = prob_ml_subtype
                save_variables["ml_stage"]              = ml_stage
                save_variables["prob_ml_stage"]         = prob_ml_stage
                save_variables["prob_subtype"]          = prob_subtype
                save_variables["prob_stage"]            = prob_stage
                save_variables["prob_subtype_stage"]    = prob_subtype_stage

                save_variables["ml_sequence_EM"]        = ml_sequence_EM
                save_variables["ml_sequence_prev_EM"]   = ml_sequence_prev_EM
                save_variables["ml_f_EM"]               = ml_f_EM
                save_variables["ml_f_prev_EM"]          = ml_f_prev_EM

                pickle_file                 = open(pickle_filename_s, 'wb')
                pickle_output               = pickle.dump(save_variables, pickle_file)
                pickle_file.close()

            n_samples                       = self.__sustainData.getNumSamples() #self.__data.shape[0]

            #order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
            self._plot_subtype_order        = np.argsort(ml_f_EM)[::-1]
            #order of biomarkers in each subtypes' positional variance diagram
            self._plot_biomarker_order      = ml_sequence_EM[self._plot_subtype_order[0], :].astype(int)

            # plot results
            if plot:
                figs, ax = self._plot_sustain_model(
                    samples_sequence=samples_sequence,
                    samples_f=samples_f,
                    n_samples=n_samples,
                    biomarker_labels=self.biomarker_labels,
                    subtype_order=self._plot_subtype_order,
                    biomarker_order=self._plot_biomarker_order,
                    save_path=Path(self.output_folder) / f"{self.dataset_name}_subtype{s}_PVD.{plot_format}",
                    **kwargs
                )
                for fig in figs:
                    fig.show()

                ax0.plot(range(self.N_iterations_MCMC), samples_likelihood, label="Subtype " + str(s+1))

        # save and show this figure after all subtypes have been calculcated
        if plot:
            ax0.legend(loc='upper right')
            fig0.tight_layout()
            fig0.savefig(Path(self.output_folder) / f"MCMC_likelihoods.{plot_format}", bbox_inches='tight')
            fig0.show()

        return samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage


    def cross_validate_sustain_model(self, test_idxs, select_fold = [], plot=False):
        # Cross-validate the SuStaIn model by running the SuStaIn algorithm (E-M
        # and MCMC) on a training dataset and evaluating the model likelihood on a test
        # dataset.
        # Parameters:
        #   'test_idxs'     - list of test set indices for each fold
        #   'select_fold'   - allows user to just run for a single fold (allows the cross-validation to be run in parallel).
        #                     leave this variable empty to iterate across folds sequentially.

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        if select_fold != []:
            if np.isscalar(select_fold):
                select_fold                 = [select_fold]
        else:
            select_fold                     = np.arange(len(test_idxs)) #test_idxs
        Nfolds                              = len(select_fold)

        is_full                             = Nfolds == len(test_idxs)

        loglike_matrix                      = np.zeros((Nfolds, self.N_S_max))

        for fold in tqdm(select_fold, "Folds: ", Nfolds, position=0, leave=True):

            indx_test                       = test_idxs[fold]
            indx_train                      = np.array([x for x in range(self.__sustainData.getNumSamples()) if x not in indx_test])

            sustainData_train               = self.__sustainData.reindex(indx_train)
            sustainData_test                = self.__sustainData.reindex(indx_test)

            ml_sequence_prev_EM             = []
            ml_f_prev_EM                    = []

            for s in range(self.N_S_max):

                pickle_filename_fold_s      = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s) + '.pickle')
                pickle_filepath             = Path(pickle_filename_fold_s)

                if pickle_filepath.exists():

                    print("Loading " + pickle_filename_fold_s)

                    pickle_file             = open(pickle_filename_fold_s, 'rb')

                    loaded_variables        = pickle.load(pickle_file)

                    ml_sequence_EM          = loaded_variables["ml_sequence_EM"]
                    ml_sequence_prev_EM     = loaded_variables["ml_sequence_prev_EM"]
                    ml_f_EM                 = loaded_variables["ml_f_EM"]
                    ml_f_prev_EM            = loaded_variables["ml_f_prev_EM"]

                    samples_likelihood      = loaded_variables["samples_likelihood"]
                    samples_sequence        = loaded_variables["samples_sequence"]
                    samples_f               = loaded_variables["samples_f"]

                    mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                    pickle_file.close()

                    samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f)

                else:
                    ml_sequence_EM,         \
                    ml_f_EM,                \
                    ml_likelihood_EM,       \
                    ml_sequence_mat_EM,     \
                    ml_f_mat_EM,            \
                    ml_likelihood_mat_EM    = self._estimate_ml_sustain_model_nplus1_clusters(sustainData_train, ml_sequence_prev_EM, ml_f_prev_EM)

                    seq_init                    = ml_sequence_EM
                    f_init                      = ml_f_EM

                    ml_sequence,            \
                    ml_f,                   \
                    ml_likelihood,          \
                    samples_sequence,       \
                    samples_f,              \
                    samples_likelihood           = self._estimate_uncertainty_sustain_model(sustainData_train, seq_init, f_init)

                    samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, samples_sequence, samples_f)

                    mean_likelihood_subj_test    = np.mean(samples_likelihood_subj_test,axis=1)

                    ml_sequence_prev_EM         = ml_sequence_EM
                    ml_f_prev_EM                = ml_f_EM

                    save_variables                                      = {}
                    save_variables["ml_sequence_EM"]                    = ml_sequence_EM
                    save_variables["ml_sequence_prev_EM"]               = ml_sequence_prev_EM
                    save_variables["ml_f_EM"]                           = ml_f_EM
                    save_variables["ml_f_prev_EM"]                      = ml_f_prev_EM

                    save_variables["samples_sequence"]                  = samples_sequence
                    save_variables["samples_f"]                         = samples_f
                    save_variables["samples_likelihood"]                = samples_likelihood

                    save_variables["mean_likelihood_subj_test"]         = mean_likelihood_subj_test

                    pickle_file                     = open(pickle_filename_fold_s, 'wb')
                    pickle_output                   = pickle.dump(save_variables, pickle_file)
                    pickle_file.close()

                if is_full:
                    loglike_matrix[fold, s]         = np.mean(np.sum(np.log(samples_likelihood_subj_test + 1e-250),axis=0))

        if not is_full:
            print("Cannot calculate CVIC and loglike_matrix without all folds. Rerun cross_validate_sustain_model after all folds calculated.")
            return [], []

        print(f"Average test set log-likelihood for each subtype model: {np.mean(loglike_matrix, 0)}")

        if plot:
            import pandas as pd
            fig, ax = plt.subplots()

            df_loglike = pd.DataFrame(data = loglike_matrix, columns = ["Subtype " + str(i+1) for i in range(self.N_S_max)])
            df_loglike.boxplot(grid=False, ax=ax, fontsize=15)
            for i in range(self.N_S_max):
                y = df_loglike[["Subtype " + str(i+1)]]
                # Add some random "jitter" to the x-axis
                x = np.random.normal(1+i, 0.04, size=len(y))
                ax.plot(x, y.values, 'r.', alpha=0.2)
            fig.savefig(Path(self.output_folder) / 'Log_likelihoods_cv_folds.png')
            fig.show()

        CVIC                            = np.zeros(self.N_S_max)

        for s in range(self.N_S_max):
            for fold in range(Nfolds):
                pickle_filename_fold_s  = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s) + '.pickle')
                pickle_filepath         = Path(pickle_filename_fold_s)

                pickle_file             = open(pickle_filename_fold_s, 'rb')
                loaded_variables        = pickle.load(pickle_file)

                mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                pickle_file.close()
    
                if fold == 0:
                    mean_likelihood_subj_test_cval    = mean_likelihood_subj_test
                else:
                    mean_likelihood_subj_test_cval    = np.concatenate((mean_likelihood_subj_test_cval, mean_likelihood_subj_test), axis=0)

            CVIC[s]                     = -2*sum(np.log(mean_likelihood_subj_test_cval))

        print("CVIC for each subtype model: " + str(CVIC))

        return CVIC, loglike_matrix


    def combine_cross_validated_sequences(self, N_subtypes, N_folds, plot_format="png", **kwargs):
        # Combine MCMC sequences across cross-validation folds to get cross-validated positional variance diagrams,
        # so that you get more realistic estimates of variance within event positions within subtypes

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')

        #*********** load ML sequence for full model for N_subtypes
        pickle_filename_s                   = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(N_subtypes-1) + '.pickle')        
        pickle_filepath                     = Path(pickle_filename_s)

        assert pickle_filepath.exists(), "Failed to find pickle file for full model with " + str(N_subtypes) + " subtypes."

        pickle_file                         = open(pickle_filename_s, 'rb')

        loaded_variables_full               = pickle.load(pickle_file)

        ml_sequence_EM_full                 = loaded_variables_full["ml_sequence_EM"]
        ml_f_EM_full                        = loaded_variables_full["ml_f_EM"]

        #REMOVED SO THAT PLOT_SUBTYPE_ORDER WORKS THE SAME HERE AS IN run_sustain_algorithm
        #re-index so that subtypes are in descending order by fraction of subjects
        # index_EM_sort                       = np.argsort(ml_f_EM_full)[::-1]
        # ml_sequence_EM_full                 = ml_sequence_EM_full[index_EM_sort,:]
        # ml_f_EM_full                        = ml_f_EM_full[index_EM_sort]

        for i in range(N_folds):

            #load the MCMC sequences for this fold's model of N_subtypes
            pickle_filename_fold_s          = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(i) + '_subtype' + str(N_subtypes-1) + '.pickle')        
            pickle_filepath                 = Path(pickle_filename_fold_s)

            assert pickle_filepath.exists(), "Failed to find pickle file for fold " + str(i)

            pickle_file                     = open(pickle_filename_fold_s, 'rb')

            loaded_variables_i              = pickle.load(pickle_file)

            ml_sequence_EM_i                = loaded_variables_i["ml_sequence_EM"]
            ml_f_EM_i                       = loaded_variables_i["ml_f_EM"]

            samples_sequence_i              = loaded_variables_i["samples_sequence"]
            samples_f_i                     = loaded_variables_i["samples_f"]

            mean_likelihood_subj_test       = loaded_variables_i["mean_likelihood_subj_test"]

            pickle_file.close()

            # Really simple approach: choose order based on this fold's fraction of subjects per subtype
            # It doesn't work very well when the fractions of subjects are similar across subtypes
            #mean_f_i                        = np.mean(samples_f_i, 1)
            #iMax_vec                        = np.argsort(mean_f_i)[::-1]
            #iMax_vec                        = iMax_vec.astype(int)

            #This approach seems to work better:
            # 1. calculate the Kendall's tau correlation matrix,
            # 2. Flatten the matrix into a vector
            # 3. Sort the vector, then unravel the flattened indices back into matrix style (x, y) indices
            # 4. Find the order in which this fold's subtypes first appear in the sorted list
            corr_mat                        = np.zeros((N_subtypes, N_subtypes))
            for j in range(N_subtypes):
                for k in range(N_subtypes):
                    corr_mat[j,k]            = stats.kendalltau(ml_sequence_EM_full[j,:], ml_sequence_EM_i[k,:]).correlation
            set_full                        = []
            set_fold_i                      = []
            i_i, i_j                        = np.unravel_index(np.argsort(corr_mat.flatten())[::-1], (N_subtypes, N_subtypes))
            for k in range(len(i_i)):
                if not i_i[k] in set_full and not i_j[k] in set_fold_i:
                    set_full.append(i_i[k].astype(int))
                    set_fold_i.append(i_j[k].astype(int))
            index_set_full                  = np.argsort(set_full).astype(int) #np.argsort(set_full)[::-1].astype(int)
            iMax_vec                        = [set_fold_i[i] for i in index_set_full]

            assert(np.all(np.sort(iMax_vec)==np.arange(N_subtypes)))

            if i == 0:
                samples_sequence_cval       = samples_sequence_i[iMax_vec,:,:]
                samples_f_cval              = samples_f_i[iMax_vec, :]
            else:
                samples_sequence_cval       = np.concatenate((samples_sequence_cval,    samples_sequence_i[iMax_vec,:,:]),  axis=2)
                samples_f_cval              = np.concatenate((samples_f_cval,           samples_f_i[iMax_vec,:]),           axis=1)

        n_samples                           = self.__sustainData.getNumSamples()

        #ADDED HERE BECAUSE THIS MAY BE CALLED BY CALLED FOR A RANGE OF N_S_max, AS IN simrun.py
        # order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
        plot_subtype_order                  = np.argsort(ml_f_EM_full)[::-1]
        # order of biomarkers in each subtypes' positional variance diagram
        plot_biomarker_order                = ml_sequence_EM_full[plot_subtype_order[0], :].astype(int)

        figs, ax = self._plot_sustain_model(
            samples_sequence=samples_sequence_cval,
            samples_f=samples_f_cval,
            n_samples=n_samples,
            cval=True,
            biomarker_labels=self.biomarker_labels,
            subtype_order=plot_subtype_order,
            biomarker_order=plot_biomarker_order,
            **kwargs
        )
        # If saving is being done here
        if "save_path" not in kwargs:
            # Handle separated subtypes
            if len(figs) > 1:
                # Loop over each figure/subtype
                for num_subtype, fig in zip(range(N_subtypes), figs):
                    # Nice confusing filename
                    plot_fname = Path(
                        self.output_folder
                    ) / f"{self.dataset_name}_subtype{N_subtypes - 1}_subtype{num_subtype}-separated_PVD_{N_folds}fold_CV.{plot_format}"
                    # Save the figure
                    fig.savefig(plot_fname, bbox_inches='tight')
                    fig.show()
            # Otherwise default single plot
            else:
                fig = figs[0]
                # save and show this figure after all subtypes have been calculcated
                plot_fname = Path(
                    self.output_folder
                ) / f"{self.dataset_name}_subtype{N_subtypes - 1}_PVD_{N_folds}fold_CV.{plot_format}"
                # Save the figure
                fig.savefig(plot_fname, bbox_inches='tight')
                fig.show()

        #return samples_sequence_cval, samples_f_cval, kendalls_tau_mat, f_mat #samples_sequence_cval

    def subtype_and_stage_individuals(self, sustainData, samples_sequence, samples_f, N_samples):
        # Subtype and stage a set of subjects. Useful for subtyping/staging subjects that were not used to build the model

        nSamples                            = sustainData.getNumSamples()  #data_local.shape[0]
        nStages                             = sustainData.getNumStages()    #self.stage_zscore.shape[1]

        n_iterations_MCMC                   = samples_sequence.shape[2]
        select_samples                      = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples))
        N_S                                 = samples_sequence.shape[0]
        temp_mean_f                         = np.mean(samples_f, axis=1)
        ix                                  = np.argsort(temp_mean_f)[::-1]

        prob_subtype_stage                  = np.zeros((nSamples, nStages + 1, N_S))
        prob_subtype                        = np.zeros((nSamples, N_S))
        prob_stage                          = np.zeros((nSamples, nStages + 1))

        for i in range(N_samples):
            sample                          = int(select_samples[i])

            this_S                          = samples_sequence[ix, :, sample]
            this_f                          = samples_f[ix, sample]

            _,                  \
            _,                  \
            total_prob_stage,   \
            total_prob_subtype, \
            total_prob_subtype_stage        = self._calculate_likelihood(sustainData, this_S, this_f)

            total_prob_subtype              = total_prob_subtype.reshape(len(total_prob_subtype), N_S)
            total_prob_subtype_norm         = total_prob_subtype        / np.tile(np.sum(total_prob_subtype, 1).reshape(len(total_prob_subtype), 1),        (1, N_S))
            total_prob_stage_norm           = total_prob_stage          / np.tile(np.sum(total_prob_stage, 1).reshape(len(total_prob_stage), 1),          (1, nStages + 1)) #removed total_prob_subtype

            #total_prob_subtype_stage_norm   = total_prob_subtype_stage  / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1), 1).reshape(nSamples, 1, 1),   (1, nStages + 1, N_S))
            total_prob_subtype_stage_norm   = total_prob_subtype_stage / np.tile(np.sum(np.sum(total_prob_subtype_stage, 1, keepdims=True), 2).reshape(nSamples, 1, 1),(1, nStages + 1, N_S))

            prob_subtype_stage              = (i / (i + 1.) * prob_subtype_stage)   + (1. / (i + 1.) * total_prob_subtype_stage_norm)
            prob_subtype                    = (i / (i + 1.) * prob_subtype)         + (1. / (i + 1.) * total_prob_subtype_norm)
            prob_stage                      = (i / (i + 1.) * prob_stage)           + (1. / (i + 1.) * total_prob_stage_norm)

        ml_subtype                          = np.nan * np.ones((nSamples, 1))
        prob_ml_subtype                     = np.nan * np.ones((nSamples, 1))
        ml_stage                            = np.nan * np.ones((nSamples, 1))
        prob_ml_stage                       = np.nan * np.ones((nSamples, 1))

        for i in range(nSamples):
            this_prob_subtype               = np.squeeze(prob_subtype[i, :])
            # if not np.isnan(this_prob_subtype).any()
            if (np.sum(np.isnan(this_prob_subtype)) == 0):
                # this_subtype = this_prob_subtype.argmax(
                this_subtype                = np.where(this_prob_subtype == np.max(this_prob_subtype))

                try:
                    ml_subtype[i]           = this_subtype
                except:
                    ml_subtype[i]           = this_subtype[0][0]
                if this_prob_subtype.size == 1 and this_prob_subtype == 1:
                    prob_ml_subtype[i]      = 1
                else:
                    try:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype]
                    except:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype[0][0]]

            this_prob_stage                 = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])
            
            if (np.sum(np.isnan(this_prob_stage)) == 0):
                # this_stage = 
                this_stage                  = np.where(this_prob_stage == np.max(this_prob_stage))
                ml_stage[i]                 = this_stage[0][0]
                prob_ml_stage[i]            = this_prob_stage[this_stage[0][0]]
        # NOTE: The above loop can be replaced with some simpler numpy calls
        # May need to do some masking to avoid NaNs, or use `np.nanargmax` depending on preference
        # E.g. ml_subtype == prob_subtype.argmax(1)
        # E.g. ml_stage == prob_subtype_stage[np.arange(prob_subtype_stage.shape[0]), :, ml_subtype].argmax(1)
        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

    # ********************* PROTECTED METHODS
    def _estimate_ml_sustain_model_nplus1_clusters(self, sustainData, ml_sequence_prev, ml_f_prev):
        # Given the previous SuStaIn model, estimate the next model in the
        # hierarchy (i.e. number of subtypes goes from N to N+1)
        #
        #
        # OUTPUTS:
        # ml_sequence       - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f              - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood     - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S = len(ml_sequence_prev) + 1
        if N_S == 1:
            # If the number of subtypes is 1, fit a single linear z-score model
            print('Finding ML solution to 1 cluster problem')
            ml_sequence,        \
            ml_f,               \
            ml_likelihood,      \
            ml_sequence_mat,    \
            ml_f_mat,           \
            ml_likelihood_mat               = self._find_ml(sustainData)
            print('Overall ML likelihood is', ml_likelihood)

        else:
            # If the number of subtypes is greater than 1, go through each subtype
            # in turn and try splitting into two subtypes
            _, _, _, p_sequence, _          = self._calculate_likelihood(sustainData, ml_sequence_prev, ml_f_prev)

            ml_sequence_prev                = ml_sequence_prev.reshape(ml_sequence_prev.shape[0], ml_sequence_prev.shape[1])
            p_sequence                      = p_sequence.reshape(p_sequence.shape[0], N_S - 1)
            p_sequence_norm                 = p_sequence / np.tile(np.sum(p_sequence, 1).reshape(len(p_sequence), 1), (N_S - 1))

            # Assign individuals to a subtype (cluster) based on the previous model
            ml_cluster_subj                 = np.zeros((sustainData.getNumSamples(), 1))   #np.zeros((len(data_local), 1))
            for m in range(sustainData.getNumSamples()):                                   #range(len(data_local)):
                ix                          = np.argmax(p_sequence_norm[m, :]) + 1

                #TEMP: MATLAB comparison
                #ml_cluster_subj[m]          = ix*np.ceil(np.random.rand())
                ml_cluster_subj[m]          = ix  # FIXME: should check this always works, as it differs to the Matlab code, which treats ix as an array

            ml_likelihood                   = -np.inf
            for ix_cluster_split in range(N_S - 1):
                this_N_cluster              = sum(ml_cluster_subj == int(ix_cluster_split + 1))

                if this_N_cluster > 1:

                    # Take the data from the individuals belonging to a particular
                    # cluster and fit a two subtype model
                    print('Splitting cluster', ix_cluster_split + 1, 'of', N_S - 1)
                    ix_i                    = (ml_cluster_subj == int(ix_cluster_split + 1)).reshape(sustainData.getNumSamples(), )
                    sustainData_i           = sustainData.reindex(ix_i)

                    print(' + Resolving 2 cluster problem')
                    this_ml_sequence_split, _, _, _, _, _ = self._find_ml_split(sustainData_i)

                    # Use the two subtype model combined with the other subtypes to
                    # inititialise the fitting of the next SuStaIn model in the
                    # hierarchy
                    this_seq_init           = ml_sequence_prev.copy()  # have to copy or changes will be passed to ml_sequence_prev

                    #replace the previous sequence with the first (row index zero) new sequence
                    this_seq_init[ix_cluster_split] = (this_ml_sequence_split[0]).reshape(this_ml_sequence_split.shape[1])

                    #add the second new sequence (row index one) to the stack of sequences, 
                    #so that you now have N_S sequences instead of N_S-1
                    this_seq_init           = np.hstack((this_seq_init.T, this_ml_sequence_split[1])).T
                    
                    #initialize fraction of subjects in each subtype to be uniform
                    this_f_init             = np.array([1.] * N_S) / float(N_S)

                    print(' + Finding ML solution from hierarchical initialisation')
                    this_ml_sequence,       \
                    this_ml_f,              \
                    this_ml_likelihood,     \
                    this_ml_sequence_mat,   \
                    this_ml_f_mat,          \
                    this_ml_likelihood_mat  = self._find_ml_mixture(sustainData, this_seq_init, this_f_init)

                    # Choose the most probable SuStaIn model from the different
                    # possible SuStaIn models initialised by splitting each subtype
                    # in turn
                    # FIXME: these arrays have an unnecessary additional axis with size = N_startpoints - remove it further upstream
                    if this_ml_likelihood[0] > ml_likelihood:
                        ml_likelihood       = this_ml_likelihood[0]
                        ml_sequence         = this_ml_sequence[:, :, 0]
                        ml_f                = this_ml_f[:, 0]
                        ml_likelihood_mat   = this_ml_likelihood_mat[0]
                        ml_sequence_mat     = this_ml_sequence_mat[:, :, 0]
                        ml_f_mat            = this_ml_f_mat[:, 0]
                    print('- ML likelihood is', this_ml_likelihood[0])
                else:
                    print(f'Cluster {ix_cluster_split + 1} of {N_S - 1} too small for subdivision')
            print(f'Overall ML likelihood is', ml_likelihood)

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    #********************************************

    def _find_ml(self, sustainData):
        # Fit the maximum likelihood model
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model

        partial_iter                        = partial(self._find_ml_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((1, sustainData.getNumStages(), self.N_startpoints)) #np.zeros((1, self.stage_zscore.shape[1], self.N_startpoints))
        ml_f_mat                            = np.zeros((1, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros(self.N_startpoints)

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = np.argmax(ml_likelihood_mat)
        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_iteration(self, sustainData, seed_seq):
        #Convenience sub-function for above

        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        # randomly initialise the sequence of the linear z-score model
        seq_init                        = self._initialise_sequence(sustainData, rng)
        f_init                          = [1]

        this_ml_sequence,   \
        this_ml_f,          \
        this_ml_likelihood, \
        _,                  \
        _,                  \
        _                               = self._perform_em(sustainData, seq_init, f_init, rng)

        return this_ml_sequence, this_ml_f, this_ml_likelihood

    #********************************************

    def _find_ml_split(self, sustainData):
        # Fit a mixture of two models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model

        N_S                                 = 2

        partial_iter                        = partial(self._find_ml_split_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = [np.where(ml_likelihood_mat == max(ml_likelihood_mat))[0][0]] #ugly bit of code to get first index where likelihood is maximum

        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_split_iteration(self, sustainData, seed_seq):
        #Convenience sub-function for above

        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        N_S                                 = 2

        # randomly initialise individuals as belonging to one of the two subtypes (clusters)
        min_N_cluster                       = 0
        while min_N_cluster == 0:
            vals = rng.random(sustainData.getNumSamples())
            cluster_assignment = np.ceil(N_S * vals).astype(int)
            # Count cluster sizes
            # Guarantee 1s and 2s counts with minlength=3
            # Ignore 0s count with [1:]
            cluster_sizes = np.bincount(cluster_assignment, minlength=3)[1:]
            # Get the minimum cluster size
            min_N_cluster = cluster_sizes.min()

        # initialise the stages of the two models by fitting a single model to each of the two sets of individuals
        seq_init                            = np.zeros((N_S, sustainData.getNumStages()))
        for s in range(N_S):
            index_s                         = cluster_assignment.reshape(cluster_assignment.shape[0], ) == (s + 1)
            temp_sustainData                = sustainData.reindex(index_s)

            temp_seq_init                   = self._initialise_sequence(sustainData, rng)
            seq_init[s, :], _, _, _, _, _   = self._perform_em(temp_sustainData, temp_seq_init, [1], rng)

        f_init                              = np.array([1.] * N_S) / float(N_S)

        # optimise the mixture of two models from the initialisation
        this_ml_sequence, \
        this_ml_f, \
        this_ml_likelihood, _, _, _         = self._perform_em(sustainData, seq_init, f_init, rng)

        return this_ml_sequence, this_ml_f, this_ml_likelihood

    #********************************************
    def _find_ml_mixture(self, sustainData, seq_init, f_init):
        # Fit a mixture of models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f          - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S                                 = seq_init.shape[0]

        partial_iter                        = partial(self._find_ml_mixture_iteration, sustainData, seq_init, f_init)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = np.zeros((N_S, sustainData.getNumStages(), self.N_startpoints))
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))

        for i in range(self.N_startpoints):
            ml_sequence_mat[:, :, i]        = pool_output_list[i][0]
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = np.where(ml_likelihood_mat == max(ml_likelihood_mat))
        ix                                  = ix[0]

        ml_sequence                         = ml_sequence_mat[:, :, ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_mixture_iteration(self, sustainData, seq_init, f_init, seed_seq):
        #Convenience sub-function for above

        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood                  = self._perform_em(sustainData, seq_init, f_init, rng)

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
    #********************************************

    def _perform_em(self, sustainData, current_sequence, current_f, rng):

        # Perform an E-M procedure to estimate parameters of SuStaIn model
        MaxIter                             = 100

        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]
        N_S                                 = current_sequence.shape[0]
        current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f)

        terminate                           = 0
        iteration                           = 0
        samples_sequence                    = np.nan * np.ones((MaxIter, N, N_S))
        samples_f                           = np.nan * np.ones((MaxIter, N_S))
        samples_likelihood                  = np.nan * np.ones((MaxIter, 1))

        samples_sequence[0, :, :]           = current_sequence.reshape(current_sequence.shape[1], current_sequence.shape[0])
        current_f                           = np.array(current_f).reshape(len(current_f))
        samples_f[0, :]                     = current_f
        samples_likelihood[0]               = current_likelihood
        while terminate == 0:

            candidate_sequence,     \
            candidate_f,            \
            candidate_likelihood            = self._optimise_parameters(sustainData, current_sequence, current_f, rng)

            HAS_converged                   = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6
            if HAS_converged:
                #print('EM converged in', iteration + 1, 'iterations')
                terminate                   = 1
            else:
                if candidate_likelihood > current_likelihood:
                    current_sequence        = candidate_sequence
                    current_f               = candidate_f
                    current_likelihood      = candidate_likelihood

            samples_sequence[iteration, :, :] = current_sequence.T.reshape(current_sequence.T.shape[0], N_S)
            samples_f[iteration, :]         = current_f
            samples_likelihood[iteration]   = current_likelihood

            if iteration == (MaxIter - 1):
                terminate                   = 1
            iteration                       = iteration + 1

        ml_sequence                         = current_sequence
        ml_f                                = current_f
        ml_likelihood                       = current_likelihood
        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _calculate_likelihood(self, sustainData, S, f):
        # Computes the likelihood of a mixture of models
        #
        #
        # OUTPUTS:
        # loglike               - the log-likelihood of the current model
        # total_prob_subj       - the total probability of the current SuStaIn model for each subject
        # total_prob_stage      - the total probability of each stage in the current SuStaIn model
        # total_prob_cluster    - the total probability of each subtype in the current SuStaIn model
        # p_perm_k              - the probability of each subjects data at each stage of each subtype in the current SuStaIn model

        M                                   = sustainData.getNumSamples()  #data_local.shape[0]
        N_S                                 = S.shape[0]
        N                                   = sustainData.getNumStages()    #self.stage_zscore.shape[1]

        f                                   = np.array(f).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S[s])  #self.__calculate_likelihood_stage_linearzscoremodel_approx(data_local, S[s])


        total_prob_cluster                  = np.squeeze(np.sum(p_perm_k * f_val_mat, 1))
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        loglike                             = np.sum(np.log(total_prob_subj + 1e-250))

        return loglike, total_prob_subj, total_prob_stage, total_prob_cluster, p_perm_k

    def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
        # Estimate the uncertainty in the subtype progression patterns and
        # proportion of individuals belonging to the SuStaIn model
        #
        #
        # OUTPUTS:
        # ml_sequence       - the most probable ordering of the stages for each subtype found across MCMC samples
        # ml_f              - the most probable proportion of individuals belonging to each subtype found across MCMC samples
        # ml_likelihood     - the likelihood of the most probable SuStaIn model found across MCMC samples
        # samples_sequence  - samples of the ordering of the stages for each subtype obtained from MCMC sampling
        # samples_f         - samples of the proportion of individuals belonging to each subtype obtained from MCMC sampling
        # samples_likeilhood - samples of the likelihood of each SuStaIn model sampled by the MCMC sampling

        # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
        seq_sigma_opt, f_sigma_opt          = self._optimise_mcmc_settings(sustainData, seq_init, f_init)

        # Run the full MCMC algorithm to estimate the uncertainty
        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood                  = self._perform_mcmc(sustainData, seq_init, f_init, self.N_iterations_MCMC, seq_sigma_opt, f_sigma_opt)

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):

        # Optimise the perturbation size for the MCMC algorithm
        n_iterations_MCMC_optimisation      = int(1e4)  # FIXME: set externally

        n_passes_optimisation               = 3

        seq_sigma_currentpass               = 1
        f_sigma_currentpass                 = 0.01  # magic number

        N_S                                 = seq_init.shape[0]

        for i in range(n_passes_optimisation):

            _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = self._perform_mcmc(   sustainData,
                                                                                                     seq_init,
                                                                                                     f_init,
                                                                                                     n_iterations_MCMC_optimisation,
                                                                                                     seq_sigma_currentpass,
                                                                                                     f_sigma_currentpass)

            samples_position_currentpass    = np.zeros(samples_sequence_currentpass.shape)
            for s in range(N_S):
                for sample in range(n_iterations_MCMC_optimisation):
                    temp_seq                        = samples_sequence_currentpass[s, :, sample]
                    temp_inv                        = np.array([0] * samples_sequence_currentpass.shape[1])
                    temp_inv[temp_seq.astype(int)]  = np.arange(samples_sequence_currentpass.shape[1])
                    samples_position_currentpass[s, :, sample] = temp_inv

            seq_sigma_currentpass           = np.std(samples_position_currentpass, axis=2, ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default
            seq_sigma_currentpass[seq_sigma_currentpass < 0.01] = 0.01  # magic number

            f_sigma_currentpass             = np.std(samples_f_currentpass, axis=1, ddof=1)         # np.std is different to Matlab std, which normalises to N-1 by default

        seq_sigma_opt                       = seq_sigma_currentpass
        f_sigma_opt                         = f_sigma_currentpass

        return seq_sigma_opt, f_sigma_opt

    def _evaluate_likelihood_setofsamples(self, sustainData, samples_sequence, samples_f):
    
        n_total                             = samples_sequence.shape[2]
    
        #reduce the number of samples to speed this function up
        if n_total >= 1e6:
            N_samples                       = int(np.round(n_total/1000))
        elif n_total >= 1e5:
            N_samples                       = int(np.round(n_total/100))
        else:
            N_samples                       = n_total        
        select_samples                      = np.round(np.linspace(0, n_total - 1, N_samples)).astype(int)               
    
        samples_sequence                    = samples_sequence[:, :, select_samples]
        samples_f                           = samples_f[:, select_samples]
    
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters
        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        n_iterations                        = samples_sequence.shape[2]
        samples_likelihood_subj             = np.zeros((M, n_iterations))
        for i in range(n_iterations):
            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]

            _, likelihood_sample_subj, _, _, _  = self._calculate_likelihood(sustainData, S, f)

            samples_likelihood_subj[:, i]   = likelihood_sample_subj

        return samples_likelihood_subj


    # ********************* ABSTRACT METHODS
    @abstractmethod
    def _initialise_sequence(self, sustainData, rng):
        pass

    @abstractmethod
    def _calculate_likelihood_stage(self, sustainData, S):
        pass

    @abstractmethod
    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        pass

    @abstractmethod
    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        pass

    @abstractmethod
    def _plot_sustain_model():
        pass

    @staticmethod
    @abstractmethod
    def plot_positional_var():
        pass

    @abstractmethod
    def subtype_and_stage_individuals_newData(self):    #up to the implementations to define exact number of params here
        pass

    # ********************* STATIC METHODS
    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)

    @staticmethod
    def check_biomarker_colours(biomarker_colours, biomarker_labels):
        if isinstance(biomarker_colours, dict):
            # Check each label exists
            assert all(i in biomarker_labels for i in biomarker_colours.keys()), "A label doesn't match!"
            # Check each colour exists
            assert all(mcolors.is_color_like(i) for i in biomarker_colours.values()), "A proper colour wasn't given!"
            # Add in any colours that aren't defined, allowing for partial colouration
            for label in biomarker_labels:
                if label not in biomarker_colours:
                    biomarker_colours[label] = "black"
        elif isinstance(biomarker_colours, (list, tuple)):
            # Check each colour exists
            assert all(mcolors.is_color_like(i) for i in biomarker_colours), "A proper colour wasn't given!"
            # Check right number of colours given
            assert len(biomarker_colours) == len(biomarker_labels), "The number of colours and labels do not match!"
            # Turn list of colours into a label:colour mapping
            biomarker_colours = {k:v for k,v in zip(biomarker_labels, biomarker_colours)}
        else:
            raise TypeError("A dictionary mapping label:colour or list/tuple of colours must be given!")
        return biomarker_colours

    # ********************* TEST METHODS
    @staticmethod
    @abstractmethod
    def generate_random_model():
        pass

    @staticmethod
    @abstractmethod
    def generate_data():
        pass

    @classmethod
    @abstractmethod
    def test_sustain(cls):
        pass
