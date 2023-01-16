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
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from pySuStaIn.AbstractSustain import AbstractSustainData
from pySuStaIn.AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for OrdinalSustain. It holds the score and negative likelihoods that get passed around and re-indexed in places.
class OrdinalSustainData(AbstractSustainData):

    def __init__(self, prob_nl, prob_score, numStages):
        self.prob_nl        = prob_nl
        self.prob_score     = prob_score
        self.__numStages    = numStages

    def getNumSamples(self):
        return self.prob_nl.shape[0]

    def getNumBiomarkers(self):
        return self.prob_nl.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        return OrdinalSustainData(self.prob_nl[index,], self.prob_score[index,], self.__numStages)

#*******************************************
#An implementation of the AbstractSustain class with multiple events for each biomarker based on deviations from normality, measured in z-scores.
#There are a fixed number of thresholds for each biomarker, specified at initialization of the OrdinalSustain object.
class OrdinalSustain(AbstractSustain):

    def __init__(self,
                 prob_nl,
                 prob_score,
                 score_vals,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the scored events model implementation of AbstractSustain
        # Parameters:
        #   prob_nl                     - probability of negative/normal class for all subjects across all biomarkers 
        #                                 dim: number of subjects x number of biomarkers
        #   prob_score                  - probability of each score for all subjects across all biomarkers
        #                                 dim: number of subjects x number of biomarkers x number of scores
        #   score_vals                  - a matrix specifying the scores for each biomarker
        #                                 dim: number of biomarkers x number of scores
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        N                               = prob_nl.shape[1]  # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        num_scores                     = score_vals.shape[1]
        IX_vals                         = np.array([[x for x in range(N)]] * num_scores).T

        stage_score            = np.array([y for x in score_vals.T for y in x])
        stage_score            = stage_score.reshape(1,len(stage_score))
        IX_select              = stage_score>0
        stage_score            = stage_score[IX_select]
        stage_score            = stage_score.reshape(1,len(stage_score))

        num_scores              = score_vals.shape[1]
        IX_vals                 = np.array([[x for x in range(N)]] * num_scores).T
        stage_biomarker_index   = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index   = stage_biomarker_index[IX_select]
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        prob_score = prob_score.transpose(0,2,1)
        prob_score = prob_score.reshape(prob_score.shape[0],prob_score.shape[1]*prob_score.shape[2])
        prob_score = prob_score[:,IX_select[0,:]]
        prob_score = prob_score.reshape(prob_nl.shape[0],stage_score.shape[1])

        self.IX_select                  = IX_select

        self.score_vals                 = score_vals
        self.stage_score                = stage_score
        self.stage_biomarker_index      = stage_biomarker_index

        self.biomarker_labels           = biomarker_labels

        numStages                       = stage_score.shape[1]
        self.__sustainData              = OrdinalSustainData(prob_nl, prob_score, numStages)

        super().__init__(self.__sustainData,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints,
                         seed)


    def _initialise_sequence(self, sustainData, rng):
        # Randomly initialises a linear z-score model ensuring that the biomarkers
        # are monotonically increasing
        #
        #
        # OUTPUTS:
        # S - a random linear z-score model under the condition that each biomarker
        # is monotonically increasing

        N                                   = np.array(self.stage_score).shape[1]
        S                                   = np.zeros(N)
        for i in range(N):

            IS_min_stage_score             = np.array([False] * N)
            possible_biomarkers             = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected               = [False] * N
                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k]        = True

                this_biomarkers             = np.array([(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) +
                                                        (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_score   = 0
                else:
                    this_min_stage_score   = min(self.stage_score[this_biomarkers])
                if (this_min_stage_score):
                    temp                    = ((this_biomarkers.astype(int) + (self.stage_score == this_min_stage_score).astype(int)) == 2).T
                    temp                    = temp.reshape(len(temp), )
                    IS_min_stage_score[temp] = True

            events                          = np.array(range(N))
            possible_events                 = np.array(events[IS_min_stage_score])
            this_index                      = np.ceil(rng.random() * ((len(possible_events)))) - 1
            S[i]                            = possible_events[int(this_index)]

        S                                   = S.reshape(1, len(S))
        return S

    def _calculate_likelihood_stage(self, sustainData, S):
        '''
         Computes the likelihood of a single scored event model
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        N = self.stage_score.shape[1]

        B = sustainData.prob_nl.shape[1]
    
        IS_normal = np.ones(B)
        IS_abnormal = np.zeros(B)
        index_reached = np.zeros(B,dtype=int)

        M = sustainData.prob_score.shape[0]
        p_perm_k = np.zeros((M,N+1))
        p_perm_k[:,0] = 1/(N+1)*np.prod(sustainData.prob_nl,1)

        for j in range(N):
            index_justreached = int(S[j])
            biomarker_justreached = int(self.stage_biomarker_index[:,index_justreached])
            index_reached[biomarker_justreached] = index_justreached
            IS_normal[biomarker_justreached] = 0
            IS_abnormal[biomarker_justreached] = 1
            bool_IS_normal = IS_normal.astype(bool)
            bool_IS_abnormal = IS_abnormal.astype(bool)
            p_perm_k[:,j+1] = 1/(N+1)*np.multiply(np.prod(sustainData.prob_score[:,index_reached[bool_IS_abnormal]],1),np.prod(sustainData.prob_nl[:,bool_IS_normal],1))

        return p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        N_S                                 = S_init.shape[0]
        N                                   = self.stage_score.shape[1]

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        # adding 1e-250 fixes divide by zero problem that happens rarely
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio                       = rng.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_location            = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event              = i

                move_event_from             = current_location[selected_event]

                this_stage_score           = self.stage_score[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_scores_biomarker  = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter                  = possible_scores_biomarker < this_stage_score
                max_filter                  = possible_scores_biomarker > this_stage_score
                events                      = np.array(range(N))
                if np.any(min_filter):
                    min_score_bound        = max(possible_scores_biomarker[min_filter])
                    min_score_bound_event  = events[((self.stage_score[0] == min_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_score_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_score_bound        = min(possible_scores_biomarker[max_filter])
                    max_score_bound_event  = events[((self.stage_score[0] == max_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_score_bound_event]
                else:
                    move_event_to_upper_bound = N
                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions      = np.array([0])
                else:
                    possible_positions      = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
                possible_sequences          = np.zeros((len(possible_positions), N))
                possible_likelihood         = np.zeros((len(possible_positions), 1))
                possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence        = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to           = possible_positions[index]

                    # move this event in its new position
                    current_sequence        = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s]       = possible_p_perm_k[:, :, index]
                    total_prob_stage        = np.sum(p_perm_k * f_val_mat, 2)
                    total_prob_subj         = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood              = max(possible_likelihood)
                this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S                      = this_S[0, :]
                S_opt[s]                    = this_S
                this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

            S_opt[s]                        = this_S

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)

        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters

        N                                   = self.stage_score.shape[1]
        N_S                                 = seq_init.shape[0]

        if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0]           = seq_init  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                seq_order                   = self.global_rng.permutation(N_S)  # this function returns different random numbers to Matlab
                for s in seq_order:
                    move_event_from         = int(np.ceil(N * self.global_rng.random())) - 1
                    current_sequence        = samples_sequence[s, :, i - 1]

                    current_location        = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event          = int(current_sequence[move_event_from])
                    this_stage_score       = self.stage_score[0, selected_event]
                    selected_biomarker      = self.stage_biomarker_index[0, selected_event]
                    possible_scores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                    # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                    min_filter              = possible_scores_biomarker < this_stage_score
                    max_filter              = possible_scores_biomarker > this_stage_score
                    events                  = np.array(range(N))
                    if np.any(min_filter):
                        min_score_bound            = max(possible_scores_biomarker[min_filter])
                        min_score_bound_event      = events[((self.stage_score[0] == min_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_lower_bound   = current_location[min_score_bound_event] + 1
                    else:
                        move_event_to_lower_bound   = 0

                    if np.any(max_filter):
                        max_score_bound            = min(possible_scores_biomarker[max_filter])
                        max_score_bound_event      = events[((self.stage_score[0] == max_score_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_upper_bound   = current_location[max_score_bound_event]
                    else:
                        move_event_to_upper_bound   = N

                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                    if move_event_to_lower_bound == move_event_to_upper_bound:
                        possible_positions          = np.array([0])
                    else:
                        possible_positions          = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                    distance                = possible_positions - move_event_from

                    if isinstance(seq_sigma, int):  # FIXME: change to float
                        this_seq_sigma      = seq_sigma
                    else:
                        this_seq_sigma      = seq_sigma[s, selected_event]

                    # use own normal PDF because stats.norm is slow
                    weight                  = AbstractSustain.calc_coeff(this_seq_sigma) * AbstractSustain.calc_exp(distance, 0., this_seq_sigma)
                    weight                  /= np.sum(weight)
                    index                   = self.global_rng.choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                    move_event_to           = possible_positions[index]

                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[s, :, i] = new_sequence

                new_f                       = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f

            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)
            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
        perm_index                          = perm_index[0]
        ml_likelihood                       = max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, *args, **kwargs):
        return OrdinalSustain.plot_positional_var(*args, score_vals=self.score_vals, **kwargs)

    def subtype_and_stage_individuals_newData(self, prob_nl_new, prob_score_new, samples_sequence, samples_f, N_samples):

        numBio_new                   = prob_nl_new.shape[1]
        assert numBio_new == self.__sustainData.getNumBiomarkers(), "Number of biomarkers in new data should be same as in training data"

        numStages = self.__sustainData.getNumStages()
        
        prob_score_new = prob_score_new.transpose(0,2,1)
        prob_score_new = prob_score_new.reshape(prob_score_new.shape[0],prob_score_new.shape[1]*prob_score_new.shape[2])
        prob_score_new = prob_score_new[:,self.IX_select[0,:]]
        prob_score_new = prob_score_new.reshape(prob_nl_new.shape[0],self.stage_score.shape[1])

        sustainData_newData             = OrdinalSustainData(prob_nl_new, prob_score_new, numStages)

        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage          = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, N_samples)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage


    # ********************* STATIC METHODS
    @staticmethod
    def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    @staticmethod
    def plot_positional_var(samples_sequence, samples_f, n_samples, score_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        # Get the number of subtypes
        N_S = samples_sequence.shape[0]
        # Get the number of features/biomarkers
        N_bio = score_vals.shape[0]
        # Check that the number of labels given match
        if biomarker_labels is not None:
            assert len(biomarker_labels) == N_bio
        # Set subtype order if not given
        if subtype_order is None:
            # Determine order if info given
            if ml_f_EM is not None:
                subtype_order = np.argsort(ml_f_EM)[::-1]
            # Otherwise determine order from samples_f
            else:
                subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
        # Unravel the stage scores from score_vals
        stage_score = score_vals.T.flatten()
        IX_select = np.nonzero(stage_score)[0]
        stage_score = stage_score[IX_select][None, :]
        # Get the scores and their number
        num_scores = np.unique(stage_score)
        N_z = len(num_scores)
        # Extract which biomarkers have which zscores/stages
        stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        # Warn user of reordering if labels and order given
        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        if biomarker_order is not None:
            # self._plot_biomarker_order is not suited to this version
            # Ignore for compatability, for now
            # One option is to reshape, sum position, and lowest->highest determines order
            if len(biomarker_order) > N_bio:
                biomarker_order = np.arange(N_bio)
        # Otherwise use default order
        else:
            biomarker_order = np.arange(N_bio)
        # If no labels given, set dummy defaults
        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
        # Otherwise reorder according to given order (or not if not given)
        else:
            biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
        # Check number of subtype titles is correct if given
        if subtype_titles is not None:
            assert len(subtype_titles) == N_S
        # Z-score colour definition
        if cmap == "original":
            # Hard-coded colours: hooray!
            colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]
            # We only have up to 5 default colours, so double-check
            if colour_mat.shape[0] > N_z:
                raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")
        else:
            raise NotImplementedError
        '''
        Note for future self/others: The use of any arbitrary colourmap is problematic, as when the same stage can have the same biomarker with different z-scores of different certainties, the colours need to mix in a visually informative way and there can be issues with RGB mixing/interpolation, particulary if there are >2 z-scores for the same biomarker at the same stage. It may be possible, but the end result may no longer be useful to look at.
        '''

        # Check biomarker label colours
        # If custom biomarker text colours are given
        if biomarker_colours is not None:
            biomarker_colours = AbstractSustain.check_biomarker_colours(
            biomarker_colours, biomarker_labels
        )
        # Default case of all-black colours
        # Unnecessary, but skips a check later
        else:
            biomarker_colours = {i:"black" for i in biomarker_labels}

        # Flag to plot subtypes separately
        if separate_subtypes:
            nrows, ncols = 1, 1
        else:
            # Determine number of rows and columns (rounded up)
            if N_S == 1:
                nrows, ncols = 1, 1
            elif N_S < 3:
                nrows, ncols = 1, N_S
            elif N_S < 7:
                nrows, ncols = 2, int(np.ceil(N_S / 2))
            else:
                nrows, ncols = 3, int(np.ceil(N_S / 3))
        # Total axes used to loop over
        total_axes = nrows * ncols
        # Create list of single figure object if not separated
        if separate_subtypes:
            subtype_loops = N_S
        else:
            subtype_loops = 1
        # Container for all figure objects
        figs = []
        # Loop over figures (only makes a diff if separate_subtypes=True)
        for i in range(subtype_loops):
            # Create the figure and axis for this subtype loop
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            figs.append(fig)
            # Loop over each axis
            for j in range(total_axes):
                # Normal functionality (all subtypes on one plot)
                if not separate_subtypes:
                    i = j
                # Handle case of a single array
                if isinstance(axs, np.ndarray):
                    ax = axs.flat[i]
                else:
                    ax = axs
                # Check if i is superfluous
                if i not in range(N_S):
                    ax.set_axis_off()
                    continue

                this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
                N = this_samples_sequence.shape[1]

                # Construct confusion matrix (vectorized)
                # We compare `this_samples_sequence` against each position
                # Sum each time it was observed at that point in the sequence
                # And normalize for number of samples/sequences
                confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]

                # Define the confusion matrix to insert the colours
                # Use 1s to start with all white
                confus_matrix_c = np.ones((N_bio, N, 3))

                # Loop over each z-score event
                for j, z in enumerate(num_scores):
                    # Determine which colours to alter
                    # I.e. red (1,0,0) means removing green & blue channels
                    # according to the certainty of red (representing z-score 1)
                    alter_level = colour_mat[j] == 0
                    # Extract the uncertainties for this score
                    confus_matrix_score = confus_matrix[(stage_score==z)[0]]
                    # Subtract the certainty for this colour
                    confus_matrix_c[
                        np.ix_(
                            stage_biomarker_index[(stage_score==z)[0]], range(N),
                            alter_level
                        )
                    ] -= np.tile(
                        confus_matrix_score.reshape((stage_score==z).sum(), N, 1),
                        (1, 1, alter_level.sum())
                    )
                    # Subtract the certainty for this colour
                    confus_matrix_c[:, :, alter_level] -= np.tile(
                        confus_matrix_score.reshape(N_bio, N, 1),
                        (1, 1, alter_level.sum())
                    )
                if subtype_titles is not None:
                    title_i = subtype_titles[i]
                else:
                    # Add axis title
                    if cval == False:
                        temp_mean_f = np.mean(samples_f, 1)
                        # Shuffle vals according to subtype_order
                        # This defaults to previous method if custom order not given
                        vals = temp_mean_f[subtype_order]

                        if n_samples != np.inf:
                            title_i = f"Subtype {i+1} (f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                        else:
                            title_i = f"Subtype {i+1} (f={vals[i]:.2f})"
                    else:
                        title_i = f"Subtype {i+1} cross-validated"
                # Plot the colourized matrix
                ax.imshow(
                    confus_matrix_c[biomarker_order, :, :],
                    interpolation='nearest'
                )
                # Add the xticks and labels
                stage_ticks = np.arange(0, N, stage_interval)
                ax.set_xticks(stage_ticks)
                ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size, rotation=stage_rot)
                # Add the yticks and labels
                ax.set_yticks(np.arange(N_bio))
                # Add biomarker labels to LHS of every row only
                if (i % ncols) == 0:
                    ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size, rotation=label_rot)
                    # Set biomarker label colours
                    for tick_label in ax.get_yticklabels():
                        tick_label.set_color(biomarker_colours[tick_label.get_text()])
                else:
                    ax.set_yticklabels([])
                # Make the event label slightly bigger than the ticks
                ax.set_xlabel(stage_label, fontsize=stage_font_size+2)
                ax.set_title(title_i, fontsize=title_font_size)
            # Tighten up the figure
            fig.tight_layout()
            # Save if a path is given
            if save_path is not None:
                # Modify path for specific subtype if specified
                # Don't modify save_path!
                if separate_subtypes:
                    save_name = f"{save_path}_subtype{i}"
                else:
                    save_name = f"{save_path}_all-subtypes"
                # Handle file format, avoids issue with . in filenames
                if "format" in save_kwargs:
                    file_format = save_kwargs.pop("format")
                # Default to png
                else:
                    file_format = "png"
                # Save the figure, with additional kwargs
                fig.savefig(
                    f"{save_name}.{file_format}",
                    **save_kwargs
                )
        return figs, axs

    # ********************* TEST METHODS
    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, sustain_kwargs, seed=42):
        # Set a global seed to propagate
        np.random.seed(seed)

        # # Set the number of biomarkers to 10
        # N = 10
        # # Set the number of subjects to 250
        # M = 250
        # # Set the ground truth number of subtypes to 2
        # N_S_gt = 2
        # Set the number of scores per biomarker to 3
        N_scores = 3
        score_vals = np.tile(np.arange(1, N_scores+1), (n_biomarkers, 1))

        # Set the fraction of individuals belonging to each subtype
        # gt_f = 1 + np.arange(n_subtypes) / 2
        # gt_f = (gt_f/np.sum(gt_f))[::-1]

        # Choose a random ground truth sequence for each subtype
        ground_truth_sequence = cls.generate_random_model(score_vals, n_subtypes)

        # Choose a random ground truth stage for each individual
        N_stages = np.sum(score_vals>0)
        N_k = N_stages + 1
        # gt_subtypes = np.random.choice(range(n_subtypes), n_samples, replace=True, p=gt_f)
        ground_truth_stages = np.ceil(np.random.rand(n_samples, 1)*N_k)-1

        # Set the proportion of individuals with correct scores to 0.9
        p_correct = 0.9
        p_nl_dist = np.full((N_scores+1), (1-p_correct)/(N_scores))
        p_nl_dist[0] = p_correct
        p_score_dist = np.full((N_scores, N_scores+1), (1-p_correct)/(N_scores))
        for score in range(N_scores):
            p_score_dist[score,score+1] = p_correct

        stage_score = score_vals.T.flatten()
        IX_select = np.nonzero(stage_score)[0]
        stage_score = stage_score[IX_select]
        stage_biomarker_index = np.tile(np.arange(n_biomarkers), (N_scores,))
        stage_biomarker_index = stage_biomarker_index[IX_select]

        prob_nl, prob_score = cls.generate_data(
            N_scores, n_samples, n_biomarkers, stage_biomarker_index, p_nl_dist,
            p_score_dist, ground_truth_subtypes, ground_truth_sequence, ground_truth_stages, stage_score
        )

        return cls(
            prob_nl, prob_score, score_vals,
            **sustain_kwargs
        )
    
    @staticmethod
    def generate_random_model(Z_vals, N_S, seed=None):
        num_biomarkers = Z_vals.shape[0]

        stage_zscore = Z_vals.T.flatten()

        IX_select = np.nonzero(stage_zscore)[0]
        stage_zscore = stage_zscore[IX_select]
        num_zscores = Z_vals.shape[0]

        stage_biomarker_index = np.tile(np.arange(num_biomarkers), (num_zscores,))
        stage_biomarker_index = stage_biomarker_index[IX_select]

        N = stage_zscore.shape[0]
        S = np.zeros((N_S, N))

        possible_biomarkers = np.unique(stage_biomarker_index)

        for s in range(N_S):
            for i in range(N):
                IS_min_stage_zscore = np.full(N, False)

                for j in possible_biomarkers:
                    IS_unselected = np.full(N, False)
                    # I have no idea what purpose this serves, so leaving for now
                    for k in set(range(N)) - set(S[s][:i]):
                        IS_unselected[k] = True

                    this_biomarkers = np.logical_and(
                        stage_biomarker_index == possible_biomarkers[j],
                        np.array(IS_unselected) == 1
                    )
                    if not np.any(this_biomarkers):
                        this_min_stage_zscore = 0
                    else:
                        this_min_stage_zscore = np.min(stage_zscore[this_biomarkers])
                    
                    if this_min_stage_zscore:
                        IS_min_stage_zscore[np.logical_and(
                            this_biomarkers,
                            stage_zscore == this_min_stage_zscore
                        )] = True

                events = np.arange(N)
                possible_events = events[IS_min_stage_zscore]
                this_index = np.ceil(np.random.rand() * len(possible_events)) - 1
                S[s][i] = possible_events[int(this_index)]
        return S

    @staticmethod
    def generate_data(N_scores, n_samples, n_biomarkers, stage_biomarker_index,
    p_nl_dist, p_score_dist, ground_truth_subtypes, ground_truth_sequence, ground_truth_stages, stage_score):
        # Simulate the data for each biomarker for each individual based on their subtype and stage
        data = np.random.choice(range(N_scores+1), n_samples*n_biomarkers, replace=True, p=p_nl_dist)
        data = data.reshape((n_samples, n_biomarkers))

        for m in range(n_samples):
            this_subtype = ground_truth_subtypes[m]
            this_stage = ground_truth_stages[m,0].astype(int)
            this_S = ground_truth_sequence[this_subtype, :].astype(int)
            
            this_ordered_biomarker_abnormal = stage_biomarker_index[this_S[:this_stage]]
            this_ordered_score_abnormal = stage_score[this_S[:this_stage]]

            temp_score_reached = np.zeros(n_biomarkers)
            for b in range(n_biomarkers):
                if (this_ordered_biomarker_abnormal==b).any():
                    temp_score_reached[b] = np.max(
                        this_ordered_score_abnormal[this_ordered_biomarker_abnormal==b]
                    )

            for score in range(N_scores):
                data[m, temp_score_reached==[score+1]] = np.random.choice(
                    range(N_scores+1),
                    np.sum(temp_score_reached==[score+1]),
                    replace=True,
                    p=p_score_dist[score,:]
                )

        # Turn the data into probabilities an individual has a normal score or one of the scores included in the Scored Event model
        prob_nl = p_nl_dist[data]
        # TODO: Refactor as above
        prob_score = np.zeros((n_samples, n_biomarkers, N_scores))
        for n in range(n_biomarkers):
            for z in range(N_scores):
                for score in range(N_scores+1):
                    prob_score[data[:, n] == score, n, z] = p_score_dist[z, score]
        return prob_nl, prob_score
