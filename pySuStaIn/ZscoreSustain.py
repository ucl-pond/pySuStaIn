###
# pySuStaIn: a Python implementation of the Subtype and Stage Inference (SuStaIn) algorithm
#
# If you use pySuStaIn, please cite the following core papers:
# 1. The original SuStaIn paper:    https://doi.org/10.1038/s41467-018-05892-0
# 2. The pySuStaIn software paper:  https://doi.org/10.1016/j.softx.2021.100811

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
from multiprocessing import Value
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from pySuStaIn.AbstractSustain import AbstractSustainData
from pySuStaIn.AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for ZscoreSustain. It holds the z-scored data that gets passed around and re-indexed in places.
class ZScoreSustainData(AbstractSustainData):

    def __init__(self, data, numStages):
        self.data           = data
        self.__numStages    = numStages

    def getNumSamples(self):
        return self.data.shape[0]

    def getNumBiomarkers(self):
        return self.data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        return ZScoreSustainData(self.data[index,], self.__numStages)

#*******************************************
#An implementation of the AbstractSustain class with multiple events for each biomarker based on deviations from normality, measured in z-scores.
#There are a fixed number of thresholds for each biomarker, specified at initialization of the ZscoreSustain object.
class ZscoreSustain(AbstractSustain):

    def __init__(self,
                 data,
                 Z_vals,
                 Z_max,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the z-score based events implementation of AbstractSustain
        # Parameters:
        #   data                        - !important! needs to be (positive) z-scores!
        #                                 dim: number of subjects x number of biomarkers
        #   Z_vals                      - a matrix specifying the z-score thresholds for each biomarker
        #                                 for M biomarkers and 3 thresholds (1,2 and 3 for example) this would be a dim: M x 3 matrix
        #   Z_max                       - a vector specifying the maximum z-score for each biomarker
        #                                 when using z-score thresholds of 1,2,3 this would typically be 5.
        #                                 for M biomarkers this would be a dim: M x 1 vector
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        N                               = data.shape[1]  # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        stage_zscore            = Z_vals.T.flatten()    #np.array([y for x in Z_vals.T for y in x])
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))
        IX_select               = stage_zscore>0
        stage_zscore            = stage_zscore[IX_select]
        stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))

        num_zscores             = Z_vals.shape[1]
        IX_vals                 = np.array([[x for x in range(N)]] * num_zscores).T
        stage_biomarker_index   = IX_vals.T.flatten()   #np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index   = stage_biomarker_index[IX_select]
        stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        self.Z_vals                     = Z_vals
        self.stage_zscore               = stage_zscore
        self.stage_biomarker_index      = stage_biomarker_index

        self.min_biomarker_zscore       = [0] * N
        self.max_biomarker_zscore       = Z_max
        self.std_biomarker_zscore       = [1] * N

        self.biomarker_labels           = biomarker_labels

        numStages                       = stage_zscore.shape[1]
        self.__sustainData              = ZScoreSustainData(data, numStages)

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

        N                                   = np.array(self.stage_zscore).shape[1]
        S                                   = np.zeros(N)
        for i in range(N):

            IS_min_stage_zscore             = np.array([False] * N)
            possible_biomarkers             = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected               = [False] * N
                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k]        = True

                this_biomarkers             = np.array([(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) +
                                                        (np.array(IS_unselected) == 1).astype(int)]) == 2
                if not np.any(this_biomarkers):
                    this_min_stage_zscore   = 0
                else:
                    this_min_stage_zscore   = min(self.stage_zscore[this_biomarkers])
                if (this_min_stage_zscore):
                    temp                    = ((this_biomarkers.astype(int) + (self.stage_zscore == this_min_stage_zscore).astype(int)) == 2).T
                    temp                    = temp.reshape(len(temp), )
                    IS_min_stage_zscore[temp] = True

            events                          = np.array(range(N))
            possible_events                 = np.array(events[IS_min_stage_zscore])
            this_index                      = np.ceil(rng.random() * ((len(possible_events)))) - 1
            S[i]                            = possible_events[int(this_index)]

        S                                   = S.reshape(1, len(S))
        return S

    def _calculate_likelihood_stage(self, sustainData, S):
        '''
         Computes the likelihood of a single linear z-score model using an
         approximation method (faster)
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        N                                   = self.stage_biomarker_index.shape[1]
        S_inv                               = np.array([0] * N)
        S_inv[S.astype(int)]                = np.arange(N)
        possible_biomarkers                 = np.unique(self.stage_biomarker_index)
        B                                   = len(possible_biomarkers)
        point_value                         = np.zeros((B, N + 2))

        # all the arange you'll need below
        arange_N                            = np.arange(N + 2)

        for i in range(B):
            b                               = possible_biomarkers[i]
            event_location                  = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            event_value                     = np.concatenate([[self.min_biomarker_zscore[i]], self.stage_zscore[self.stage_biomarker_index == b], [self.max_biomarker_zscore[i]]])
            for j in range(len(event_location) - 1):

                if j == 0:  # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits

                    # original
                    #temp                   = np.arange(event_location[j],event_location[j+1]+2)
                    #point_value[i,temp]    = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)

                    # fastest by a bit
                    temp                    = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp]    = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

                else:
                    # original
                    #temp                   = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                    #point_value[i, temp]   = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

                    # fastest by a bit
                    temp                    = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                    N_j                     = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp]    = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

        stage_value                         = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        p_perm_k                            = np.zeros((M, N + 1))

        # optimised likelihood calc - take log and only call np.exp once after loop
        sigmat = np.array(self.std_biomarker_zscore)

        factor                              = np.log(1. / np.sqrt(np.pi * 2.0) * sigmat)
        coeff                               = np.log(1. / float(N + 1))

        # original
        """
        for j in range(N+1):
            x                   = (data-np.tile(stage_value[:,j],(M,1)))/sigmat
            p_perm_k[:,j]       = coeff+np.sum(factor-.5*x*x,1)
        """
        # faster - do the tiling once
        # stage_value_tiled                   = np.tile(stage_value, (M, 1))
        # N_biomarkers                        = stage_value.shape[0]
        # for j in range(N + 1):
        #     stage_value_tiled_j             = stage_value_tiled[:, j].reshape(M, N_biomarkers)
        #     x                               = (sustainData.data - stage_value_tiled_j) / sigmat  #(data_local - stage_value_tiled_j) / sigmat
        #     p_perm_k[:, j]                  = coeff + np.sum(factor - .5 * np.square(x), 1)
        # p_perm_k                            = np.exp(p_perm_k)

        # even faster - do in one go
        x = (sustainData.data[:, :, None] - stage_value) / sigmat[None, :, None]
        p_perm_k = coeff + np.sum(factor[None, :, None] - 0.5 * np.square(x), 1)
        p_perm_k = np.exp(p_perm_k)

        return p_perm_k


    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()   #data_local.shape[0]
        N_S                                 = S_init.shape[0]
        N                                   = self.stage_zscore.shape[1]

        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted                   = p_perm_k * f_val_mat
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

                this_stage_zscore           = self.stage_zscore[0, selected_event]
                selected_biomarker          = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker  = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter                  = possible_zscores_biomarker < this_stage_zscore
                max_filter                  = possible_zscores_biomarker > this_stage_zscore
                events                      = np.array(range(N))
                if np.any(min_filter):
                    min_zscore_bound        = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event  = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound        = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event  = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
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
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood         = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood              = max(possible_likelihood)
                this_S                      = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S                      = this_S[0, :]
                S_opt[s]                    = this_S
                this_p_perm_k               = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s]           = this_p_perm_k[:, :, 0]

            S_opt[s]                        = this_S

        p_perm_k_weighted                   = p_perm_k * f_val_mat
        #adding 1e-250 fixes divide by zero problem that happens rarely
        #p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        p_perm_k_norm                       = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        total_prob_stage                    = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj                     = np.sum(total_prob_stage, 1)

        likelihood_opt                      = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt


    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters

        N                                   = self.stage_zscore.shape[1]
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
                    this_stage_zscore       = self.stage_zscore[0, selected_event]
                    selected_biomarker      = self.stage_biomarker_index[0, selected_event]
                    possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

                    # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                    min_filter              = possible_zscores_biomarker < this_stage_zscore
                    max_filter              = possible_zscores_biomarker > this_stage_zscore
                    events                  = np.array(range(N))
                    if np.any(min_filter):
                        min_zscore_bound            = max(possible_zscores_biomarker[min_filter])
                        min_zscore_bound_event      = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_lower_bound   = current_location[min_zscore_bound_event] + 1
                    else:
                        move_event_to_lower_bound   = 0

                    if np.any(max_filter):
                        max_zscore_bound            = min(possible_zscores_biomarker[max_filter])
                        max_zscore_bound_event      = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_upper_bound   = current_location[max_zscore_bound_event]
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
        return ZscoreSustain.plot_positional_var(*args, Z_vals=self.Z_vals, **kwargs)

    def subtype_and_stage_individuals_newData(self, data_new, samples_sequence, samples_f, N_samples):

        numStages_new                   = self.__sustainData.getNumStages() #data_new.shape[1]
        sustainData_newData             = ZScoreSustainData(data_new, numStages_new)

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
    def plot_positional_var(samples_sequence, samples_f, n_samples, Z_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        # Get the number of subtypes
        N_S = samples_sequence.shape[0]
        # Get the number of features/biomarkers
        N_bio = Z_vals.shape[0]
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
        elif isinstance(subtype_order, tuple):
            subtype_order = list(subtype_order)
        # Unravel the stage zscores from Z_vals
        stage_zscore = Z_vals.T.flatten()
        IX_select = np.nonzero(stage_zscore)[0]
        stage_zscore = stage_zscore[IX_select][None, :]
        # Get the z-scores and their number
        zvalues = np.unique(stage_zscore)
        N_z = len(zvalues)
        # Extract which biomarkers have which zscores/stages
        stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        # Warn user of reordering if labels and order given
        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        if biomarker_order is not None:
            # self._plot_biomarker_order is not suited to zscore version
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
                for j, z in enumerate(zvalues):
                    # Determine which colours to alter
                    # I.e. red (1,0,0) means removing green & blue channels
                    # according to the certainty of red (representing z-score 1)
                    alter_level = colour_mat[j] == 0
                    # Extract the uncertainties for this z-score
                    confus_matrix_zscore = confus_matrix[(stage_zscore==z)[0]]
                    # Subtract the certainty for this colour
                    confus_matrix_c[
                        np.ix_(
                            stage_biomarker_index[(stage_zscore==z)[0]], range(N),
                            alter_level
                        )
                    ] -= np.tile(
                        confus_matrix_zscore.reshape((stage_zscore==z).sum(), N, 1),
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
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, 
    ground_truth_subtypes, sustain_kwargs, seed=42):
        # Set a global seed to propagate
        np.random.seed(seed)
        # Create Z values
        Z_vals = np.tile(np.arange(1, 4), (n_biomarkers, 1))
        Z_vals[0, 2] = 0

        Z_max = np.full((n_biomarkers,), 5)
        Z_max[2] = 2

        ground_truth_sequences = cls.generate_random_model(Z_vals, n_subtypes)
        N_stages = np.sum(Z_vals > 0) + 1

        ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
        ground_truth_stages = np.vstack((ground_truth_stages_control, ground_truth_stages_other)).astype(int)

        data, data_denoised, stage_value = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            Z_vals,
            Z_max
        )

        return cls(
            data, Z_vals, Z_max,
            **sustain_kwargs
        )

    @staticmethod
    def generate_random_model(Z_vals, N_S, seed=None):
        num_biomarkers = Z_vals.shape[0]

        stage_zscore = Z_vals.T.flatten()#[np.newaxis, :]

        IX_select = np.nonzero(stage_zscore)[0]
        stage_zscore = stage_zscore[IX_select]#[np.newaxis, :]
        num_zscores = Z_vals.shape[0]

        stage_biomarker_index = np.tile(np.arange(num_biomarkers), (num_zscores,))
        stage_biomarker_index = stage_biomarker_index[IX_select]#[np.newaxis, :]

        N = stage_zscore.shape[0]
        S = np.zeros((N_S, N))
        # Moved outside loop, no need
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

    # TODO: Refactor this as above
    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, Z_vals, Z_max):
        B = Z_vals.shape[0]
        stage_zscore = np.array([y for x in Z_vals.T for y in x])
        stage_zscore = stage_zscore.reshape(1,len(stage_zscore))
        IX_select = stage_zscore>0
        stage_zscore = stage_zscore[IX_select]
        stage_zscore = stage_zscore.reshape(1,len(stage_zscore))

        num_zscores = Z_vals.shape[1]
        IX_vals = np.array([[x for x in range(B)]] * num_zscores).T
        stage_biomarker_index = np.array([y for x in IX_vals.T for y in x])
        stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        stage_biomarker_index = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

        min_biomarker_zscore = [0]*B
        max_biomarker_zscore = Z_max
        std_biomarker_zscore = [1]*B

        N = stage_biomarker_index.shape[1]
        N_S = gt_ordering.shape[0]

        possible_biomarkers = np.unique(stage_biomarker_index)
        stage_value = np.zeros((B,N+2,N_S))

        for s in range(N_S):
            S = gt_ordering[s,:]
            S_inv = np.array([0]*N)
            S_inv[S.astype(int)] = np.arange(N)
            for i in range(B):
                b = possible_biomarkers[i]
                event_location = np.concatenate([[0], S_inv[(stage_biomarker_index == b)[0]], [N]])

                event_value = np.concatenate([[min_biomarker_zscore[i]], stage_zscore[stage_biomarker_index == b], [max_biomarker_zscore[i]]])

                for j in range(len(event_location)-1):

                    if j == 0: # FIXME: nasty hack to get Matlab indexing to match up - necessary here because indices are used for linspace limits
                        index = np.arange(event_location[j],event_location[j+1]+2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+2)
                    else:
                        index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                        stage_value[i,index,s] = np.linspace(event_value[j],event_value[j+1],event_location[j+1]-event_location[j]+1)

        M = stages.shape[0]
        data_denoised = np.zeros((M,B))
        for m in range(M):
            data_denoised[m,:] = stage_value[:,int(stages[m]),subtypes[m]]
        data = data_denoised + norm.ppf(np.random.rand(B,M).T)*np.tile(std_biomarker_zscore,(M,1))

        return data, data_denoised, stage_value
