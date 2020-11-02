###
# pySuStaIn: Python translation of Matlab version of SuStaIn algorithm (https://www.nature.com/articles/s41467-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
#
# For questions/comments related to: object orient implementation of pySustain
# contact: Leon Aksman (l.aksman@ucl.ac.uk)
# For questions/comments related to: the SuStaIn algorithm
# contact: Alex Young (alexandra.young@kcl.ac.uk)
###
from tqdm import tqdm
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
                 seed):
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


    def _initialise_sequence(self, sustainData):
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
            this_index                      = np.ceil(np.random.rand() * ((len(possible_events)))) - 1
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

    def _optimise_parameters(self, sustainData, S_init, f_init):
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
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = np.random.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio                       = np.random.permutation(N)  # this will produce different random numbers to Matlab
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

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations):
            if i > 0:
                seq_order                   = np.random.permutation(N_S)  # this function returns different random numbers to Matlab
                for s in seq_order:
                    move_event_from         = int(np.ceil(N * np.random.rand())) - 1
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
                    index                   = np.random.choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                    move_event_to           = possible_positions[index]

                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[s, :, i] = new_sequence

                new_f                       = samples_f[:, i - 1] + f_sigma * np.random.randn()
                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f

            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)
            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < np.random.rand():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
        perm_index                          = perm_index[0]
        ml_likelihood                       = max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, samples_sequence, samples_f, n_samples, cval=False, plot_order=None, title_font_size=8):

        colour_mat                          = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1]]) #, [0.5, 0, 1], [0, 1, 1]])

        temp_mean_f                         = np.mean(samples_f, 1)
        vals                                = np.sort(temp_mean_f)[::-1]
        vals                                = np.array([np.round(x * 100.) for x in vals]) / 100.
        ix                                  = np.argsort(temp_mean_f)[::-1]

        N_S                                 = samples_sequence.shape[0]
        N_bio                               = len(self.biomarker_labels)

        if N_S == 1:
            fig, ax                         = plt.subplots()
            total_axes                      = 1;
        elif N_S < 3:
            fig, ax                         = plt.subplots(1, N_S)
            total_axes                      = N_S
        elif N_S < 7:
            fig, ax                         = plt.subplots(2, int(np.ceil(N_S / 2)))
            total_axes                      = 2 * int(np.ceil(N_S / 2))
        else:
            fig, ax                         = plt.subplots(3, int(np.ceil(N_S / 3)))
            total_axes                      = 3 * int(np.ceil(N_S / 3))


        for i in range(total_axes):        #range(N_S):

            if i not in range(N_S):
                ax.flat[i].set_axis_off()
                continue

            this_samples_sequence           = samples_sequence[ix[i],:,:].T
            markers                         = np.unique(self.stage_biomarker_index)
            N                               = this_samples_sequence.shape[1]

            confus_matrix                   = np.zeros((N, N))
            for j in range(N):
                confus_matrix[j, :]         = sum(this_samples_sequence == j)
            confus_matrix                   /= float(this_samples_sequence.shape[0])

            zvalues                         = np.unique(self.stage_score)
            N_z                             = len(zvalues)
            confus_matrix_z                 = np.zeros((N_bio, N, N_z))
            for z in range(N_z):
                confus_matrix_z[self.stage_biomarker_index[self.stage_score == zvalues[z]], :, z] = confus_matrix[(self.stage_score == zvalues[z])[0],:]

            confus_matrix_c                 = np.ones((N_bio, N, 3))
            for z in range(N_z):
                this_confus_matrix          = confus_matrix_z[:, :, z]
                this_colour                 = colour_mat[z, :]
                alter_level                 = this_colour == 0

                this_colour_matrix          = np.zeros((N_bio, N, 3))
                this_colour_matrix[:, :, alter_level] = np.tile(this_confus_matrix[markers, :].reshape(N_bio, N, 1), (1, 1, sum(alter_level)))
                confus_matrix_c             = confus_matrix_c - this_colour_matrix

            TITLE_FONT_SIZE                 = title_font_size
            X_FONT_SIZE                     = 8
            Y_FONT_SIZE                     = 7

            # must be a smarter way of doing this, but subplots(1,1) doesn't produce an array...
            if N_S > 1:
                ax_i                        = ax.flat[i] #ax[i]
                ax_i.imshow(confus_matrix_c, interpolation='nearest')      #, cmap=plt.cm.Blues)
                ax_i.set_xticks(np.arange(N))
                ax_i.set_xticklabels(range(1, N+1), rotation=45, fontsize=X_FONT_SIZE)

                ax_i.set_yticks(np.arange(N_bio))
                ax_i.set_yticklabels([]) #['']* N_bio)
                if i == 0:
                    ax_i.set_yticklabels(np.array(self.biomarker_labels, dtype='object'), ha='right', fontsize=Y_FONT_SIZE)
                    for tick in ax_i.yaxis.get_major_ticks():
                        tick.label.set_color('black')

                #ax[i].set_ylabel('Biomarker name') #, fontsize=20)
                ax_i.set_xlabel('SuStaIn stage', fontsize=X_FONT_SIZE)
                ax_i.set_title('Group ' + str(i) + ' (f=' + str(vals[i])  + r', n$\sim$' + str(int(np.round(vals[i] * n_samples)))  + ')', fontsize=TITLE_FONT_SIZE)

            else: #**** first plot
                ax.imshow(confus_matrix_c) #, interpolation='nearest')#, cmap=plt.cm.Blues) #[...,::-1]
                ax.set_xticks(np.arange(N))
                ax.set_xticklabels(range(1, N+1), rotation=45, fontsize=X_FONT_SIZE)

                ax.set_yticks(np.arange(N_bio))
                ax.set_yticklabels(np.array(self.biomarker_labels, dtype='object'), ha='right', fontsize=Y_FONT_SIZE)

                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_color('black')

                ax.set_xlabel('SuStaIn stage', fontsize=X_FONT_SIZE)
                ax.set_title('Group ' + str(i) + ' (f=' + str(vals[i])  + r', n$\sim$' + str(int(np.round(vals[i] * n_samples)))  + ')', fontsize=TITLE_FONT_SIZE)

        plt.tight_layout()
        if cval:
            fig.suptitle('Cross validation')

        return fig, ax


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