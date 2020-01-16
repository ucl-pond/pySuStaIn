###
# pySuStaIn: Python translation of Matlab version of SuStaIn algorithm (https://www.nature.com/articles/s41467-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk)
#
# For questions/comments related to: object orient implementation of pySustain
# contact: Leon Aksman (l.aksman@ucl.ac.uk)
###

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

from AbstractSustain import AbstractSustainData
from AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for MixtureSustain. It holds the positive/negative likelihoods that get passed around and re-indexed in places.
class MixtureSustainData(AbstractSustainData):

    def __init__(self, L_yes, L_no, numStages):

        assert(L_yes.shape[0] == L_no.shape[0] and L_yes.shape[1] == L_no.shape[1])

        self.L_yes          = L_yes
        self.L_no           = L_no
        self.__numStages    = numStages

    def getNumSamples(self):
        return self.L_yes.shape[0]

    def getNumBiomarkers(self):
        return self.L_no.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        return MixtureSustainData(self.L_yes[index,], self.L_no[index,], self.__numStages)

#*******************************************
#An implementation of the AbstractSustain class with z-score based events
class MixtureSustain(AbstractSustain):

    def __init__(self,
                 L_yes,
                 L_no,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints):
        # The initializer for the mixture model based events implementation of AbstractSustain
        # Parameters:
        #   L_yes                       - probability of positive class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   L_no                        - probability of negative class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop

        N                               =  L_yes.shape[1] # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        self.biomarker_labels           = biomarker_labels

        numStages                       = L_yes.shape[1]    #number of stages == number of biomarkers here
        self.__sustainData              = MixtureSustainData(L_yes, L_no, numStages)

        super().__init__(self.__sustainData,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints)

    def _initialise_sequence(self, sustainData):
        # Randomly initialises a sequence

        S                                   = MixtureSustain.randperm_local(sustainData.getNumStages()) #np.random.permutation(sustainData.getNumStages())
        S                                   = S.reshape(1, len(S))
        return S


    def _calculate_likelihood_stage(self, sustainData, S):
        '''
         Computes the likelihood of a single event based model

        Inputs:
        =======
        sustainData - a MixtureData type that contains:
            L_yes - likelihood an event has occurred in each subject
                    dim: number of subjects x number of biomarkers
            L_no -  likelihood an event has not occurred in each subject
                    dim: number of subjects x number of biomarkers
            S -     the current ordering of the z-score stages for a particular subtype
                    dim: 1 x number of events
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        M                                   = sustainData.getNumSamples()
        N                                   = sustainData.getNumStages()

        S_int                               = S.astype(int)

        arange_Np1                          = np.arange(0, N+1)

        p_perm_k                            = np.zeros((M, N+1))

        #**** THIS VERSION IS ROUGHLY 10x FASTER THAN THE ONE BELOW
        cp_yes                              = np.cumprod(sustainData.L_yes[:, S_int],        1)
        cp_no                               = np.cumprod(sustainData.L_no[:,  S_int[::-1]],  1)   #do the cumulative product from the end of the sequence
        for i in arange_Np1:

            if i == 0:
                p_perm_k[:, i]              = 1 / (N + 1) * cp_no[:,N-1]
            elif i == N:
                p_perm_k[:, i]              = 1 / (N + 1) * cp_yes[:,N-1]
            else:
                p_perm_k[:, i]              = 1 / (N + 1) * cp_yes[:,i-1] * cp_no[:,N-i-1]

        #**** STRAIGHTFORWARD VERSION - MUCH SLOWER
        # for i in arange_Np1: #range(N+1):
        #     occur                           = S_int[arange_Np1[0:i]]    #S_int[range(0, i, 1)] #S_int[0:(i - 1)]
        #     notoccur                        = S_int[i:]
        #
        #     p_perm_k[:, i]                  = 1 / (N + 1) * np.prod(sustainData.L_yes[:, occur], 1) * np.prod(sustainData.L_no[:, notoccur], 1)

        return p_perm_k


    def _optimise_parameters(self, sustainData, S_init, f_init):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = S_init.shape[0]
        N                                   = sustainData.getNumStages()

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
        order_seq                           = MixtureSustain.randperm_local(N_S)    #np.random.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio                       = MixtureSustain.randperm_local(N) #np.random.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence            = S_opt[s]
                assert(len(current_sequence)==N)
                current_location            = np.array([0] * N)
                current_location[current_sequence.astype(int)] = np.arange(N)

                selected_event              = i

                move_event_from             = current_location[selected_event]

                possible_positions          = np.arange(N)
                possible_sequences          = np.zeros((len(possible_positions), N))
                possible_likelihood         = np.zeros((len(possible_positions), 1))
                possible_p_perm_k           = np.zeros((M, N + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence        = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to           = possible_positions[index]

                    #move this event in its new position
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

        M                                   = sustainData.getNumSamples()
        N                                   = sustainData.getNumStages()
        N_S                                 = seq_init.shape[0]

        if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0]           = seq_init  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init

        for i in range(n_iterations):
            if i % (n_iterations / 10) == 0:
                print('Iteration', i, 'of', n_iterations, ',', int(float(i) / float(n_iterations) * 100.), '% complete')
            if i > 0:
                seq_order                   = MixtureSustain.randperm_local(N_S) #np.random.permutation(N_S)  # this function returns different random numbers to Matlab
                for s in seq_order:
                    move_event_from         = int(np.ceil(N * np.random.rand())) - 1

                    current_sequence        = samples_sequence[s, :, i - 1]

                    current_location        = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    #select an event in the sequence to move
                    selected_event          = int(current_sequence[move_event_from])

                    possible_positions      = np.arange(N)

                    distance                = possible_positions - move_event_from

                    if isinstance(seq_sigma, int):  # FIXME: change to float       ##if ((seq_sigma.shape[0]==1) + (seq_sigma.shape[1]==1)) == 2:
                        this_seq_sigma      = seq_sigma
                    else:
                        this_seq_sigma      = seq_sigma[s, selected_event]

                    # use own normal PDF because stats.norm is slow
                    weight                  = AbstractSustain.calc_coeff(this_seq_sigma) * AbstractSustain.calc_exp(distance, 0., this_seq_sigma)
                    weight                  /= np.sum(weight)

                    #TEMP: MATLAB comparison
                    #index                   = 0
                    index                   = np.random.choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                    move_event_to           = possible_positions[index]

                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[s, :, i] = new_sequence

                new_f                       = samples_f[:, i - 1] + f_sigma * np.random.randn()
                # TEMP: MATLAB comparison
                #new_f                       = samples_f[:, i - 1] + f_sigma * stats.norm.ppf(np.random.rand(1,N_S))

                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f
            S                               = samples_sequence[:, :, i]

            #f                               = samples_f[:, i]
            #likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)

            p_perm_k                        = np.zeros((M, N+1, N_S))
            for s in range(N_S):
                p_perm_k[:,:,s]             = self._calculate_likelihood_stage(sustainData, S[s,:])


            #NOTE: added extra axes to get np.tile to work the same as Matlab's repmat in this 3D tiling
            f_val_mat                       = np.tile(samples_f[:,i, np.newaxis, np.newaxis], (1, N+1, M))
            f_val_mat                       = np.transpose(f_val_mat, (2, 1, 0))

            total_prob_stage                = np.sum(p_perm_k * f_val_mat, 2)
            total_prob_subj                 = np.sum(total_prob_stage, 1)

            likelihood_sample               = sum(np.log(total_prob_subj + 1e-250))

            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < np.random.rand():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
        perm_index                          = perm_index[0][0]
        ml_likelihood                       = max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, samples_sequence, samples_f, n_samples, cval=False, plot_order=None):

        temp_mean_f                         = np.mean(samples_f, 1)
        vals                                = np.sort(temp_mean_f)[::-1]
        vals                                = np.array([np.round(x * 100.) for x in vals]) / 100.
        ix                                  = np.argsort(temp_mean_f)[::-1]


        N_S                                 = samples_sequence.shape[0]
        N_bio                               = len(self.biomarker_labels)

        N_stages                            = samples_sequence.shape[1]

        #confus_matrix_plotting              = zeros(size(samples_sequence, 2), size(samples_sequence, 2), size(samples_sequence, 1));
        confus_matrix_plotting              = np.zeros((N_stages, N_stages, N_S))

        if N_S > 1:
            fig, ax                         = plt.subplots(1, N_S)
        else:
            fig, ax                         = plt.subplots()

        if plot_order is None:
            plot_order                      = samples_sequence[ix[0], :, samples_sequence.shape[2]-1].astype(int)
        biomarker_labels_plot_order         = [self.biomarker_labels[i].replace('_', ' ') for i in plot_order]

        for i in range(N_S):
            this_samples_sequence           = np.squeeze(samples_sequence[ix[i], :, :]).T

            N                               = this_samples_sequence.shape[1]

            confus_matrix                   = np.zeros((N, N))
            for j in range(N):
                confus_matrix[j, :]         = sum(this_samples_sequence == j)
            confus_matrix                   /= float(max(this_samples_sequence.shape))

            out_mat_i                       = np.tile(1 - confus_matrix[plot_order,:].reshape(N, N, 1), (1,1,3))

            #this_colour_matrix[:, :, alter_level] = np.tile(this_confus_matrix[markers, :].reshape(N_bio, N, 1), (1, 1, sum(alter_level)))

            TITLE_FONT_SIZE                 = 8
            X_FONT_SIZE                     = 8
            Y_FONT_SIZE                     = 7 #10
            if N_S > 1:
                ax[i].imshow(out_mat_i, interpolation='nearest')      #, cmap=plt.cm.Blues)
                ax[i].set_xticks(np.arange(N))
                ax[i].set_xticklabels(range(1, N+1), fontsize=X_FONT_SIZE) #rotation=45,

                ax[i].set_yticks(np.arange(N_bio))
                ax[i].set_yticklabels([]) #['']* N_bio)
                if i == 0:
                    ax[i].set_yticklabels(np.array(biomarker_labels_plot_order, dtype='object'), ha='right', fontsize=Y_FONT_SIZE)      #rotation=30, ha='right', rotation_mode='anchor'
                    for tick in ax[i].yaxis.get_major_ticks():
                        tick.label.set_color('black')

                #ax[i].set_ylabel('Biomarker name') #, fontsize=20)
                ax[i].set_xlabel('Event position', fontsize=X_FONT_SIZE)
                ax[i].set_title('Group ' + str(i) + ' (f=' + str(vals[i])  + ', n=' + str(int(np.round(vals[i] * n_samples)))  + ')', fontsize=TITLE_FONT_SIZE)

            else: #**** one subtype
                ax.imshow(out_mat_i) #, interpolation='nearest')#, cmap=plt.cm.Blues) #[...,::-1]
                ax.set_xticks(np.arange(N))
                ax.set_xticklabels(range(1, N+1), fontsize=X_FONT_SIZE) #rotation=45,

                ax.set_yticks(np.arange(N_bio))
                ax.set_yticklabels(np.array(biomarker_labels_plot_order, dtype='object'), ha='right', fontsize=Y_FONT_SIZE)           #rotation=30, ha='right', rotation_mode='anchor'

                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_color('black')

                #ax.set_ylabel('Biomarker name') #, fontsize=20)
                ax.set_xlabel('Event position', fontsize=X_FONT_SIZE)
                ax.set_title('Group ' + str(i) + ' (f=' + str(vals[i])  + ', n=' + str(int(np.round(vals[i] * n_samples)))  + ')', fontsize=TITLE_FONT_SIZE)

        plt.tight_layout()
        if cval:
            fig.suptitle('Cross validation')

        return fig, ax

    def subtype_and_stage_individuals_newData(self, L_yes_new, L_no_new, samples_sequence, samples_f, N_samples):

        numStages_new                   = L_yes_new.shape[1]    #number of stages == number of biomarkers here

        assert numStages_new == self.__sustainData.getNumStages(), "Number of stages in new data should be same as in training data"

        sustainData_newData             = MixtureSustainData(L_yes_new, L_no_new, numStages_new)

        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage                   = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, N_samples)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage

    # ********************* STATIC METHODS
    @staticmethod
    def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)

    @staticmethod
    def randperm_local(N):

        # TEMP: MATLAB comparison
        #return np.arange(N)

        return np.random.permutation(N)
