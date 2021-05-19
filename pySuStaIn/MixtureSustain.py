###
# pySuStaIn: Python translation of Matlab version of SuStaIn algorithm (https://www.nature.com/articles/s41467-018-05892-0)
# Authors: Peter Wijeratne (p.wijeratne@ucl.ac.uk) and Leon Aksman (l.aksman@ucl.ac.uk)
# Contributors: Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
#
# For questions/comments related to: object orient implementation of pySustain
# contact: Leon Aksman (l.aksman@ucl.ac.uk)
# For questions/comments related to: the SuStaIn algorithm
# contact: Alex Young (alexandra.young@kcl.ac.uk)
###
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

from pySuStaIn.AbstractSustain import AbstractSustainData
from pySuStaIn.AbstractSustain import AbstractSustain

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
#An implementation of the AbstractSustain class with mixture model based events
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
                 use_parallel_startpoints,
                 seed):
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
        #   seed                        - random number seed

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
                         use_parallel_startpoints,
                         seed)

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

        # Even faster version to avoid loops
        p_perm_k[:, 0] = cp_no[:, -1]
        p_perm_k[:, 1:-1] = cp_no[:, :-1][:, ::-1] * cp_yes[:, :-1]
        p_perm_k[:, -1] = cp_yes[:, -1]
        p_perm_k *= 1 / (N + 1)

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
        # the second summation axis is different to Matlab version
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))
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
        p_perm_k_norm                       = p_perm_k_weighted / np.tile(np.sum(np.sum(p_perm_k_weighted, 1), 1).reshape(M, 1, 1), (1, N + 1, N_S))  # the second summation axis is different to Matlab version
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

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                seq_order                   = MixtureSustain.randperm_local(N_S) #np.random.permutation(N_S)  # this function returns different random numbers to Matlab

                # Abstract out seq_order loop
                move_event_from = np.ceil(N * np.random.rand(len(seq_order))).astype(int) - 1
                current_sequence = samples_sequence[seq_order, :, i - 1]

                selected_event = current_sequence[np.arange(current_sequence.shape[0]), move_event_from]

                possible_positions = np.arange(N) + np.zeros((len(seq_order),1))

                distance = np.arange(N) + np.zeros((len(seq_order),1)) - move_event_from[:, np.newaxis]

                weight = AbstractSustain.calc_coeff(seq_sigma) * AbstractSustain.calc_exp(distance, 0., seq_sigma)
                weight = np.divide(weight, weight.sum(1)[:, None])

                index = [np.random.choice(np.arange(len(row)), 1, replace=True, p=row)[0] for row in weight]

                move_event_to = np.arange(N)[index]

                r = current_sequence.shape[0]
                # Don't need to copy, but doing it for clarity
                new_seq = current_sequence.copy()
                new_seq[np.arange(r), move_event_from] = new_seq[np.arange(r), move_event_to]
                new_seq[np.arange(r), move_event_to] = selected_event

                samples_sequence[seq_order, :, i] = new_seq

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

            likelihood_sample               = np.sum(np.log(total_prob_subj + 1e-250))

            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < np.random.rand():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.where(samples_likelihood == np.max(samples_likelihood))
        perm_index                          = perm_index[0][0]
        ml_likelihood                       = np.max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, samples_sequence, samples_f, n_samples, cval=False, plot_order=None, title_font_size=10):

        temp_mean_f                         = np.mean(samples_f, 1)
        vals                                = np.sort(temp_mean_f)[::-1]
        vals                                = np.array([np.round(x * 100.) for x in vals]) / 100.
        ix                                  = np.argsort(temp_mean_f)[::-1]


        N_S                                 = samples_sequence.shape[0]
        N_bio                               = len(self.biomarker_labels)

        N_stages                            = samples_sequence.shape[1]
        N_MCMC_samples                      = samples_sequence.shape[2]
        
        confus_matrix_plotting              = np.zeros((N_stages, N_stages, N_S))

#         if N_S > 1:
#             fig, ax                         = plt.subplots(1, N_S)
#         else:
#             fig, ax                         = plt.subplots()
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
            
        if plot_order is None:
            plot_order                      = samples_sequence[ix[0], :, samples_sequence.shape[2]-1].astype(int)

        biomarker_labels_plot_order         = [self.biomarker_labels[i].replace('_', ' ') for i in plot_order]

        for i in range(total_axes):        #for i in range(N_S):

            if i not in range(N_S):
                ax.flat[i].set_axis_off()
                continue

            this_samples_sequence           = samples_sequence[ix[i],:,:].T
		        	
            N                               = this_samples_sequence.shape[1]

            confus_matrix                   = np.zeros((N, N))
            for j in range(N):
                confus_matrix[j, :]         = sum(this_samples_sequence == j)
            confus_matrix                   /= N_MCMC_samples #float(max(this_samples_sequence.shape))

            out_mat_i                       = np.tile(1 - confus_matrix[plot_order,:].reshape(N, N, 1), (1,1,3))

            #this_colour_matrix[:, :, alter_level] = np.tile(this_confus_matrix[markers, :].reshape(N_bio, N, 1), (1, 1, sum(alter_level)))

            TITLE_FONT_SIZE                 = title_font_size
            X_FONT_SIZE                     = 10 #8
            Y_FONT_SIZE                     = 10 #7

            if cval == False:                
                if n_samples != np.inf:
                    title_i                 = 'Subtype ' + str(i+1) + ' (f=' + str(vals[i])  + r', n=' + str(int(np.round(vals[i] * n_samples)))  + ')'
                else:
                    title_i                 = 'Subtype ' + str(i+1) + ' (f=' + str(vals[i]) + ')'
            else:
                title_i                     = 'Subtype ' + str(i+1) + ' cross-validated'

            if N_S > 1:
                ax_i                        = ax.flat[i] #ax[i]
                ax_i.imshow(out_mat_i, interpolation='nearest')      #, cmap=plt.cm.Blues)
                ax_i.set_xticks(np.arange(N))
                ax_i.set_xticklabels(range(1, N+1), fontsize=X_FONT_SIZE) #rotation=45,

                ax_i.set_yticks(np.arange(N_bio))
                ax_i.set_yticklabels([]) #['']* N_bio)
                if i == 0:
                    ax_i.set_yticklabels(np.array(biomarker_labels_plot_order, dtype='object'), ha='right', fontsize=Y_FONT_SIZE)      #rotation=30, ha='right', rotation_mode='anchor'
                    for tick in ax_i.yaxis.get_major_ticks():
                        tick.label.set_color('black')

                ax_i.set_xlabel('Event position', fontsize=X_FONT_SIZE)
                ax_i.set_title(title_i, fontsize=TITLE_FONT_SIZE)

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
                ax.set_title(title_i, fontsize=TITLE_FONT_SIZE)
                    
        plt.tight_layout()
        #if cval:
        #    fig.suptitle('Cross validation')

        return fig, ax

    def subtype_and_stage_individuals_newData(self, L_yes_new, L_no_new, samples_sequence, samples_f, N_samples):

        numStages_new                   = L_yes_new.shape[1]    #number of stages == number of biomarkers here

        assert numStages_new == self.__sustainData.getNumStages(), "Number of stages in new data should be same as in training data"

        sustainData_newData             = MixtureSustainData(L_yes_new, L_no_new, numStages_new)

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

    # ********************* TEST METHODS
    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, sustain_kwargs, seed=42, mixture_type="mixture_GMM"):
        # Avoid import outside of testing
        from mixture_model import fit_all_gmm_models, fit_all_kde_models
        # Set a global seed to propagate (particularly for mixture_model)
        np.random.seed(seed)

        ground_truth_sequences = cls.generate_random_model(n_biomarkers, n_subtypes)

        N_stages = n_biomarkers

        ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
        ground_truth_stages = np.vstack(
            (ground_truth_stages_control, ground_truth_stages_other)
        ).astype(int)

        data, data_denoised = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            mixture_type
        )
        # choose which subjects will be cases and which will be controls
        MIN_CASE_STAGE = np.round((n_biomarkers + 1) * 0.8)
        index_case = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
        index_control = np.where(ground_truth_stages ==  0)[0]

        labels = 2 * np.ones(data.shape[0], dtype=int) # 2 - intermediate value, not used in mixture model fitting
        labels[index_case] = 1                         # 1 - cases
        labels[index_control] = 0                      # 0 - controls

        data_case_control = data[labels != 2, :]
        labels_case_control = labels[labels != 2]
        if mixture_type == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels)
        elif mixture_type == "mixture_KDE":
            mixtures = fit_all_kde_models(data, labels)
        
        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)
        for i in range(n_biomarkers):
            if mixture_type == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif mixture_type == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))

        return cls(
            L_yes, L_no,
            **sustain_kwargs
        )

    # TODO: Refactor as Zscore func
    def generate_random_model(N_biomarkers, N_S):
        S                                   = np.zeros((N_S, N_biomarkers))
        #try 30 times to find a unique sequence for each subtype
        for i in range(30): 
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

    # TODO: Refactor as Zscore func
    def generate_data(subtypes, stages, gt_ordering, mixture_style):
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
            stage_i                         = np.asscalar(stages[i])

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
