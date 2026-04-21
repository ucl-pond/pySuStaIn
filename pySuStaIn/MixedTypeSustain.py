
from operator import index

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.AbstractSustain import AbstractSustain, AbstractSustainData
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm

class MixedTypeSustainData(AbstractSustainData):
    """
    Data container for Mixed-SuStaIn (z-score + ordinal + event inputs).
    """

    def __init__(self, zdata, prob_nl, prob_score, numStages):
        """
        Parameters
        ----------
        zdata : ndarray or None
            Z-scored data.
            Shape: (n_subjects, n_zscore_biomarkers).
        prob_nl : ndarray or None
            P(normal/no-event) for ordinal/event biomarkers.
            Shape: (n_subjects, n_ordinal_event_biomarkers).
        prob_score : ndarray or None
            P(score/yes-event) for ordinal/event biomarkers.
            Shape: (n_subjects, n_ordinal_event_biomarkers, n_scores).
        numStages : int
            Total number of stages in the mixed model.
        """
        self.zdata = zdata
        self.prob_nl = prob_nl
        self.prob_score = prob_score
        self.__numStages = numStages
        self.__numSamples = self._validate_and_get_num_samples()


    def _validate_and_get_num_samples(self):
        num_samples = None
        if self.zdata is not None:
            num_samples = self.zdata.shape[0]

        if self.prob_nl is None and self.prob_score is not None:
            raise ValueError("prob_score provided without prob_nl.")
        if self.prob_nl is not None and self.prob_score is None:
            raise ValueError("prob_nl provided without prob_score.")

        if self.prob_nl is not None:
            if num_samples is None:
                num_samples = self.prob_nl.shape[0]
            else:
                assert num_samples == self.prob_nl.shape[0], "zdata and prob_nl must have same number of samples"
        if self.prob_score is not None:
            if num_samples is None:
                num_samples = self.prob_score.shape[0]
            else:
                assert num_samples == self.prob_score.shape[0], "zdata/prob_nl and prob_score must have same number of samples"

        if num_samples is None:
            raise ValueError("MixedTypeSustainData requires zdata or prob_nl/prob_score.")
        return num_samples

    def getNumSamples(self):
        return self.__numSamples
    
    def getNumBiomarkers(self):
        num_biomarkers = 0
        if self.zdata is not None:
            num_biomarkers += self.zdata.shape[1]
        if self.prob_nl is not None:
            num_biomarkers += self.prob_nl.shape[1]
        return num_biomarkers
    
    def getNumStages(self):
        return self.__numStages
    
    def reindex(self, index):
        zdata = self.zdata[index,] if self.zdata is not None else None
        prob_nl = self.prob_nl[index,] if self.prob_nl is not None else None
        prob_score = self.prob_score[index,] if self.prob_score is not None else None
        return MixedTypeSustainData(zdata, prob_nl, prob_score, self.__numStages)
        
class MixedTypeSustain(AbstractSustain):
    """
    MixedTypeSuStaIn combines the logic of Z-Score SuStaIn and ordinal SuStaIn to jointly
    model three biomarker types: z-score, ordinal, and event.

    Event biomarkers are modeled in ordinal SuStaIn as ordinal biomarkers with exactly one
    discrete step (`score_vals=1`). Ordinal biomarkers use discrete score steps; the model
    combines both by building one shared discrete block, where `prob_nl` is the no-event/normal
    mixture (`L_no`) and `prob_score` provides the step/yes-event mixture (`L_yes`).
    """

    @staticmethod
    def _combine_ordinal_event_inputs(ordinal_prob_nl, ordinal_prob_score, ordinal_score_vals, event_prob_yes, event_prob_no):
        """Merge split ordinal/event blocks into the internal combined representation."""
        num_ordinal_biomarkers = 0 if ordinal_prob_nl is None else ordinal_prob_nl.shape[1]
        num_event_biomarkers = 0 if event_prob_yes is None else event_prob_yes.shape[1]

        if ordinal_score_vals is None:
            if num_ordinal_biomarkers > 0 and ordinal_prob_score is not None:
                n_scores = ordinal_prob_score.shape[2]
                ordinal_score_vals = np.tile(np.arange(1, n_scores + 1), (num_ordinal_biomarkers, 1))
            else:
                ordinal_score_vals = np.zeros((0, 0), dtype=int)
        else:
            ordinal_score_vals = np.asarray(ordinal_score_vals)

        if num_ordinal_biomarkers > 0 and num_event_biomarkers > 0:
            max_scores = max(ordinal_score_vals.shape[1], 1)
            combined_score_vals = np.zeros(
                (num_ordinal_biomarkers + num_event_biomarkers, max_scores),
                dtype=ordinal_score_vals.dtype
            )
            combined_score_vals[:num_ordinal_biomarkers, :ordinal_score_vals.shape[1]] = ordinal_score_vals
            combined_score_vals[num_ordinal_biomarkers:, 0] = 1

            combined_prob_score = np.zeros(
                (ordinal_prob_score.shape[0], num_ordinal_biomarkers + num_event_biomarkers, max_scores),
                dtype=float
            )
            combined_prob_score[:, :num_ordinal_biomarkers, :ordinal_prob_score.shape[2]] = ordinal_prob_score
            combined_prob_score[:, num_ordinal_biomarkers:, 0] = event_prob_yes

            combined_prob_nl = np.hstack((ordinal_prob_nl, event_prob_no))
        elif num_ordinal_biomarkers > 0:
            combined_score_vals = ordinal_score_vals
            combined_prob_score = ordinal_prob_score
            combined_prob_nl = ordinal_prob_nl
        elif num_event_biomarkers > 0:
            combined_score_vals = np.ones((num_event_biomarkers, 1), dtype=int)
            combined_prob_score = event_prob_yes[:, :, None]
            combined_prob_nl = event_prob_no
        else:
            combined_score_vals = np.zeros((0, 0), dtype=int)
            combined_prob_score = None
            combined_prob_nl = None

        return combined_prob_nl, combined_prob_score, combined_score_vals

    @staticmethod
    def _build_stage_index(score_vals):
        """Flatten score matrix into active event scores and their biomarker indices."""
        score_vals = np.asarray(score_vals)
        if score_vals.ndim != 2:
            raise ValueError("score_vals must be a 2D array")
        if score_vals.size == 0 or score_vals.shape[1] == 0:
            return np.array([], dtype=float), np.array([], dtype=int)

        stage_score = score_vals.T.flatten()
        ix_select = stage_score > 0
        stage_score = stage_score[ix_select]
        stage_biomarker_index = np.tile(np.arange(score_vals.shape[0]), (score_vals.shape[1],))[ix_select]
        return stage_score, stage_biomarker_index

    def __init__(
            self, 
            zscore_data, z_vals, z_max, zscore_biomarker_labels, # zscore parameters
            ordinal_prob_nl, ordinal_prob_score, ordinal_score_vals, ordinal_biomarker_labels, # ordinal parameters
            event_prob_yes, event_prob_no, event_biomarker_labels, # event parameters
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed=None
            ):
        """
        The initializer for the mixed (z-score + ordinal + event) implementation of AbstractSustain.

        Parameters
        ----------
        zscore_data : ndarray
            Z-scored data (positive z-scores).
            Shape: (n_subjects, n_zscore_biomarkers).
        z_vals : ndarray
            Z-score thresholds for each z-score biomarker.
            Shape: (n_zscore_biomarkers, n_zscore_thresholds).
        z_max : ndarray
            Maximum z-score for each z-score biomarker.
            Shape: (n_zscore_biomarkers, 1).
        zscore_biomarker_labels : list[str]
            Names of z-score biomarkers.

        ordinal_prob_nl : ndarray or None
            P(normal) for ordinal biomarkers.
            Shape: (n_subjects, n_ordinal_biomarkers).
        ordinal_prob_score : ndarray or None
            P(score level) for ordinal biomarkers.
            Shape: (n_subjects, n_ordinal_biomarkers, n_scores).
        ordinal_score_vals : ndarray or None
            Discrete score levels per ordinal biomarker.
            Shape: (n_ordinal_biomarkers, n_scores).
        ordinal_biomarker_labels : list[str]
            Names of ordinal biomarkers.

        event_prob_yes : ndarray or None
            P(event/abnormal) for event biomarkers.
            Shape: (n_subjects, n_event_biomarkers).
        event_prob_no : ndarray or None
            P(no-event/normal) for event biomarkers.
            Shape: (n_subjects, n_event_biomarkers).
        event_biomarker_labels : list[str]
            Names of event biomarkers.

        N_startpoints : int
            Number of startpoints for the maximum likelihood step, typically 25.
        N_S_max : int
            Maximum number of subtypes, should be 1 or more.
        N_iterations_MCMC : int
            Number of MCMC iterations, typically 1e5 or 1e6 (lower for debugging).
        output_folder : str
            Where to save pickle files, etc.
        dataset_name : str
            Name used for output file naming.
        use_parallel_startpoints : bool
            Whether to parallelize the maximum likelihood loop.
        seed : int or None
            Random number seed.
        """
        # ----- z-score inputs
        if zscore_data is not None and not np.all(zscore_data == 0):
            num_zscore_biomarkers = zscore_data.shape[1]
        else:
            num_zscore_biomarkers = 0
            z_vals = []

        # ----- ordinal inputs
        if ordinal_prob_nl is not None:
            num_ordinal_biomarkers = ordinal_prob_nl.shape[1]
            assert ordinal_prob_score is not None and ordinal_score_vals is not None, \
                "ordinal_prob_nl requires ordinal_prob_score and ordinal_score_vals"
            assert ordinal_prob_score.ndim == 3, "ordinal_prob_score must have shape (subjects, biomarkers, scores)"
            assert ordinal_score_vals.ndim == 2, "ordinal_score_vals must have shape (biomarkers, scores)"
            assert ordinal_prob_score.shape[1] == num_ordinal_biomarkers, \
                "ordinal_prob_score biomarkers must match ordinal_prob_nl"
            assert ordinal_score_vals.shape[0] == num_ordinal_biomarkers, \
                "ordinal_score_vals biomarkers must match ordinal_prob_nl"
        else:
            num_ordinal_biomarkers = 0
            ordinal_prob_score = None
            ordinal_score_vals = np.zeros((0, 0), dtype=int)

        # ----- event inputs
        if event_prob_yes is not None or event_prob_no is not None:
            assert event_prob_yes is not None and event_prob_no is not None, \
                "event_prob_yes and event_prob_no must both be provided"
            assert event_prob_yes.shape == event_prob_no.shape, \
                "event_prob_yes and event_prob_no must have the same shape"
            num_event_biomarkers = event_prob_yes.shape[1]
        else:
            num_event_biomarkers = 0

        assert len(zscore_biomarker_labels) == num_zscore_biomarkers, "number of zscore biomarkers does not match with biomarker labels"
        assert len(ordinal_biomarker_labels) == num_ordinal_biomarkers, "number of ordinal biomarkers does not match with biomarker labels"
        assert len(event_biomarker_labels) == num_event_biomarkers, "number of event biomarkers does not match with biomarker labels"

        # ----- validate subject counts
        subject_counts = []
        if zscore_data is not None:
            subject_counts.append(zscore_data.shape[0])
        if ordinal_prob_nl is not None:
            subject_counts.append(ordinal_prob_nl.shape[0])
            subject_counts.append(ordinal_prob_score.shape[0])
        if num_event_biomarkers > 0:
            subject_counts.append(event_prob_no.shape[0])
            subject_counts.append(event_prob_yes.shape[0])

        if len(subject_counts) == 0:
            raise ValueError("MixedTypeSustain requires zscore and/or ordinal/event data")
        assert len(set(subject_counts)) == 1, "number of subjects does not match across provided data blocks"
        num_subjects = subject_counts[0]

        # ----- merge ordinal + event into one ordinal/event block
        # order is fixed: [ordinal biomarkers..., event biomarkers...]
        combined_prob_nl, combined_prob_score, combined_score_vals = self._combine_ordinal_event_inputs(
            ordinal_prob_nl, ordinal_prob_score, ordinal_score_vals, event_prob_yes, event_prob_no
        )

        ordinal_event_biomarker_labels = ordinal_biomarker_labels + event_biomarker_labels
        num_ordinal_event_biomarkers = len(ordinal_event_biomarker_labels)
        num_biomarkers = num_zscore_biomarkers + num_ordinal_event_biomarkers
        max_score_value = int(max(
            combined_score_vals.shape[1] if num_ordinal_event_biomarkers > 0 else 0,
            z_vals.shape[1] if num_zscore_biomarkers > 0 else 0
        ))
        
        bool_zscore_biomarkers = [True]*num_zscore_biomarkers + [False]*num_ordinal_event_biomarkers
        bool_zscore_biomarkers = np.array(bool_zscore_biomarkers)

        zscore_indices = iter(z_vals)
        ordinal_event_indices = iter(combined_score_vals)

        mixed_data_vals = np.zeros((num_biomarkers, max_score_value))
        for i in range(num_biomarkers):
            if bool_zscore_biomarkers[i]: # biomarker is zscore
                row = next(zscore_indices)
                mixed_data_vals[i, :len(row)] = row
            else: # biomarker is ordinal/event
                row = next(ordinal_event_indices)
                mixed_data_vals[i, :len(row)] = row

        stage_score, stage_biomarker_index = self._build_stage_index(mixed_data_vals)
        stage_score = stage_score.reshape(1, len(stage_score))
        stage_biomarker_index = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

        # initialise parameters
        # - zscore
        self.zscore_data = zscore_data
        self.z_vals = z_vals
        self.zscore_biomarker_labels = zscore_biomarker_labels
        self.min_biomarker_zscore = [0] * num_zscore_biomarkers
        self.max_biomarker_zscore = z_max
        # - ordinal/event merged
        self.prob_nl = combined_prob_nl
        self.prob_score = combined_prob_score
        self.score_vals = combined_score_vals
        self.ordinal_biomarker_labels = ordinal_event_biomarker_labels
        self.ordinal_event_biomarker_labels = ordinal_event_biomarker_labels
        # - keep original split inputs available
        self.ordinal_prob_nl = ordinal_prob_nl
        self.ordinal_prob_score = ordinal_prob_score
        self.ordinal_score_vals = ordinal_score_vals
        self.event_prob_yes = event_prob_yes
        self.event_prob_no = event_prob_no
        self.event_biomarker_labels = event_biomarker_labels
        # - data combined
        self.mixed_data_vals = mixed_data_vals
        self.stage_score = stage_score
        self.stage_biomarker_index = stage_biomarker_index
        self.bool_zscore_biomarkers = bool_zscore_biomarkers
        self.num_biomarkers = num_biomarkers
        self.num_stages = stage_biomarker_index.shape[1]
        self.num_subjects = num_subjects
        self.biomarker_labels = zscore_biomarker_labels + ordinal_event_biomarker_labels
        # - model
        self.N_startpoints = N_startpoints
        self.N_S_max = N_S_max
        self.N_iterations_MCMC = N_iterations_MCMC
        # - general 
        self.output_folder = output_folder
        self.dataset_name = dataset_name
        self.use_parallel_startpoints = use_parallel_startpoints

        # MixedTypeSustain keeps merged ordinal/event prob_score in 3D (M, B, n_scores).
        # Event/mixture biomarkers correspond to n_scores=1.
        self.__sustainData = MixedTypeSustainData(zscore_data, combined_prob_nl, combined_prob_score, self.num_stages)

        # initialise abstract sustain
        super().__init__(self.__sustainData, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints, seed)

        print("Init done for MixedTypeSustain")

    def _initialise_sequence(self, sustainData, rng): # done :)
        """
        Randomly initialize a valid sequence with monotonic biomarker progression.

        Output
        ------
        S : ndarray
            Random mixed-model sequence with per-biomarker monotonicity.
            Shape: (1, n_stages).
        """
        N = self.num_stages
        S = np.zeros(N)

        for i in range(N):

            IS_min_stage_score = np.array([False] * N)
            possible_biomarkers = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):

                IS_unselected = [False] * N

                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k] = True
                    
                this_biomarkers = np.array(
                    [(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) +
                     (np.array(IS_unselected) == 1).astype(int)]) == 2
                
                if not np.any(this_biomarkers):
                    this_min_stage_score = 0
                else:
                    this_min_stage_score = min(self.stage_score[this_biomarkers])
                if (this_min_stage_score):
                    temp = ((this_biomarkers.astype(int) + (self.stage_score == this_min_stage_score).astype(int)) == 2).T
                    temp = temp.reshape(len(temp), )
                    IS_min_stage_score[temp] = True

            events = np.array(range(N))
            possible_events = np.array(events[IS_min_stage_score])
            this_index = np.ceil(rng.random() * ((len(possible_events)))) - 1
            S[i] = possible_events[int(this_index)]

        S = S.reshape(1, len(S))

        return S

    def _calculate_likelihood_stage(self, sustainData, S):
        """
        Compute p(data | stage, subtype-sequence S) for one subtype sequence.

        p_perm_k is computed by factorizing across biomarkers: first build a per-biomarker
        likelihood tensor p_perm_k_biomarkers with shape (n_subjects, n_stages + 1, n_biomarkers),
        using Gaussian likelihoods for z-score biomarkers and prob_nl/prob_score for
        ordinal/event biomarkers, then multiply across biomarkers (np.prod(..., axis=2))
        and apply a uniform stage prior.

        Output
        ------
        p_perm_k : ndarray
            Stage likelihoods for each subject.
            Shape: (n_subjects, n_stages + 1).
        """
        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        ordinal_event_mask = ~self.bool_zscore_biomarkers
        n_ordinal_event = int(np.sum(ordinal_event_mask))

        if sustainData.prob_nl is not None:
            prob_nl = sustainData.prob_nl.copy()
            prob_score = sustainData.prob_score.copy()
        else:
            prob_nl = np.empty((M, n_ordinal_event))
            prob_score = np.empty((M, n_ordinal_event, 0))

        p_perm_k_biomarkers = np.zeros((M, N + 1, self.num_biomarkers))

        S = np.asarray(S).reshape(-1)

        S_inv = np.zeros(self.num_stages, dtype=int)
        S_inv[S.astype(int)] = np.arange(self.num_stages)

        point_value = np.zeros((self.num_biomarkers, self.num_stages + 2))
        arange_N = np.arange(self.num_stages + 2)
        possible_biomarkers = np.unique(self.stage_biomarker_index)

        idx_zscore = 0
        for i in range(self.num_biomarkers):
            b = possible_biomarkers[i]
            if self.bool_zscore_biomarkers[b]:
                event_location = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [self.num_stages]])
                event_value = np.concatenate([[self.min_biomarker_zscore[idx_zscore]], self.stage_score[self.stage_biomarker_index == b], [self.max_biomarker_zscore[idx_zscore]]])

                for j in range(len(event_location) - 1):
                    if j == 0:
                        temp = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                        N_j = event_location[j + 1] - event_location[j] + 2
                        point_value[b, temp] = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])
                    else:
                        temp = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                        N_j = event_location[j + 1] - event_location[j] + 1
                        point_value[b, temp] = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

                idx_zscore += 1

        stage_value_zscore = (0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:])[self.bool_zscore_biomarkers]
        if sustainData.zdata is not None:
            zscored_data = np.array(sustainData.zdata[:, :, None], dtype=np.float64)
            x = zscored_data - stage_value_zscore
            x = np.transpose(x, (0, 2, 1))
            p_perm_k_biomarkers[:, :, self.bool_zscore_biomarkers] = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

        if n_ordinal_event > 0:
            stage_value_ordinal_event = np.zeros((self.num_stages + 1, self.num_biomarkers))
            for stage in range(self.num_stages):
                index_justreached = int(S[stage])
                biomarker_justreached = int(self.stage_biomarker_index[0][index_justreached])
                stage_value_justreached = int(self.stage_score[0][index_justreached])
                if not self.bool_zscore_biomarkers[biomarker_justreached]:
                    index = stage + 1
                    stage_value_ordinal_event[index:, biomarker_justreached] = stage_value_justreached
            stage_value_ordinal_event = stage_value_ordinal_event[:, ordinal_event_mask]
            p_perm_k_biomarkers[:, 0, ordinal_event_mask] = prob_nl

            for stage in range(N):
                probability_stage = prob_nl.copy()
                stage_value_justreached = stage_value_ordinal_event[stage + 1]

                for i, value in enumerate(stage_value_justreached):
                    if value > 0:
                        idx_value = int(value) - 1
                        probability_stage[:, i] = prob_score[:, i, idx_value]

                p_perm_k_biomarkers[:, stage + 1, ordinal_event_mask] = probability_stage

        coeff = 1. / float(N + 1)
        p_perm_k = np.prod(p_perm_k_biomarkers, 2)
        p_perm_k = coeff * p_perm_k

        return p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        """ Optimise the parameters of the SuStaIn model."""
        N_S = S_init.shape[0]
        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        
        S_opt = S_init.copy()
        f_opt = np.asarray(f_init).reshape(1, 1, N_S)
        p_perm_k = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted = p_perm_k * f_opt
        p_perm_k_norm = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
        f_opt = (np.sum(p_perm_k_norm, axis=(0, 1)) / np.sum(p_perm_k_norm)).reshape(1, 1, N_S)

        order_seq = rng.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio = rng.permutation(N)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence = S_opt[s]
                current_location = np.argsort(current_sequence.astype(int))

                selected_event = i

                move_event_from = current_location[selected_event]

                this_stage_zscore = self.stage_score[0, selected_event]
                selected_biomarker = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter = possible_zscores_biomarker < this_stage_zscore
                max_filter = possible_zscores_biomarker > this_stage_zscore
                events = np.arange(self.num_stages)
                if np.any(min_filter):
                    min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event = events[np.logical_and(self.stage_score[0] == min_zscore_bound, self.stage_biomarker_index[0] == selected_biomarker)]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event = events[np.logical_and(self.stage_score[0] == max_zscore_bound, self.stage_biomarker_index[0] == selected_biomarker)]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = N

                lower_bound = int(np.asarray(move_event_to_lower_bound).reshape(-1)[0])
                upper_bound = int(np.asarray(move_event_to_upper_bound).reshape(-1)[0])
                if lower_bound == upper_bound:
                    # Single valid insertion point (Matlab-style range(N,N) behavior).
                    possible_positions = np.array([lower_bound], dtype=int)
                else:
                    possible_positions = np.arange(lower_bound, upper_bound)
                possible_sequences = np.zeros((len(possible_positions), N))
                possible_likelihood = np.zeros((len(possible_positions), 1))
                possible_p_perm_k = np.zeros((M, N + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to = possible_positions[index]

                    # move this event in its new position
                    current_sequence = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                    new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                    total_prob_stage = np.sum(p_perm_k * f_opt, 2)
                    total_prob_subj = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood = max(possible_likelihood)
                this_S = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S = this_S[0, :]
                S_opt[s] = this_S
                this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]

            S_opt[s] = this_S

        p_perm_k_weighted = p_perm_k * f_opt
        p_perm_k_norm = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt = (np.sum(p_perm_k_norm, axis=(0, 1)) / np.sum(p_perm_k_norm)).reshape(1, 1, N_S)

        f_opt = f_opt.reshape(N_S)
        total_prob_stage = np.sum(p_perm_k * f_opt, 2)
        total_prob_subj = np.sum(total_prob_stage, 1)

        likelihood_opt = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        """Take MCMC samples of the uncertainty in the SuStaIn model parameters"""
        N = sustainData.getNumStages()
        N_S = seq_init.shape[0]

        if isinstance(f_sigma, float):
            f_sigma = np.array([f_sigma])

        samples_sequence = np.zeros((N_S, N, n_iterations))
        samples_f = np.zeros((N_S, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0] = seq_init
        samples_f[:, 0] = f_init

        tqdm_update_iters = int(n_iterations / 1000) if n_iterations > 100000 else None

        for iter_idx in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if iter_idx > 0:
                subtype_order = self.global_rng.permutation(N_S)
                for subtype_idx in subtype_order:
                    move_event_from = int(np.ceil(N * self.global_rng.random())) - 1
                    current_sequence = samples_sequence[subtype_idx, :, iter_idx - 1]

                    current_location = np.argsort(current_sequence.astype(int))

                    selected_event = int(current_sequence[move_event_from])
                    this_stage_score = self.stage_score[0, selected_event]
                    selected_biomarker = self.stage_biomarker_index[0, selected_event]
                    possible_scores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                    # Keep each biomarker's events monotonic when proposing moves.
                    min_filter = possible_scores_biomarker < this_stage_score
                    max_filter = possible_scores_biomarker > this_stage_score
                    events = np.arange(N)

                    if np.any(min_filter):
                        min_score_bound = max(possible_scores_biomarker[min_filter])
                        min_score_bound_event = events[np.logical_and(self.stage_score[0] == min_score_bound, self.stage_biomarker_index[0] == selected_biomarker)]
                        move_event_to_lower_bound = current_location[min_score_bound_event] + 1
                    else:
                        move_event_to_lower_bound = 0

                    if np.any(max_filter):
                        max_score_bound = min(possible_scores_biomarker[max_filter])
                        max_score_bound_event = events[np.logical_and(self.stage_score[0] == max_score_bound, self.stage_biomarker_index[0] == selected_biomarker)]
                        move_event_to_upper_bound = current_location[max_score_bound_event]
                    else:
                        move_event_to_upper_bound = N

                    if move_event_to_lower_bound == move_event_to_upper_bound:
                        possible_positions = np.array([0])
                    else:
                        possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                    distance = possible_positions - move_event_from

                    if isinstance(seq_sigma, int):
                        this_seq_sigma = seq_sigma
                    else:
                        this_seq_sigma = seq_sigma[subtype_idx, selected_event]

                    weight = AbstractSustain.calc_coeff(this_seq_sigma) * AbstractSustain.calc_exp(distance, 0.0, this_seq_sigma)
                    weight /= np.sum(weight)
                    index = self.global_rng.choice(range(len(possible_positions)), 1, replace=True, p=weight)
                    move_event_to = possible_positions[index]

                    current_sequence = np.delete(current_sequence, move_event_from, 0)
                    new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[subtype_idx, :, iter_idx] = new_sequence

                new_f = samples_f[:, iter_idx - 1] + f_sigma * self.global_rng.standard_normal()
                new_f = np.fabs(new_f) / np.sum(np.fabs(new_f))
                samples_f[:, iter_idx] = new_f

            S = samples_sequence[:, :, iter_idx]
            f = samples_f[:, iter_idx]
            likelihood_sample, _, _, _, _ = self._calculate_likelihood(sustainData, S, f)
            samples_likelihood[iter_idx] = likelihood_sample

            if iter_idx > 0:
                log_ratio = samples_likelihood[iter_idx] - samples_likelihood[iter_idx - 1]
                if log_ratio < np.log(self.global_rng.random()):
                    samples_likelihood[iter_idx] = samples_likelihood[iter_idx - 1]
                    samples_sequence[:, :, iter_idx] = samples_sequence[:, :, iter_idx - 1]
                    samples_f[:, iter_idx] = samples_f[:, iter_idx - 1]

        perm_index = np.argmax(samples_likelihood)
        ml_likelihood = np.max(samples_likelihood)
        ml_sequence = samples_sequence[:, :, perm_index]
        ml_f = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
    
    def _plot_sustain_model(self, *args, **kwargs):
        """Plot mixed SuStaIn model outputs (delegates to plotting utilities)."""
        return MixedTypeSustain.plot_positional_var(*args, score_vals=self.mixed_data_vals, **kwargs)

    def subtype_and_stage_individuals_newData(self, zscore_data_new, ordinal_prob_nl_new, ordinal_prob_score_new, event_prob_yes_new, event_prob_no_new, samples_sequence, samples_f, N_samples):
        """Subtype/stage new subjects from split mixed inputs (z-score, ordinal, event)."""
        num_stages_new = self.__sustainData.getNumStages()
        num_zscore_biomarkers = int(np.sum(self.bool_zscore_biomarkers))
        num_ordinal_biomarkers = len(self.ordinal_biomarker_labels) - len(self.event_biomarker_labels)
        num_event_biomarkers = len(self.event_biomarker_labels)

        if num_zscore_biomarkers > 0:
            assert zscore_data_new is not None, "zscore_data_new is required for this trained mixed model"
            assert zscore_data_new.shape[1] == num_zscore_biomarkers, \
                "Number of z-score biomarkers in new data should match training data"
        elif zscore_data_new is not None:
            assert zscore_data_new.shape[1] == 0, \
                "This trained mixed model has no z-score biomarkers"

        subject_counts = []
        if zscore_data_new is not None:
            subject_counts.append(zscore_data_new.shape[0])

        if num_ordinal_biomarkers > 0:
            assert ordinal_prob_nl_new is not None and ordinal_prob_score_new is not None, \
                "ordinal_prob_nl_new and ordinal_prob_score_new are required for this trained mixed model"
            assert ordinal_prob_nl_new.shape[1] == num_ordinal_biomarkers, \
                "Number of ordinal biomarkers in new data should match training data"
            assert ordinal_prob_score_new.ndim == 3, \
                "ordinal_prob_score_new must have shape (subjects, biomarkers, scores)"
            assert ordinal_prob_score_new.shape[1] == num_ordinal_biomarkers, \
                "Number of ordinal biomarkers in ordinal_prob_score_new should match training data"
            subject_counts.extend([ordinal_prob_nl_new.shape[0], ordinal_prob_score_new.shape[0]])
        else:
            assert ordinal_prob_nl_new is None and ordinal_prob_score_new is None, \
                "This trained mixed model has no ordinal biomarkers"

        if num_event_biomarkers > 0:
            assert event_prob_yes_new is not None and event_prob_no_new is not None, \
                "event_prob_yes_new and event_prob_no_new are required for this trained mixed model"
            assert event_prob_yes_new.shape[1] == num_event_biomarkers, \
                "Number of event biomarkers in new data should match training data"
            assert event_prob_yes_new.shape == event_prob_no_new.shape, \
                "event_prob_yes_new and event_prob_no_new must have identical shape"
            subject_counts.extend([event_prob_yes_new.shape[0], event_prob_no_new.shape[0]])
        else:
            assert event_prob_yes_new is None and event_prob_no_new is None, \
                "This trained mixed model has no event biomarkers"

        assert len(subject_counts) > 0, "At least one modality must be provided for new data"
        assert len(set(subject_counts)) == 1, "New data modalities must have the same number of subjects"

        prob_nl_new, prob_score_new, _ = self._combine_ordinal_event_inputs(
            ordinal_prob_nl_new, ordinal_prob_score_new, self.ordinal_score_vals, event_prob_yes_new, event_prob_no_new
        )

        sustainData_newData = MixedTypeSustainData(zscore_data_new, prob_nl_new, prob_score_new, num_stages_new)

        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage  = self.subtype_and_stage_individuals(
            sustainData_newData, samples_sequence, samples_f, N_samples
        )

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage

    @staticmethod
    def plot_positional_var(samples_sequence, samples_f, n_samples, score_vals, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="original", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        """Plot positional variance diagrams for mixed score matrices."""
        N_S = samples_sequence.shape[0]
        N_bio = score_vals.shape[0]

        if biomarker_labels is not None:
            assert len(biomarker_labels) == N_bio

        if subtype_order is None:
            if ml_f_EM is not None:
                subtype_order = np.argsort(ml_f_EM)[::-1]
            else:
                subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
        elif isinstance(subtype_order, tuple):
            subtype_order = list(subtype_order)

        stage_score = score_vals.T.flatten()
        IX_select = np.nonzero(stage_score)[0]
        stage_score = stage_score[IX_select][None, :]
        num_scores = np.unique(stage_score)
        N_z = len(num_scores)
        stage_biomarker_index = np.tile(np.arange(N_bio), (N_z,))
        stage_biomarker_index = stage_biomarker_index[IX_select]

        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        if biomarker_order is not None:
            if len(biomarker_order) > N_bio:
                biomarker_order = np.arange(N_bio)
        else:
            biomarker_order = np.arange(N_bio)

        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
        else:
            biomarker_labels = [biomarker_labels[i] for i in biomarker_order]

        if subtype_titles is not None:
            assert len(subtype_titles) == N_S

        if cmap == "original":
            colour_mat = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 1], [0, 1, 1], [0, 1, 0.5]])[:N_z]
            if colour_mat.shape[0] > N_z:
                raise ValueError(f"Colours are only defined for {len(colour_mat)} z-scores!")
        else:
            raise NotImplementedError

        if biomarker_colours is not None:
            biomarker_colours = AbstractSustain.check_biomarker_colours(
            biomarker_colours, biomarker_labels
        )
        else:
            biomarker_colours = {i:"black" for i in biomarker_labels}

        if separate_subtypes:
            nrows, ncols = 1, 1
            if figsize is None:
                fig_width = max(12, stage_score.shape[1] * 1.5)
                fig_height = max(6, N_bio * 0.3 + 2)
                figsize = (fig_width, fig_height)
        else:
            nrows, ncols = N_S, 1
            if figsize is None:
                fig_width = max(16, stage_score.shape[1] * 1.8)
                fig_height = max(8, N_S * (N_bio * 0.25 + 3))
                figsize = (fig_width, fig_height)

        total_axes = nrows * ncols
        subtype_loops = N_S if separate_subtypes else 1
        figs = []
        mean_f = np.mean(samples_f, 1)[subtype_order]
        score_masks = [(stage_score == score)[0] for score in num_scores]

        for loop_idx in range(subtype_loops):
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            figs.append(fig)
            axs_flat = np.atleast_1d(axs).flat

            for axis_idx in range(total_axes):
                subtype_idx = loop_idx if separate_subtypes else axis_idx
                ax = next(axs_flat)

                if subtype_idx not in range(N_S):
                    ax.set_axis_off()
                    continue

                this_samples_sequence = samples_sequence[subtype_order[subtype_idx], :, :].T
                N = this_samples_sequence.shape[1]
                confus_matrix = (this_samples_sequence == np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]
                confus_matrix_c = np.ones((N_bio, N, 3))

                for j, score_mask in enumerate(score_masks):
                    alter_level = colour_mat[j] == 0
                    confus_matrix_score = confus_matrix[score_mask]
                    confus_matrix_c[
                        np.ix_(
                            stage_biomarker_index[score_mask], range(N),
                            alter_level
                        )
                    ] -= np.tile(
                        confus_matrix_score.reshape(score_mask.sum(), N, 1),
                        (1, 1, alter_level.sum())
                    )

                if subtype_titles is not None:
                    title_i = subtype_titles[subtype_idx]
                else:
                    if cval == False:
                        if n_samples != np.inf:
                            title_i = f"Subtype {subtype_idx+1} (f={mean_f[subtype_idx]:.2f}, n={np.round(mean_f[subtype_idx] * n_samples):n})"
                        else:
                            title_i = f"Subtype {subtype_idx+1} (f={mean_f[subtype_idx]:.2f})"
                    else:
                        title_i = f"Subtype {subtype_idx+1} cross-validated"

                ax.imshow(
                    confus_matrix_c[biomarker_order, :, :],
                    interpolation='nearest'
                )
                stage_ticks = np.arange(0, N, stage_interval)
                ax.set_xticks(stage_ticks)
                ax.set_xticklabels(stage_ticks + 1, fontsize=stage_font_size, rotation=stage_rot)
                ax.set_yticks(np.arange(N_bio))

                if (subtype_idx % ncols) == 0:
                    ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size, rotation=label_rot)
                    for tick_label in ax.get_yticklabels():
                        tick_label.set_color(biomarker_colours[tick_label.get_text()])
                else:
                    ax.set_yticklabels([])

                ax.set_xlabel(stage_label, fontsize=stage_font_size + 2)
                ax.set_title(title_i, fontsize=title_font_size)

            fig.tight_layout()
            if save_path is not None:
                if separate_subtypes:
                    save_name = f"{save_path}_subtype{loop_idx}"
                else:
                    save_name = f"{save_path}_all-subtypes"

                current_save_kwargs = dict(save_kwargs)
                file_format = current_save_kwargs.pop("format", "png")
                fig.savefig(
                    f"{save_name}.{file_format}",
                    **current_save_kwargs
                )
        return figs, axs

    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, sustain_kwargs, seed=42):
        """Build a synthetic mixed dataset and return an initialized MixedTypeSustain model."""
        rng = np.random.default_rng(seed)

        n_biomarkers = int(n_biomarkers)
        n_samples = int(n_samples)
        n_subtypes = int(n_subtypes)
        ground_truth_subtypes = np.asarray(ground_truth_subtypes, dtype=int).reshape(-1)
        if n_biomarkers < 1:
            raise ValueError("n_biomarkers must be >= 1")
        if len(ground_truth_subtypes) != n_samples:
            raise ValueError("ground_truth_subtypes length must match n_samples")

        # Split biomarkers into near-equal z-score / ordinal / event groups,
        # assigning leftovers with priority: z-score, then ordinal, then event.
        base = n_biomarkers // 3
        remainder = n_biomarkers % 3
        n_zscore_biomarkers = base
        n_ordinal_biomarkers = base
        n_event_biomarkers = base
        if remainder >= 1:
            n_zscore_biomarkers += 1
        if remainder >= 2:
            n_ordinal_biomarkers += 1

        z_vals = np.tile(np.arange(1, 4), (n_zscore_biomarkers, 1)) if n_zscore_biomarkers > 0 else np.zeros((0, 0), dtype=int)
        z_max = np.full((n_zscore_biomarkers,), 5) if n_zscore_biomarkers > 0 else np.array([])

        ordinal_score_vals = np.tile(np.arange(1, 4), (n_ordinal_biomarkers, 1)) if n_ordinal_biomarkers > 0 else np.zeros((0, 0), dtype=int)

        ground_truth_sequences = cls.generate_random_model(
            z_vals, ordinal_score_vals, n_event_biomarkers, n_subtypes, seed=seed
        )
        n_stages = int(np.sum(z_vals > 0) + np.sum(ordinal_score_vals > 0) + n_event_biomarkers)

        n_controls = int(np.round(n_samples * 0.25))
        n_cases = n_samples - n_controls
        ground_truth_stages = np.vstack((
            np.zeros((n_controls, 1), dtype=int),
            rng.integers(1, n_stages + 1, size=(n_cases, 1), endpoint=False, dtype=int)
        ))

        zscore_data, _, _, ordinal_prob_nl, ordinal_prob_score, event_prob_yes, event_prob_no = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            z_vals,
            z_max,
            ordinal_score_vals,
            n_event_biomarkers,
            prob_correct_ordinal=0.9,
            mixture_style="mixture_GMM",
            seed=seed
        )

        model_kwargs = dict(sustain_kwargs)
        biomarker_labels = model_kwargs.pop("biomarker_labels", None)
        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(n_biomarkers)]
        else:
            biomarker_labels = list(biomarker_labels)
            if len(biomarker_labels) != n_biomarkers:
                raise ValueError("biomarker_labels length in sustain_kwargs must match n_biomarkers")

        zscore_biomarker_labels = biomarker_labels[:n_zscore_biomarkers]
        ordinal_biomarker_labels = biomarker_labels[n_zscore_biomarkers:n_zscore_biomarkers + n_ordinal_biomarkers]
        event_biomarker_labels = biomarker_labels[n_zscore_biomarkers + n_ordinal_biomarkers:]

        return cls(
            zscore_data,
            z_vals,
            z_max,
            zscore_biomarker_labels,
            ordinal_prob_nl,
            ordinal_prob_score,
            ordinal_score_vals,
            ordinal_biomarker_labels,
            event_prob_yes,
            event_prob_no,
            event_biomarker_labels,
            **model_kwargs
        )

    @staticmethod
    def generate_random_model(z_vals, ordinal_scores, n_event_biomarkers, N_S, seed=None):
        """Sample random subtype sequences with per-biomarker monotonic score ordering."""
        rng = np.random.default_rng(seed)

        z_vals = np.asarray(z_vals) if z_vals is not None else np.zeros((0, 0), dtype=int)
        ordinal_scores = np.asarray(ordinal_scores) if ordinal_scores is not None else np.zeros((0, 0), dtype=int)
        n_event_biomarkers = int(n_event_biomarkers)
        N_S = int(N_S)

        if N_S < 1:
            raise ValueError("N_S must be >= 1")

        n_zscore_biomarkers = z_vals.shape[0]
        n_ordinal_biomarkers = ordinal_scores.shape[0]
        n_biomarkers = n_zscore_biomarkers + n_ordinal_biomarkers + n_event_biomarkers

        max_score_cols = int(max(
            z_vals.shape[1] if z_vals.ndim == 2 and n_zscore_biomarkers > 0 else 0,
            ordinal_scores.shape[1] if ordinal_scores.ndim == 2 and n_ordinal_biomarkers > 0 else 0,
            1 if n_event_biomarkers > 0 else 0
        ))
        if max_score_cols == 0:
            return np.zeros((N_S, 0), dtype=int)

        score_vals = np.zeros((n_biomarkers, max_score_cols), dtype=float)
        if n_zscore_biomarkers > 0:
            score_vals[:n_zscore_biomarkers, :z_vals.shape[1]] = z_vals
        if n_ordinal_biomarkers > 0:
            start = n_zscore_biomarkers
            score_vals[start:start + n_ordinal_biomarkers, :ordinal_scores.shape[1]] = ordinal_scores
        if n_event_biomarkers > 0:
            score_vals[-n_event_biomarkers:, 0] = 1

        stage_score, stage_biomarker_index = MixedTypeSustain._build_stage_index(score_vals)

        N = stage_score.shape[0]
        S = np.zeros((N_S, N), dtype=int)
        possible_biomarkers = np.unique(stage_biomarker_index)

        for s in range(N_S):
            for i in range(N):
                is_min_stage_score = np.full(N, False)
                is_unselected = np.full(N, False)
                is_unselected[list(set(range(N)) - set(S[s, :i]))] = True

                for b in possible_biomarkers:
                    biomarker_unselected = np.logical_and(stage_biomarker_index == b, is_unselected)
                    if not np.any(biomarker_unselected):
                        continue

                    min_score = np.min(stage_score[biomarker_unselected])
                    is_min_stage_score[np.logical_and(biomarker_unselected, stage_score == min_score)] = True

                possible_events = np.arange(N)[is_min_stage_score]
                S[s, i] = rng.choice(possible_events)

        return S

    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, z_vals, z_max, ordinal_scores, n_event_biomarkers,
                      prob_correct_ordinal=0.9, mixture_style="mixture_GMM", seed=None):
        """Generate synthetic mixed data and split likelihood inputs for each modality."""
        rng = np.random.default_rng(seed)

        subtypes = np.asarray(subtypes, dtype=int).reshape(-1)
        stages = np.asarray(stages, dtype=int).reshape(-1)
        gt_ordering = np.asarray(gt_ordering)
        z_vals = np.asarray(z_vals) if z_vals is not None else np.zeros((0, 0), dtype=int)
        ordinal_scores = np.asarray(ordinal_scores) if ordinal_scores is not None else np.zeros((0, 0), dtype=int)
        n_event_biomarkers = int(n_event_biomarkers)

        n_zscore_biomarkers = z_vals.shape[0]
        n_ordinal_biomarkers = ordinal_scores.shape[0]
        n_biomarkers = n_zscore_biomarkers + n_ordinal_biomarkers + n_event_biomarkers

        max_score_cols = int(max(
            z_vals.shape[1] if z_vals.ndim == 2 and n_zscore_biomarkers > 0 else 0,
            ordinal_scores.shape[1] if ordinal_scores.ndim == 2 and n_ordinal_biomarkers > 0 else 0,
            1 if n_event_biomarkers > 0 else 0
        ))

        score_vals = np.zeros((n_biomarkers, max_score_cols), dtype=float)
        if n_zscore_biomarkers > 0:
            score_vals[:n_zscore_biomarkers, :z_vals.shape[1]] = z_vals
        if n_ordinal_biomarkers > 0:
            start = n_zscore_biomarkers
            score_vals[start:start + n_ordinal_biomarkers, :ordinal_scores.shape[1]] = ordinal_scores
        if n_event_biomarkers > 0:
            score_vals[-n_event_biomarkers:, 0] = 1

        bool_zscore_biomarkers = np.array([True] * n_zscore_biomarkers + [False] * (n_ordinal_biomarkers + n_event_biomarkers))

        n_samples = len(subtypes)
        n_subtypes = gt_ordering.shape[0]
        n_zscore = int(bool_zscore_biomarkers.sum())
        n_ordinal_event = int((~bool_zscore_biomarkers).sum())

        stage_score, stage_biomarker_index = MixedTypeSustain._build_stage_index(score_vals)
        n_stages = stage_score.shape[0]

        if np.any(stages < 0) or np.any(stages > n_stages):
            raise ValueError("stages must be between 0 and number of stages (inclusive)")

        if n_zscore > 0 and len(z_max) != n_zscore:
            raise ValueError("z_max length must match number of z-score biomarkers")

        if n_ordinal_event != (n_ordinal_biomarkers + n_event_biomarkers):
            raise ValueError("Internal ordinal/event split mismatch while building mixed score matrix")

        if np.isscalar(prob_correct_ordinal):
            prob_correct_ordinal = np.full(n_ordinal_biomarkers, float(prob_correct_ordinal))
        else:
            prob_correct_ordinal = np.asarray(prob_correct_ordinal, dtype=float).reshape(-1)
            if len(prob_correct_ordinal) != n_ordinal_biomarkers:
                raise ValueError("prob_correct_ordinal must be scalar or length n_ordinal_biomarkers")

        def _create_distribution(score, prob, ind):
            dist = np.full((score + 1), (1.0 - prob) / score)
            dist[ind] = prob
            return dist

        zscore_data = None
        ordinal_data_out = None
        event_data_out = None
        ordinal_prob_nl = None
        ordinal_prob_score = None
        event_prob_yes = None
        event_prob_no = None

        # ----- z-score branch
        if n_zscore > 0:
            zscore_global_idx = np.where(bool_zscore_biomarkers)[0]
            zscore_local_idx = {g: i for i, g in enumerate(zscore_global_idx)}
            stage_value = np.zeros((n_zscore, n_stages + 2, n_subtypes))

            for s in range(n_subtypes):
                S = gt_ordering[s, :].astype(int)
                S_inv = np.zeros(n_stages, dtype=int)
                S_inv[S] = np.arange(n_stages)

                for g_idx in zscore_global_idx:
                    local_idx = zscore_local_idx[g_idx]
                    event_location = np.concatenate([[0], S_inv[stage_biomarker_index == g_idx], [n_stages]])
                    event_value = np.concatenate([[0], stage_score[stage_biomarker_index == g_idx], [z_max[local_idx]]])

                    for j in range(len(event_location) - 1):
                        if j == 0:
                            index = np.arange(event_location[j], event_location[j + 1] + 2)
                            stage_value[local_idx, index, s] = np.linspace(
                                event_value[j], event_value[j + 1], event_location[j + 1] - event_location[j] + 2
                            )
                        else:
                            index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                            stage_value[local_idx, index, s] = np.linspace(
                                event_value[j], event_value[j + 1], event_location[j + 1] - event_location[j] + 1
                            )

            zscore_data = np.zeros((n_samples, n_zscore))
            for m in range(n_samples):
                zscore_data[m, :] = stage_value[:, int(stages[m]), subtypes[m]]
            zscore_data = zscore_data + rng.standard_normal((n_samples, n_zscore))

        # ----- shared ordinal/event stage logic
        if n_ordinal_event > 0:
            ordinal_global_idx = np.where(~bool_zscore_biomarkers)[0]
            ordinal_local_idx = {g: i for i, g in enumerate(ordinal_global_idx)}

            stage_value_ordinal_event = np.zeros((n_stages + 1, n_ordinal_event, n_subtypes))
            for s in range(n_subtypes):
                S = gt_ordering[s, :].astype(int)
                for stage in range(n_stages):
                    event_idx = int(S[stage])
                    biomarker_g = int(stage_biomarker_index[event_idx])
                    if biomarker_g not in ordinal_local_idx:
                        continue
                    biomarker_l = ordinal_local_idx[biomarker_g]
                    score_reached = int(stage_score[event_idx])
                    stage_value_ordinal_event[stage + 1:, biomarker_l, s] = score_reached

            # ----- ordinal branch (discrete scores)
            if n_ordinal_biomarkers > 0:
                ordinal_only_scores = score_vals[ordinal_global_idx[:n_ordinal_biomarkers]].max(axis=1).astype(int)
                max_score = int(np.max(ordinal_only_scores))
                ordinal_data = np.zeros((n_samples, n_ordinal_biomarkers), dtype=int)
                p_nl_dists = []
                p_score_dists = []

                for b in range(n_ordinal_biomarkers):
                    score = int(ordinal_only_scores[b])
                    prob = float(prob_correct_ordinal[b])
                    p_nl_dist = _create_distribution(score, prob, ind=0)
                    p_nl_dists.append(p_nl_dist)
                    ordinal_data[:, b] = rng.choice(score + 1, n_samples, replace=True, p=p_nl_dist)

                    p_score_dist = np.full((score, score + 1), (1.0 - prob) / score)
                    for score_idx in range(score):
                        p_score_dist[score_idx, :] = _create_distribution(score, prob, ind=score_idx + 1)
                    p_score_dists.append(p_score_dist)

                for m in range(n_samples):
                    subtype_m = subtypes[m]
                    stage_m = int(stages[m])
                    score_reached = stage_value_ordinal_event[stage_m, :n_ordinal_biomarkers, subtype_m]
                    for b, value in enumerate(score_reached):
                        if value > 0:
                            idx_value = int(value)
                            ordinal_data[m, b] = rng.choice(
                                ordinal_only_scores[b] + 1, 1, replace=True, p=p_score_dists[b][idx_value - 1, :]
                            )[0]

                ordinal_prob_nl = np.zeros((n_samples, n_ordinal_biomarkers))
                ordinal_prob_score = np.zeros((n_samples, n_ordinal_biomarkers, max_score))
                for b in range(n_ordinal_biomarkers):
                    score = int(ordinal_only_scores[b])
                    ordinal_prob_nl[:, b] = p_nl_dists[b][ordinal_data[:, b]]
                    for z in range(score):
                        for score_value in range(score + 1):
                            ordinal_prob_score[ordinal_data[:, b] == score_value, b, z] = p_score_dists[b][z, score_value]
                ordinal_data_out = ordinal_data

            # ----- event branch (mixture-style continuous data)
            if n_event_biomarkers > 0:
                if mixture_style not in ("mixture_GMM", "mixture_KDE"):
                    raise ValueError("mixture_style must be 'mixture_GMM' or 'mixture_KDE'")
                if mixture_style == "mixture_KDE":
                    raise NotImplementedError("mixture_KDE event generation is not validated in MixedTypeSustain.generate_data yet.")

                mean_controls = np.zeros(n_event_biomarkers)
                std_controls = np.array([0.25] * n_event_biomarkers)

                mean_cases = np.array([1.5] * n_event_biomarkers)
                std_cases = rng.uniform(0.25, 0.50, n_event_biomarkers)

                event_data = np.zeros((n_samples, n_event_biomarkers))
                event_reached = stage_value_ordinal_event[:, n_ordinal_biomarkers:, :] > 0

                for m in range(n_samples):
                    subtype_m = subtypes[m]
                    stage_m = int(stages[m])
                    reached_m = event_reached[stage_m, :, subtype_m]

                    for b in range(n_event_biomarkers):
                        if reached_m[b]:
                            event_data[m, b] = rng.normal(mean_cases[b], std_cases[b])
                        else:
                            event_data[m, b] = rng.normal(mean_controls[b], std_controls[b])

                event_prob_no = np.zeros((n_samples, n_event_biomarkers))
                event_prob_yes = np.zeros((n_samples, n_event_biomarkers))
                for b in range(n_event_biomarkers):
                    control_pdf = stats.norm.pdf(event_data[:, b], loc=mean_controls[b], scale=std_controls[b])
                    case_pdf = stats.norm.pdf(event_data[:, b], loc=mean_cases[b], scale=std_cases[b])

                    normalizer = control_pdf + case_pdf + 1e-250
                    event_prob_no[:, b] = control_pdf / normalizer
                    event_prob_yes[:, b] = case_pdf / normalizer

                event_data_out = event_data

        return zscore_data, ordinal_data_out, event_data_out, ordinal_prob_nl, ordinal_prob_score, event_prob_yes, event_prob_no
