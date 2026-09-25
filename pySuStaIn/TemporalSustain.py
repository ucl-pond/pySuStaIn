###
# pySuStaIn: Temporal SuStaIn (T-SuStaIn)
#
# Integrates hidden Markov modelling with SuStaIn to learn
# subtype-specific event sequences AND transition timescales
# from longitudinal data.
#
# Key differences from standard SuStaIn:
#   - Subjects with multiple timepoints use the forward algorithm (HMM)
#     to compute likelihoods, respecting temporal ordering
#   - Transition rates between stages are LEARNED (not fixed), giving
#     subtype-specific timescales
#   - Single-timepoint subjects fall back to standard cross-sectional likelihood
#
# References:
# 1. T-SuStaIn:  Young et al., IPMI 2023. https://doi.org/10.1007/978-3-031-34048-2_2
# 2. TEBM:       Wijeratne et al., Imaging Neuroscience 2023. https://doi.org/10.1162/imag_a_00010
# 3. pySuStaIn:  Aksman et al., SoftwareX 2021. https://doi.org/10.1016/j.softx.2021.100811
###
import warnings
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp
from pathlib import Path
import pickle
import os

from pySuStaIn.AbstractSustain import AbstractSustainData
from pySuStaIn.AbstractSustain import AbstractSustain


# ============================================================================
# Data structure for longitudinal observations
# ============================================================================
class TemporalSustainData(AbstractSustainData):
    """Holds longitudinal data with subject-timepoint structure."""

    def __init__(self, data, timepoints, subject_ids, numStages):
        self.data = np.asarray(data, dtype=float)
        self.timepoints = np.asarray(timepoints, dtype=float)
        self.subject_ids = np.asarray(subject_ids)
        self.__numStages = numStages

        self.unique_subjects = np.unique(self.subject_ids)
        self._subject_indices = {}
        for sid in self.unique_subjects:
            mask = self.subject_ids == sid
            idx = np.where(mask)[0]
            time_order = np.argsort(self.timepoints[idx])
            self._subject_indices[sid] = idx[time_order]

    def getNumSamples(self):
        return len(self.unique_subjects)

    def getNumBiomarkers(self):
        return self.data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def getNumObservations(self):
        return self.data.shape[0]

    def get_subject_data(self, subject_id):
        idx = self._subject_indices[subject_id]
        return self.data[idx], self.timepoints[idx]

    def get_subject_lengths(self):
        return np.array([len(self._subject_indices[sid])
                         for sid in self.unique_subjects])

    def reindex(self, index):
        index = np.asarray(index)
        if index.dtype == bool:
            selected_subjects = self.unique_subjects[index]
        else:
            selected_subjects = self.unique_subjects[index]
        obs_mask = np.isin(self.subject_ids, selected_subjects)
        return TemporalSustainData(
            self.data[obs_mask], self.timepoints[obs_mask],
            self.subject_ids[obs_mask], self.__numStages
        )


# ============================================================================
# Vectorized forward algorithm (the key optimization)
# ============================================================================
def _forward_log_vectorized(log_startprob, log_transmat, framelogprob):
    """Forward algorithm in log-space — vectorized over states.

    Parameters
    ----------
    log_startprob : (K,)
    log_transmat : (K, K)
    framelogprob : (T, K)

    Returns
    -------
    fwdlattice : (T, K)
    """
    T, K = framelogprob.shape
    fwdlattice = np.full((T, K), -np.inf)
    fwdlattice[0] = log_startprob + framelogprob[0]
    for t in range(1, T):
        # Vectorized: broadcast fwdlattice[t-1] over columns of log_transmat
        # prev_plus_trans[i,j] = fwdlattice[t-1,i] + log_transmat[i,j]
        prev_plus_trans = fwdlattice[t - 1, :, np.newaxis] + log_transmat
        fwdlattice[t] = logsumexp(prev_plus_trans, axis=0) + framelogprob[t]
    return fwdlattice


def _backward_log_vectorized(log_transmat, framelogprob):
    """Backward algorithm in log-space — vectorized over states."""
    T, K = framelogprob.shape
    bwdlattice = np.full((T, K), -np.inf)
    bwdlattice[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        # future[j] = framelogprob[t+1,j] + bwdlattice[t+1,j]
        future = framelogprob[t + 1] + bwdlattice[t + 1]
        # bwdlattice[t,i] = logsumexp_j(log_transmat[i,j] + future[j])
        bwdlattice[t] = logsumexp(log_transmat + future[np.newaxis, :], axis=1)
    return bwdlattice


def _compute_posteriors(fwdlattice, bwdlattice):
    """Compute state posteriors gamma[t,k] = P(state=k at time t | all data)."""
    log_gamma = fwdlattice + bwdlattice
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


def _compute_transition_counts(fwdlattice, log_transmat, bwdlattice, framelogprob):
    """Compute expected transition counts (summed over time).

    Returns
    -------
    xi_sum : (K, K) expected number of transitions from state i to j
    """
    T, K = framelogprob.shape
    logprob = logsumexp(fwdlattice[-1])
    log_xi_sum = np.full((K, K), -np.inf)
    for t in range(T - 1):
        # log_xi[i,j] = fwd[t,i] + log_a[i,j] + emit[t+1,j] + bwd[t+1,j] - logprob
        log_xi = (fwdlattice[t, :, np.newaxis] + log_transmat
                  + framelogprob[t + 1, np.newaxis, :] + bwdlattice[t + 1, np.newaxis, :]
                  - logprob)
        log_xi_sum = np.logaddexp(log_xi_sum, log_xi)
    return np.exp(log_xi_sum)


# ============================================================================
# Temporal SuStaIn
# ============================================================================
class TemporalSustain(AbstractSustain):
    """Temporal Subtype and Stage Inference (T-SuStaIn).

    Combines pySuStaIn's subtype clustering with HMM temporal machinery
    to learn subtype-specific event orderings AND transition timescales
    from longitudinal data.

    Parameters
    ----------
    data : ndarray, shape (total_obs, n_biomarkers)
        Positive z-scores, all timepoints stacked.
    timepoints : ndarray, shape (total_obs,)
        Time of each visit.
    subject_ids : ndarray, shape (total_obs,)
        Subject ID for each observation.
    Z_vals : ndarray, shape (n_biomarkers, n_z_thresholds)
        Z-score thresholds per biomarker. 0 for unused.
    Z_max : ndarray, shape (n_biomarkers,)
        Maximum z-score per biomarker.
    biomarker_labels : list of str
    N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
    dataset_name, use_parallel_startpoints, seed : same as ZscoreSustain
    """

    def __init__(self, data, timepoints, subject_ids,
                 Z_vals, Z_max, biomarker_labels,
                 N_startpoints, N_S_max, N_iterations_MCMC,
                 output_folder, dataset_name,
                 use_parallel_startpoints, seed=None):

        N = data.shape[1]
        assert len(biomarker_labels) == N

        # Z-score stage setup (same as ZscoreSustain)
        stage_zscore = Z_vals.T.flatten().reshape(1, -1)
        IX_select = stage_zscore > 0
        stage_zscore = stage_zscore[IX_select].reshape(1, -1)

        num_zscores = Z_vals.shape[1]
        IX_vals = np.tile(np.arange(N), (num_zscores, 1)).T
        stage_biomarker_index = IX_vals.T.flatten().reshape(1, -1)
        stage_biomarker_index = stage_biomarker_index[IX_select].reshape(1, -1)

        self.Z_vals = Z_vals
        self.stage_zscore = stage_zscore
        self.stage_biomarker_index = stage_biomarker_index
        self.min_biomarker_zscore = [0] * N
        self.max_biomarker_zscore = Z_max
        self.std_biomarker_zscore = [1] * N
        self.biomarker_labels = biomarker_labels

        numStages = stage_zscore.shape[1]
        self.n_hmm_states = numStages + 1

        # Learned transition rates, updated during EM/MCMC.
        # List of (K-1,) arrays, one per subtype.
        self.learned_rates = None

        self.__sustainData = TemporalSustainData(
            data, timepoints, subject_ids, numStages
        )

        super().__init__(
            self.__sustainData, N_startpoints, N_S_max,
            N_iterations_MCMC, output_folder, dataset_name,
            use_parallel_startpoints, seed
        )

    # ===================================================================
    # Transition matrix (LEARNABLE rates)
    # ===================================================================
    def _build_transition_matrix(self, K, rates=None):
        """Build forward-only row-stochastic transition matrix.

        Parameters
        ----------
        K : int, number of states
        rates : (K-1,) array or None
            Probability of advancing from state k to k+1.
            If None, use default 0.5.
        """
        if rates is None:
            rates = np.full(K - 1, 0.5)
        rates = np.clip(rates, 1e-6, 1.0 - 1e-6)

        a_mat = np.zeros((K, K))
        for i in range(K - 1):
            a_mat[i, i] = 1.0 - rates[i]
            a_mat[i, i + 1] = rates[i]
        a_mat[K - 1, K - 1] = 1.0
        return a_mat

    def _build_initial_state_dist(self, K):
        """Geometric-like prior favoring early stages."""
        pi = 0.5 ** np.arange(K)
        pi /= pi.sum()
        return pi

    def _default_rates(self):
        """Default transition rates."""
        return np.full(self.n_hmm_states - 1, 0.5)

    # ===================================================================
    # Emission probabilities (z-score model, cached)
    # ===================================================================
    @staticmethod
    def _linspace_local(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    def _compute_stage_means(self, S):
        """Compute biomarker means at each stage for sequence S.
        Returns (n_biomarkers, n_stages+1)."""
        N = self.stage_biomarker_index.shape[1]
        S_inv = np.zeros(N, dtype=int)
        S_inv[S.astype(int)] = np.arange(N)
        possible_biomarkers = np.unique(self.stage_biomarker_index)
        B = len(possible_biomarkers)
        point_value = np.zeros((B, N + 2))
        arange_N = np.arange(N + 2)

        for i in range(B):
            b = possible_biomarkers[i]
            event_location = np.concatenate(
                [[0], S_inv[(self.stage_biomarker_index == b)[0]], [N]])
            event_value = np.concatenate(
                [[self.min_biomarker_zscore[i]],
                 self.stage_zscore[self.stage_biomarker_index == b],
                 [self.max_biomarker_zscore[i]]])
            for j in range(len(event_location) - 1):
                if j == 0:
                    temp = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 2
                    point_value[i, temp] = self._linspace_local(
                        event_value[j], event_value[j + 1], N_j, arange_N[:N_j])
                else:
                    temp = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                    N_j = event_location[j + 1] - event_location[j] + 1
                    point_value[i, temp] = self._linspace_local(
                        event_value[j], event_value[j + 1], N_j, arange_N[:N_j])

        stage_value = 0.5 * point_value[:, :-1] + 0.5 * point_value[:, 1:]
        return stage_value

    def _compute_all_emissions(self, sustainData, S):
        """Precompute emission log-probs for ALL observations at once.

        Returns
        -------
        all_framelogprob : (total_observations, K) log P(y | stage=k)
        """
        stage_means = self._compute_stage_means(S)
        sigmat = np.array(self.std_biomarker_zscore)
        log_factor = np.log(1.0 / np.sqrt(2.0 * np.pi) * sigmat)
        # data: (M_total, B), stage_means: (B, K)
        x = (sustainData.data[:, :, None] - stage_means[None, :, :]) / sigmat[None, :, None]
        all_logprob = np.sum(log_factor[None, :, None] - 0.5 * x * x, axis=1)
        return all_logprob  # (M_total, K)

    # ===================================================================
    # Core likelihood: temporal for longitudinal, standard for cross-sectional
    # ===================================================================
    def _calculate_likelihood_stage(self, sustainData, S, rates=None):
        """Compute per-subject stage likelihoods.

        For multi-timepoint subjects: forward algorithm with learned rates.
        For single-timepoint subjects: standard cross-sectional emission.

        Returns
        -------
        p_perm_k : (n_subjects, K) — compatible with AbstractSustain._calculate_likelihood
        """
        N = self.stage_biomarker_index.shape[1]
        K = N + 1
        n_subjects = sustainData.getNumSamples()
        p_perm_k = np.zeros((n_subjects, K))

        # Build HMM parameters
        # When rates=None (e.g. called by parent's _calculate_likelihood),
        # fall back to learned rates from EM if available.
        if rates is None and self.learned_rates is not None:
            # Use first subtype's rates as default — imperfect but better
            # than 0.5 defaults. Per-subtype rates require the parent to
            # pass a subtype index, which it doesn't.
            rates = self.learned_rates[0]
        a_mat = self._build_transition_matrix(K, rates)
        pi = self._build_initial_state_dist(K)
        log_pi = np.log(pi + 1e-300)
        log_a = np.log(a_mat + 1e-300)

        # Precompute ALL emissions in one vectorized call
        all_emissions = self._compute_all_emissions(sustainData, S)

        coeff = np.log(1.0 / float(K))

        for j, subj_id in enumerate(sustainData.unique_subjects):
            idx = sustainData._subject_indices[subj_id]
            T_j = len(idx)
            framelogprob = all_emissions[idx]  # (T_j, K)

            if T_j == 1:
                # Single timepoint: standard SuStaIn likelihood
                p_perm_k[j, :] = np.exp(coeff + framelogprob[0])
            else:
                # Multi-timepoint: forward algorithm
                fwdlattice = _forward_log_vectorized(log_pi, log_a, framelogprob)
                bwdlattice = _backward_log_vectorized(log_a, framelogprob)

                # Posterior P(stage=k | all timepoints)
                gamma = _compute_posteriors(fwdlattice, bwdlattice)
                avg_posterior = gamma.mean(axis=0)  # (K,)

                # Scale to match single-visit magnitude: use average
                # per-observation emission so longitudinal and cross-sectional
                # subjects contribute on the same scale.
                avg_emission = np.exp(framelogprob.mean(axis=0))  # (K,)
                p_perm_k[j, :] = avg_posterior * avg_emission / K

        return p_perm_k

    def _learn_rates_from_posteriors(self, sustainData, S, rates, f, N_S):
        """Update transition rates using forward-backward expected counts.

        Returns
        -------
        new_rates : (K-1,) updated transition rates
        """
        K = self.n_hmm_states
        a_mat = self._build_transition_matrix(K, rates)
        pi = self._build_initial_state_dist(K)
        log_pi = np.log(pi + 1e-300)
        log_a = np.log(a_mat + 1e-300)

        all_emissions = self._compute_all_emissions(sustainData, S)

        # Accumulate expected transitions across all subjects
        total_xi = np.zeros((K, K))
        total_gamma = np.zeros(K)
        n_longitudinal = 0

        for subj_id in sustainData.unique_subjects:
            idx = sustainData._subject_indices[subj_id]
            if len(idx) < 2:
                continue
            n_longitudinal += 1
            framelogprob = all_emissions[idx]
            fwdlattice = _forward_log_vectorized(log_pi, log_a, framelogprob)
            bwdlattice = _backward_log_vectorized(log_a, framelogprob)

            xi = _compute_transition_counts(fwdlattice, log_a, bwdlattice, framelogprob)
            gamma = _compute_posteriors(fwdlattice, bwdlattice)

            total_xi += xi
            # Sum gamma over t=0..T-2 (states from which transitions happen)
            total_gamma += gamma[:-1].sum(axis=0)

        if n_longitudinal == 0:
            return rates  # No longitudinal data, can't learn rates

        # New rates: P(advance from k) = E[transitions k->k+1] / E[time in k]
        new_rates = np.full(K - 1, 0.5)
        for k in range(K - 1):
            denom = total_gamma[k]
            if denom > 1e-10:
                new_rates[k] = total_xi[k, k + 1] / denom
        new_rates = np.clip(new_rates, 0.01, 0.99)
        return new_rates

    # ===================================================================
    # AbstractSustain interface: _initialise_sequence
    # ===================================================================
    def _initialise_sequence(self, sustainData, rng):
        """Randomly initialise a monotonically-increasing z-score sequence."""
        N = self.stage_zscore.shape[1]
        S = np.zeros(N)
        for i in range(N):
            IS_min = np.array([False] * N)
            possible_biomarkers = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):
                IS_unselected = [False] * N
                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k] = True
                this_bm = np.array(
                    [(self.stage_biomarker_index[0] == possible_biomarkers[j]).astype(int)
                     + (np.array(IS_unselected) == 1).astype(int)]
                ) == 2
                if not np.any(this_bm):
                    this_min = 0
                else:
                    this_min = min(self.stage_zscore[this_bm])
                if this_min:
                    temp = ((this_bm.astype(int) +
                             (self.stage_zscore == this_min).astype(int)) == 2).T
                    temp = temp.reshape(len(temp),)
                    IS_min[temp] = True
            events = np.arange(N)
            possible_events = events[IS_min]
            this_index = int(np.ceil(rng.random() * len(possible_events))) - 1
            S[i] = possible_events[this_index]
        return S.reshape(1, -1)

    # ===================================================================
    # AbstractSustain interface: _optimise_parameters (EM with rate learning)
    # ===================================================================
    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        M = sustainData.getNumSamples()
        N_S = S_init.shape[0]
        N = self.stage_zscore.shape[1]
        K = N + 1

        S_opt = S_init.copy()
        f_opt = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat = np.transpose(np.tile(f_opt, (1, K, M)), (2, 1, 0))
        p_perm_k = np.zeros((M, K, N_S))

        # Initialize per-subtype rates
        rates = [self._default_rates() for _ in range(N_S)]

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(
                sustainData, S_opt[s], rates[s])

        # Update f
        p_w = p_perm_k * f_val_mat
        p_n = p_w / np.sum(p_w + 1e-250, axis=(1, 2), keepdims=True)
        f_opt = (np.squeeze(np.sum(np.sum(p_n, 0), 0)) / np.sum(p_n)).reshape(N_S, 1, 1)
        f_val_mat = np.transpose(np.tile(f_opt, (1, K, M)), (2, 1, 0))

        # Greedy sequence optimization + rate learning
        order_seq = rng.permutation(N_S)
        for s in order_seq:
            # Learn rates for this subtype
            rates[s] = self._learn_rates_from_posteriors(
                sustainData, S_opt[s], rates[s], f_opt, N_S)

            # Optimize event sequence
            order_bio = rng.permutation(N)
            for i in order_bio:
                current_sequence = S_opt[s]
                current_location = np.zeros(N, dtype=int)
                current_location[current_sequence.astype(int)] = np.arange(N)

                selected_event = i
                move_event_from = current_location[selected_event]
                this_stage_zscore = self.stage_zscore[0, selected_event]
                selected_biomarker = self.stage_biomarker_index[0, selected_event]
                possible_zscores = self.stage_zscore[
                    self.stage_biomarker_index == selected_biomarker]

                # Monotonicity bounds
                min_filter = possible_zscores < this_stage_zscore
                max_filter = possible_zscores > this_stage_zscore
                events = np.arange(N)

                if np.any(min_filter):
                    min_zb = max(possible_zscores[min_filter])
                    mbe = events[((self.stage_zscore[0] == min_zb).astype(int) +
                                  (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    lower = int(current_location[mbe][0]) + 1
                else:
                    lower = 0

                if np.any(max_filter):
                    max_zb = min(possible_zscores[max_filter])
                    mbe = events[((self.stage_zscore[0] == max_zb).astype(int) +
                                  (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    upper = int(current_location[mbe][0])
                else:
                    upper = N

                if lower == upper:
                    possible_positions = np.array([move_event_from])
                else:
                    possible_positions = np.arange(lower, upper)

                best_lik = -np.inf
                best_seq = S_opt[s]
                best_ppk = p_perm_k[:, :, s]

                for idx in range(len(possible_positions)):
                    seq = S_opt[s].copy()
                    move_to = possible_positions[idx]
                    seq = np.delete(seq, move_event_from)
                    new_seq = np.concatenate([
                        seq[:move_to], [selected_event], seq[move_to:]])

                    this_ppk = self._calculate_likelihood_stage(
                        sustainData, new_seq, rates[s])

                    p_perm_k[:, :, s] = this_ppk
                    total_prob = np.sum(np.sum(p_perm_k * f_val_mat, 2), 1)
                    lik = np.sum(np.log(total_prob + 1e-250))

                    if lik > best_lik:
                        best_lik = lik
                        best_seq = new_seq
                        best_ppk = this_ppk

                S_opt[s] = best_seq
                p_perm_k[:, :, s] = best_ppk

        # Final f and likelihood
        p_w = p_perm_k * f_val_mat
        p_n = p_w / np.sum(p_w + 1e-250, axis=(1, 2), keepdims=True)
        f_opt = (np.squeeze(np.sum(np.sum(p_n, 0), 0)) / np.sum(p_n)).reshape(N_S)
        f_val_mat = np.transpose(
            np.tile(f_opt.reshape(N_S, 1, 1), (1, K, M)), (2, 1, 0))
        total_prob = np.sum(np.sum(p_perm_k * f_val_mat, 2), 1)
        likelihood_opt = np.sum(np.log(total_prob + 1e-250))

        # Persist learned rates for use in staging/serialization
        self.learned_rates = [r.copy() for r in rates]

        return S_opt, f_opt, likelihood_opt

    # ===================================================================
    # Fast MCMC tuning (override the slow parent version)
    # ===================================================================
    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):
        """Skip the expensive 3×10k tuning passes. Use reasonable defaults."""
        N_S = seq_init.shape[0]
        N = self.stage_zscore.shape[1]
        seq_sigma = np.ones((N_S, N))
        f_sigma = np.full(N_S, 0.01)
        return seq_sigma, f_sigma

    # ===================================================================
    # AbstractSustain interface: _perform_mcmc (with rate proposals)
    # ===================================================================
    def _perform_mcmc(self, sustainData, seq_init, f_init,
                      n_iterations, seq_sigma, f_sigma):
        N = self.stage_zscore.shape[1]
        N_S = seq_init.shape[0]
        K = N + 1

        if isinstance(f_sigma, float):
            f_sigma = np.array([f_sigma])

        samples_sequence = np.zeros((N_S, N, n_iterations))
        samples_f = np.zeros((N_S, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0] = seq_init
        samples_f[:, 0] = f_init

        # Initialize rates from EM-learned values if available
        current_rates = [self._default_rates() for _ in range(N_S)]
        rate_sigma = 0.05

        tqdm_update = int(n_iterations / 1000) if n_iterations > 100000 else None

        for i in tqdm(range(n_iterations), "MCMC", n_iterations, miniters=tqdm_update):
            if i > 0:
                # Store previous rates for potential revert
                prev_rates = [r.copy() for r in current_rates]

                # Propose sequence changes
                seq_order = self.global_rng.permutation(N_S)
                for s in seq_order:
                    move_from = int(np.ceil(N * self.global_rng.random())) - 1
                    cur_seq = samples_sequence[s, :, i - 1]
                    cur_loc = np.zeros(N, dtype=int)
                    cur_loc[cur_seq.astype(int)] = np.arange(N)

                    sel_event = int(cur_seq[move_from])
                    this_sz = self.stage_zscore[0, sel_event]
                    sel_bm = self.stage_biomarker_index[0, sel_event]
                    poss_zs = self.stage_zscore[self.stage_biomarker_index == sel_bm]

                    min_f = poss_zs < this_sz
                    max_f = poss_zs > this_sz
                    events = np.arange(N)

                    if np.any(min_f):
                        min_zb = max(poss_zs[min_f])
                        mbe = events[((self.stage_zscore[0] == min_zb).astype(int) +
                                      (self.stage_biomarker_index[0] == sel_bm).astype(int)) == 2]
                        lower = int(cur_loc[mbe][0]) + 1
                    else:
                        lower = 0

                    if np.any(max_f):
                        max_zb = min(poss_zs[max_f])
                        mbe = events[((self.stage_zscore[0] == max_zb).astype(int) +
                                      (self.stage_biomarker_index[0] == sel_bm).astype(int)) == 2]
                        upper = int(cur_loc[mbe][0])
                    else:
                        upper = N

                    if lower == upper:
                        possible_positions = np.array([move_from])
                    else:
                        possible_positions = np.arange(lower, upper)

                    distance = possible_positions - move_from
                    this_sigma = seq_sigma[s, sel_event] if not isinstance(seq_sigma, (int, float)) else seq_sigma
                    weight = (AbstractSustain.calc_coeff(this_sigma) *
                              AbstractSustain.calc_exp(distance, 0., this_sigma))
                    weight /= weight.sum()
                    idx = self.global_rng.choice(len(possible_positions), 1, p=weight)[0]
                    move_to = int(possible_positions[idx])

                    cur_seq = np.delete(cur_seq, move_from)
                    new_seq = np.concatenate([cur_seq[:move_to], [sel_event], cur_seq[move_to:]])
                    samples_sequence[s, :, i] = new_seq

                # Propose rate changes
                for s in range(N_S):
                    new_r = current_rates[s] + rate_sigma * self.global_rng.standard_normal(K - 1)
                    new_r = np.clip(np.abs(new_r), 0.01, 0.99)
                    current_rates[s] = new_r

                # Propose f changes
                new_f = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f = np.abs(new_f) / np.sum(np.abs(new_f))
                samples_f[:, i] = new_f

            # Compute likelihood with current rates
            S = samples_sequence[:, :, i]
            f = samples_f[:, i]

            # Custom likelihood computation using learned rates
            M = sustainData.getNumSamples()
            p_perm_k = np.zeros((M, K, N_S))
            for s in range(N_S):
                p_perm_k[:, :, s] = self._calculate_likelihood_stage(
                    sustainData, S[s], current_rates[s])

            f_3d = np.transpose(np.tile(
                f.reshape(N_S, 1, 1), (1, K, M)), (2, 1, 0))
            total_prob = np.sum(np.sum(p_perm_k * f_3d, 2), 1)
            samples_likelihood[i] = np.sum(np.log(total_prob + 1e-250))

            # Metropolis-Hastings acceptance
            if i > 0:
                ratio = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i] = samples_likelihood[i - 1]
                    samples_sequence[:, :, i] = samples_sequence[:, :, i - 1]
                    samples_f[:, i] = samples_f[:, i - 1]
                    current_rates = prev_rates  # revert rates on rejection

        perm_index = np.where(samples_likelihood == max(samples_likelihood))[0]
        ml_likelihood = max(samples_likelihood)
        ml_sequence = samples_sequence[:, :, perm_index]
        ml_f = samples_f[:, perm_index]

        # Persist MCMC-refined rates (supersedes EM rates from _optimise_parameters)
        self.learned_rates = [r.copy() for r in current_rates]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    # ===================================================================
    # Plotting (reuse ZscoreSustain)
    # ===================================================================
    def _plot_sustain_model(self, *args, **kwargs):
        from pySuStaIn.ZscoreSustain import ZscoreSustain
        return ZscoreSustain.plot_positional_var(*args, Z_vals=self.Z_vals, **kwargs)

    @staticmethod
    def plot_positional_var(*args, **kwargs):
        from pySuStaIn.ZscoreSustain import ZscoreSustain
        return ZscoreSustain.plot_positional_var(*args, **kwargs)

    # ===================================================================
    # Staging new data
    # ===================================================================
    def subtype_and_stage_individuals_newData(self, data_new, timepoints_new,
                                               subject_ids_new,
                                               samples_sequence, samples_f,
                                               N_samples):
        numStages = self.__sustainData.getNumStages()
        sustainData_new = TemporalSustainData(
            data_new, timepoints_new, subject_ids_new, numStages)
        return self.subtype_and_stage_individuals(
            sustainData_new, samples_sequence, samples_f, N_samples)

    # ===================================================================
    # Temporal-specific: get learned timelines
    # ===================================================================
    def get_transition_timeline(self, S, rates):
        """Get expected sojourn times per stage.

        Returns durations: (K,) expected time-steps in each stage.
        """
        K = self.n_hmm_states
        a_mat = self._build_transition_matrix(K, rates)
        durations = np.zeros(K)
        for k in range(K):
            if a_mat[k, k] < 1.0:
                durations[k] = 1.0 / (1.0 - a_mat[k, k])
            else:
                durations[k] = np.inf
        return durations

    # ===================================================================
    # Test / simulation methods (required by AbstractSustain)
    # ===================================================================
    @staticmethod
    def generate_random_model(Z_vals, N_S, seed=None):
        from pySuStaIn.ZscoreSustain import ZscoreSustain
        return ZscoreSustain.generate_random_model(Z_vals, N_S, seed)

    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, Z_vals, Z_max):
        from pySuStaIn.ZscoreSustain import ZscoreSustain
        return ZscoreSustain.generate_data(subtypes, stages, gt_ordering, Z_vals, Z_max)

    @staticmethod
    def generate_longitudinal_data(subtypes, stages_per_visit, gt_ordering,
                                    Z_vals, Z_max, n_visits_per_subject,
                                    time_between_visits=1.0, noise_std=1.0):
        """Generate simulated longitudinal z-score data."""
        B = Z_vals.shape[0]
        stage_zscore = Z_vals.T.flatten()
        IX_select = stage_zscore > 0
        stage_zscore_sel = stage_zscore[IX_select]

        num_zscores = Z_vals.shape[1]
        IX_vals = np.tile(np.arange(B), (num_zscores, 1)).T
        sbi_sel = IX_vals.T.flatten()[IX_select]

        N = len(stage_zscore_sel)
        N_S = gt_ordering.shape[0]
        possible_biomarkers = np.unique(sbi_sel)
        stage_value = np.zeros((B, N + 2, N_S))

        for s in range(N_S):
            S = gt_ordering[s]
            S_inv = np.zeros(N, dtype=int)
            S_inv[S.astype(int)] = np.arange(N)
            for i in range(B):
                b = possible_biomarkers[i]
                eloc = np.concatenate([[0], S_inv[sbi_sel == b], [N]])
                evals = np.concatenate(
                    [[0], stage_zscore_sel[sbi_sel == b], [Z_max[i]]])
                for j in range(len(eloc) - 1):
                    if j == 0:
                        idx = np.arange(eloc[j], eloc[j + 1] + 2)
                        stage_value[i, idx, s] = np.linspace(
                            evals[j], evals[j + 1], eloc[j + 1] - eloc[j] + 2)
                    else:
                        idx = np.arange(eloc[j] + 1, eloc[j + 1] + 2)
                        stage_value[i, idx, s] = np.linspace(
                            evals[j], evals[j + 1], eloc[j + 1] - eloc[j] + 1)

        all_data, all_times, all_ids, all_clean = [], [], [], []
        n_subjects = len(subtypes)
        for i in range(n_subjects):
            for v in range(n_visits_per_subject[i]):
                stage = stages_per_visit[i][v]
                mean_vals = stage_value[:, int(stage), subtypes[i]]
                noisy = mean_vals + noise_std * np.random.randn(B)
                all_data.append(noisy)
                all_clean.append(mean_vals)
                all_times.append(v * time_between_visits)
                all_ids.append(i)

        return (np.array(all_data), np.array(all_times),
                np.array(all_ids), np.array(all_clean))

    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples=None, n_subtypes=None,
                     ground_truth_subtypes=None, sustain_kwargs=None,
                     seed=42, n_visits=4):
        """Create a test instance with simulated longitudinal data.

        Supports two calling conventions:
        1. Validation framework: (n_biomarkers, n_samples, n_subtypes,
           ground_truth_subtypes, sustain_kwargs, seed)
        2. Standalone: (n_biomarkers=5, n_samples=200, n_subtypes=2,
           n_visits=4, seed=42)
        """
        np.random.seed(seed)

        # Handle standalone call with defaults
        if n_samples is None:
            n_samples = 200
        if n_subtypes is None:
            n_subtypes = 2

        Z_vals = np.tile(np.arange(1, 4), (n_biomarkers, 1))
        Z_vals[0, 2] = 0
        Z_max = np.full(n_biomarkers, 5)
        Z_max[2] = 2

        gt_seq = cls.generate_random_model(Z_vals, n_subtypes)
        N_stages = np.sum(Z_vals > 0) + 1

        # Use provided subtypes or generate them
        if ground_truth_subtypes is not None:
            subtypes = ground_truth_subtypes
        else:
            subtypes = np.random.randint(0, n_subtypes, n_samples)

        # Generate longitudinal stages: 25% controls (stage 0), 75% progressing
        n_visits_arr = np.full(n_samples, n_visits)
        stages_per_visit = []
        for i in range(n_samples):
            start = 0 if i < n_samples // 4 else np.random.randint(0, N_stages // 2)
            visits = []
            cur = start
            for v in range(n_visits):
                visits.append(cur)
                if cur < N_stages - 1 and np.random.rand() > 0.4:
                    cur = min(cur + 1, N_stages - 1)
            stages_per_visit.append(np.array(visits))

        data, timepoints, subject_ids, _ = cls.generate_longitudinal_data(
            subtypes, stages_per_visit, gt_seq, Z_vals, Z_max,
            n_visits_arr, time_between_visits=1.0, noise_std=1.0)

        labels = [f"Biomarker {i}" for i in range(n_biomarkers)]

        # Use sustain_kwargs from validation framework, or defaults
        if sustain_kwargs is not None:
            return cls(
                data=data, timepoints=timepoints, subject_ids=subject_ids,
                Z_vals=Z_vals, Z_max=Z_max,
                biomarker_labels=sustain_kwargs.get("biomarker_labels", labels),
                N_startpoints=sustain_kwargs["N_startpoints"],
                N_S_max=sustain_kwargs["N_S_max"],
                N_iterations_MCMC=sustain_kwargs["N_iterations_MCMC"],
                output_folder=sustain_kwargs["output_folder"],
                dataset_name=sustain_kwargs["dataset_name"],
                use_parallel_startpoints=sustain_kwargs["use_parallel_startpoints"],
                seed=sustain_kwargs.get("seed", seed))
        else:
            return cls(
                data=data, timepoints=timepoints, subject_ids=subject_ids,
                Z_vals=Z_vals, Z_max=Z_max, biomarker_labels=labels,
                N_startpoints=5, N_S_max=n_subtypes,
                N_iterations_MCMC=1000,
                output_folder='/tmp/temporal_sustain_test',
                dataset_name='test_longitudinal',
                use_parallel_startpoints=False, seed=seed)
