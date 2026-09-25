#!/usr/bin/env python3
"""Test suite for Temporal SuStaIn (optimized version)."""
import sys
import os
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pySuStaIn.TemporalSustain import (
    TemporalSustain, TemporalSustainData,
    _forward_log_vectorized, _backward_log_vectorized, _compute_posteriors
)


def test_data_structure():
    print("=" * 60)
    print("TEST 1: TemporalSustainData")
    print("=" * 60)

    data = np.random.randn(9, 5)
    timepoints = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3])
    subject_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    sd = TemporalSustainData(data, timepoints, subject_ids, numStages=10)

    assert sd.getNumSamples() == 3
    assert sd.getNumBiomarkers() == 5
    assert sd.getNumObservations() == 9

    d0, t0 = sd.get_subject_data(0)
    assert d0.shape == (3, 5)
    assert np.all(t0 == [0, 1, 2])

    lengths = sd.get_subject_lengths()
    assert np.all(lengths == [3, 2, 4])

    sd_sub = sd.reindex(np.array([True, False, True]))
    assert sd_sub.getNumSamples() == 2
    assert sd_sub.getNumObservations() == 7

    print("  PASSED\n")


def test_forward_algorithm():
    print("=" * 60)
    print("TEST 2: Vectorized forward algorithm")
    print("=" * 60)
    from scipy.special import logsumexp

    K, T = 4, 5
    log_pi = np.log([0.7, 0.2, 0.08, 0.02])
    a_mat = np.array([
        [0.7, 0.3, 0.0, 0.0],
        [0.0, 0.7, 0.3, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ])
    log_a = np.log(a_mat + 1e-300)
    np.random.seed(42)
    framelogprob = np.random.randn(T, K) * 2 - 5

    fwd = _forward_log_vectorized(log_pi, log_a, framelogprob)
    assert fwd.shape == (T, K)
    log_lik = logsumexp(fwd[-1])
    assert np.isfinite(log_lik)

    # Forward-only: P(state=0) should decrease over time
    bwd = _backward_log_vectorized(log_a, framelogprob)
    gamma = _compute_posteriors(fwd, bwd)
    assert gamma.shape == (T, K)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
    assert gamma[-1, 0] <= gamma[0, 0] + 0.01, \
        "Forward-only: P(state=0) should decrease over time"

    print(f"  Log-likelihood: {log_lik:.4f}")
    print(f"  P(state=0) over time: {gamma[:, 0].round(3)}")
    print("  PASSED\n")


def test_emission_and_transition():
    print("=" * 60)
    print("TEST 3: Emissions + transition matrix")
    print("=" * 60)

    n_bm = 4
    Z_vals = np.tile(np.arange(1, 3), (n_bm, 1))
    Z_max = np.full(n_bm, 4)
    data = np.random.randn(5, n_bm).clip(0)

    ts = TemporalSustain(
        data=data, timepoints=np.zeros(5), subject_ids=np.arange(5),
        Z_vals=Z_vals, Z_max=Z_max,
        biomarker_labels=[f"B{i}" for i in range(n_bm)],
        N_startpoints=1, N_S_max=1, N_iterations_MCMC=10,
        output_folder='/tmp/test_em', dataset_name='test',
        use_parallel_startpoints=False, seed=42)

    K = ts.n_hmm_states

    # Test transition matrix
    a_mat = ts._build_transition_matrix(K)
    assert a_mat.shape == (K, K)
    np.testing.assert_allclose(a_mat.sum(axis=1), 1.0, atol=1e-10)
    for i in range(K):
        assert np.all(a_mat[i, :i] == 0), f"Backward transition in row {i}"
    assert a_mat[K - 1, K - 1] == 1.0

    # Test with custom rates (K-1 rates for K states)
    rates = np.linspace(0.2, 0.8, K - 1)
    a_custom = ts._build_transition_matrix(K, rates)
    for i in range(K - 1):
        np.testing.assert_allclose(a_custom[i, i + 1], rates[i], atol=1e-6)

    # Test emissions
    rng = np.random.default_rng(42)
    S = ts._initialise_sequence(ts._TemporalSustain__sustainData, rng)[0]
    stage_means = ts._compute_stage_means(S)
    N_events = ts.stage_biomarker_index.shape[1]
    assert stage_means.shape == (n_bm, N_events + 1)

    # Monotonicity
    for b in range(n_bm):
        assert np.all(np.diff(stage_means[b]) >= -1e-10)

    print(f"  Transition matrix: {K}x{K}, forward-only ✓")
    print(f"  Stage means: {stage_means.shape}, monotonic ✓")
    print("  PASSED\n")


def test_rate_learning():
    print("=" * 60)
    print("TEST 4: Transition rate learning")
    print("=" * 60)

    n_bm, n_subj, n_visits = 3, 30, 4
    Z_vals = np.tile(np.arange(1, 3), (n_bm, 1))
    Z_max = np.full(n_bm, 4)
    np.random.seed(42)

    # Generate data with FAST progression (high true rates)
    N_events = np.sum(Z_vals > 0)
    gt_seq = np.arange(N_events).reshape(1, -1).astype(float)

    all_data, all_times, all_ids = [], [], []
    for subj in range(n_subj):
        start = np.random.randint(0, 3)
        for v in range(n_visits):
            stage = min(start + v, N_events)  # fast: advance every visit
            obs = np.random.randn(n_bm) * 0.5
            for e in range(min(stage, N_events)):
                obs[e % n_bm] += 2.0
            all_data.append(np.clip(obs, 0, None))
            all_times.append(float(v))
            all_ids.append(subj)

    data = np.array(all_data)

    ts = TemporalSustain(
        data=data, timepoints=np.array(all_times),
        subject_ids=np.array(all_ids),
        Z_vals=Z_vals, Z_max=Z_max,
        biomarker_labels=[f"B{i}" for i in range(n_bm)],
        N_startpoints=1, N_S_max=1, N_iterations_MCMC=10,
        output_folder='/tmp/test_rates', dataset_name='test_r',
        use_parallel_startpoints=False, seed=42)

    rng = np.random.default_rng(42)
    S = ts._initialise_sequence(ts._TemporalSustain__sustainData, rng)[0]

    # Start with default rates (0.5)
    initial_rates = ts._default_rates()
    # Learn rates
    learned_rates = ts._learn_rates_from_posteriors(
        ts._TemporalSustain__sustainData, S, initial_rates, [1.0], 1)

    print(f"  Initial rates: {initial_rates.round(3)}")
    print(f"  Learned rates: {learned_rates.round(3)}")

    # Fast-progressing data should produce rates > 0.5 for early stages
    assert np.any(learned_rates != initial_rates), "Rates didn't change"
    assert np.mean(learned_rates[:3]) > 0.5, \
        f"Early rates should be > 0.5 for fast data, got {learned_rates[:3].round(3)}"
    print("  Rates changed from defaults ✓")
    print("  Early rates > 0.5 (correct direction for fast data) ✓")
    print("  PASSED\n")


def test_mixed_longitudinal_crosssectional():
    print("=" * 60)
    print("TEST 5: Mixed longitudinal + cross-sectional data")
    print("=" * 60)

    n_bm = 3
    Z_vals = np.tile(np.arange(1, 3), (n_bm, 1))
    Z_max = np.full(n_bm, 4)
    np.random.seed(42)
    N_events = np.sum(Z_vals > 0)

    all_data, all_times, all_ids = [], [], []
    subj = 0

    # 10 subjects with 1 visit (cross-sectional)
    for _ in range(10):
        stage = np.random.randint(0, N_events + 1)
        obs = np.random.randn(n_bm) * 0.5
        for e in range(min(stage, N_events)):
            obs[e % n_bm] += 2.0
        all_data.append(np.clip(obs, 0, None))
        all_times.append(0.0)
        all_ids.append(subj)
        subj += 1

    # 15 subjects with 3-5 visits (longitudinal)
    for _ in range(15):
        n_visits = np.random.randint(3, 6)
        start = np.random.randint(0, 3)
        cur = start
        for v in range(n_visits):
            obs = np.random.randn(n_bm) * 0.5
            for e in range(min(cur, N_events)):
                obs[e % n_bm] += 2.0
            all_data.append(np.clip(obs, 0, None))
            all_times.append(float(v))
            all_ids.append(subj)
            if cur < N_events and np.random.rand() > 0.4:
                cur += 1
        subj += 1

    data = np.array(all_data)

    ts = TemporalSustain(
        data=data, timepoints=np.array(all_times),
        subject_ids=np.array(all_ids),
        Z_vals=Z_vals, Z_max=Z_max,
        biomarker_labels=[f"B{i}" for i in range(n_bm)],
        N_startpoints=1, N_S_max=1, N_iterations_MCMC=10,
        output_folder='/tmp/test_mixed', dataset_name='test_m',
        use_parallel_startpoints=False, seed=42)

    rng = np.random.default_rng(42)
    S = ts._initialise_sequence(ts._TemporalSustain__sustainData, rng)[0]

    p_perm_k = ts._calculate_likelihood_stage(
        ts._TemporalSustain__sustainData, S)

    n_subjects = 25  # 10 cross-sectional + 15 longitudinal
    assert p_perm_k.shape[0] == n_subjects, \
        f"Expected {n_subjects}, got {p_perm_k.shape[0]}"
    assert np.all(p_perm_k >= 0), "Negative probabilities"
    assert np.all(np.isfinite(p_perm_k)), "Non-finite probabilities"
    assert np.all(p_perm_k.sum(axis=1) > 0), "Some subjects have zero total prob"

    lengths = ts._TemporalSustain__sustainData.get_subject_lengths()
    single = lengths == 1
    multi = lengths > 1
    print(f"  Single-visit subjects: {single.sum()}, multi-visit: {multi.sum()}")
    print(f"  Single-visit prob range: [{p_perm_k[single].sum(1).min():.2e}, {p_perm_k[single].sum(1).max():.2e}]")
    print(f"  Multi-visit prob range:  [{p_perm_k[multi].sum(1).min():.2e}, {p_perm_k[multi].sum(1).max():.2e}]")
    print("  PASSED\n")


def test_full_pipeline():
    print("=" * 60)
    print("TEST 6: Full pipeline (small scale)")
    print("=" * 60)

    output_folder = '/tmp/temporal_sustain_full_test'
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    ts = TemporalSustain.test_sustain(
        n_biomarkers=3, n_samples=30, n_subtypes=1, n_visits=3, seed=42)

    ts.output_folder = output_folder
    ts.N_iterations_MCMC = 100
    ts.N_startpoints = 2

    print("  Running SuStaIn algorithm...")
    import time
    t0 = time.time()
    results = ts.run_sustain_algorithm(plot=False)
    elapsed = time.time() - t0

    samples_sequence, samples_f, ml_subtype, prob_ml_subtype, \
        ml_stage, prob_ml_stage, prob_subtype_stage = results

    n_subjects = ts._TemporalSustain__sustainData.getNumSamples()
    assert ml_subtype.shape[0] == n_subjects
    assert ml_stage.shape[0] == n_subjects
    assert samples_sequence.shape[2] == 100

    valid = ~np.isnan(ml_stage.flatten())
    assert np.sum(valid) > n_subjects * 0.5

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Subjects staged: {np.sum(valid)}/{n_subjects}")
    print(f"  Stage distribution: {np.histogram(ml_stage[valid], bins=5)[0]}")
    print("  PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("TEMPORAL SUSTAIN TEST SUITE (OPTIMIZED)")
    print("=" * 60 + "\n")

    test_data_structure()
    test_forward_algorithm()
    test_emission_and_transition()
    test_rate_learning()
    test_mixed_longitudinal_crosssectional()
    test_full_pipeline()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
