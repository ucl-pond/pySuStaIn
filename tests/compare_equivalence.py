"""
Compare two versions of pySuStaIn for statistical equivalence.

A change that alters the use of the random number generator (or that
changes floating-point roundoff enough to flip an MCMC decision) cannot
pass `validation.py`, which compares against fixed reference results.
This script instead fits ZscoreSustain on the same synthetic data with
several seeds, once with each version of the code, and compares the
distributions of the results:

- the maximum MCMC likelihood for each number of subtypes
  (Mann-Whitney U test between the two versions)
- recovery of the ground-truth sequences (Kendall tau)
- agreement of the maximum-likelihood subtype of each subject between the
  two versions for the same seed (adjusted Rand index)
- the positional variance of the events in the MCMC samples

Usage:
    python compare_equivalence.py --ref /path/to/old/checkout \\
        --new /path/to/new/checkout --seeds 10 --out equivalence_results

Each checkout is put first on PYTHONPATH of a separate process, so the two
versions never share an interpreter. The synthetic data is generated once,
with the reference checkout, and both versions fit the same data.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def simulate(n_biomarkers, n_samples, n_subtypes, data_seed):
    """Make synthetic z-score data with known sequences and subtypes."""
    from pySuStaIn import ZscoreSustain

    rng = np.random.RandomState(data_seed)
    subtype_fractions = np.array([0.5, 0.3, 0.2])[:n_subtypes]
    subtype_fractions /= subtype_fractions.sum()
    gt_subtypes = rng.choice(
        range(n_subtypes), n_samples, replace=True, p=subtype_fractions
    ).astype(int)

    # generate_random_model and generate_data use the legacy global RNG
    np.random.seed(data_seed)
    Z_vals = np.tile(np.arange(1, 4), (n_biomarkers, 1))
    Z_vals[0, 2] = 0
    Z_max = np.full((n_biomarkers,), 5)
    Z_max[2] = 2
    gt_sequences = ZscoreSustain.generate_random_model(Z_vals, n_subtypes)
    N_stages = np.sum(Z_vals > 0) + 1
    n_controls = int(np.round(n_samples * 0.25))
    gt_stages = np.vstack(
        (
            np.zeros((n_controls, 1)),
            np.random.randint(1, N_stages + 1, (n_samples - n_controls, 1)),
        )
    ).astype(int)
    data, _, _ = ZscoreSustain.generate_data(
        gt_subtypes, gt_stages, gt_sequences, Z_vals, Z_max
    )
    return data, Z_vals, Z_max, gt_sequences, gt_subtypes


def run_one(args):
    """Fit one model with the pySuStaIn found on sys.path; save a summary."""
    import pickle

    import pySuStaIn

    d = np.load(args.data_file)
    data, Z_vals, Z_max = d["data"], d["Z_vals"], d["Z_max"]
    gt_sequences, gt_subtypes = d["gt_sequences"], d["gt_subtypes"]
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = pySuStaIn.ZscoreSustain(
        data,
        Z_vals,
        Z_max,
        [str(i) for i in range(data.shape[1])],
        args.n_startpoints,
        args.n_subtypes_max,
        args.n_mcmc,
        str(out),
        "equiv",
        args.parallel,
        seed=args.seed,
    )
    model.run_sustain_algorithm()

    summary = {"gt_sequences": gt_sequences, "gt_subtypes": gt_subtypes}
    for s in range(args.n_subtypes_max):
        with open(out / "pickle_files" / f"equiv_subtype{s}.pickle", "rb") as f:
            v = pickle.load(f)
        summary[f"samples_sequence_{s}"] = v["samples_sequence"]
        summary[f"samples_likelihood_{s}"] = np.asarray(v["samples_likelihood"]).ravel()
        summary[f"ml_subtype_{s}"] = np.asarray(v["ml_subtype"]).ravel()
    np.savez(out / "summary.npz", **summary)
    print(f"Saved {out / 'summary.npz'}")


def make_data(args):
    """Generate the synthetic data with the pySuStaIn on sys.path, and save it."""
    data, Z_vals, Z_max, gt_sequences, gt_subtypes = simulate(
        args.n_biomarkers, args.n_samples, args.n_subtypes, args.data_seed
    )
    np.savez(
        args.make_data,
        data=data,
        Z_vals=Z_vals,
        Z_max=Z_max,
        gt_sequences=gt_sequences,
        gt_subtypes=gt_subtypes,
    )


def git_head(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def ml_sequences(summary, s):
    """Sequences of the MCMC sample with the highest likelihood."""
    best = np.argmax(summary[f"samples_likelihood_{s}"])
    return summary[f"samples_sequence_{s}"][:, :, best]


def sequence_tau(sequences, gt_sequences):
    """Mean Kendall tau to ground truth, for the best matching of subtypes."""
    from scipy.stats import kendalltau

    n = len(gt_sequences)
    best = -np.inf
    for perm in itertools.permutations(range(n)):
        taus = [
            kendalltau(np.argsort(sequences[perm[k]]), np.argsort(gt_sequences[k]))[0]
            for k in range(n)
        ]
        best = max(best, float(np.mean(taus)))
    return best


def positional_variance(samples_sequence):
    """Mean variance of the position of each event across MCMC samples."""
    positions = np.argsort(samples_sequence, axis=1)
    return float(np.mean(np.var(positions, axis=2)))


def compare(args):
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import adjusted_rand_score

    out = Path(args.out)
    report = {}
    for s in range(args.n_subtypes_max):
        rows = {"ref": [], "new": []}
        ari = []
        for seed in range(args.seeds):
            summaries = {}
            for version in ("ref", "new"):
                summaries[version] = np.load(
                    out / version / f"seed{seed}" / "summary.npz"
                )
                sm = summaries[version]
                row = {
                    "max_likelihood": float(np.max(sm[f"samples_likelihood_{s}"])),
                    "positional_variance": positional_variance(
                        sm[f"samples_sequence_{s}"]
                    ),
                }
                if s + 1 == args.n_subtypes:
                    row["kendall_tau"] = sequence_tau(
                        ml_sequences(sm, s), sm["gt_sequences"]
                    )
                rows[version].append(row)
            ari.append(
                adjusted_rand_score(
                    summaries["ref"][f"ml_subtype_{s}"],
                    summaries["new"][f"ml_subtype_{s}"],
                )
            )
        result = {
            "ari_ref_vs_new": {
                "median": float(np.median(ari)),
                "min": float(np.min(ari)),
            }
        }
        for metric in rows["ref"][0]:
            a = np.array([r[metric] for r in rows["ref"]])
            b = np.array([r[metric] for r in rows["new"]])
            if np.array_equal(a, b):
                p = 1.0
            else:
                p = float(mannwhitneyu(a, b).pvalue)
            result[metric] = {
                "ref_median": float(np.median(a)),
                "new_median": float(np.median(b)),
                "max_abs_diff": float(np.max(np.abs(a - b))),
                "mannwhitney_p": p,
            }
        report[f"N_S={s + 1}"] = result
    print(json.dumps(report, indent=2))
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ref", help="Checkout of the reference version")
    parser.add_argument("--new", help="Checkout of the new version")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="equivalence_results")
    parser.add_argument("--n-biomarkers", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-subtypes", type=int, default=3)
    parser.add_argument("--n-subtypes-max", type=int, default=3)
    parser.add_argument("--n-startpoints", type=int, default=10)
    parser.add_argument("--n-mcmc", type=int, default=int(1e4))
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    # Internal: generate the data, or fit one model, in this process
    parser.add_argument("--make-data", help=argparse.SUPPRESS)
    parser.add_argument("--data-file", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.make_data:
        make_data(args)
        return
    if args.run_dir:
        run_one(args)
        return

    if not args.compare_only:
        if not (args.ref and args.new):
            parser.error("--ref and --new are required unless --compare-only")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        params = {
            k: v
            for k, v in vars(args).items()
            if k not in ("compare_only", "make_data", "data_file", "run_dir", "seed")
        }
        params["ref"] = str(Path(args.ref).resolve())
        params["new"] = str(Path(args.new).resolve())
        params["ref_head"] = git_head(args.ref)
        params["new_head"] = git_head(args.new)
        manifest = out / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != params:
            sys.exit(
                f"{out} holds results of another comparison (see {manifest}); "
                "use a new --out directory or delete it"
            )
        manifest.write_text(json.dumps(params, indent=2))

        data_file = out / "data.npz"
        forwarded = [
            f"--n-biomarkers={args.n_biomarkers}",
            f"--n-samples={args.n_samples}",
            f"--n-subtypes={args.n_subtypes}",
            f"--n-subtypes-max={args.n_subtypes_max}",
            f"--n-startpoints={args.n_startpoints}",
            f"--n-mcmc={args.n_mcmc}",
            f"--data-seed={args.data_seed}",
        ] + (["--parallel"] if args.parallel else [])
        if not data_file.exists():
            env = dict(os.environ, PYTHONPATH=params["ref"])
            subprocess.run(
                [sys.executable, __file__, f"--make-data={data_file}"] + forwarded,
                env=env,
                check=True,
            )
        forwarded.append(f"--data-file={data_file}")
        for version, path in (("ref", args.ref), ("new", args.new)):
            env = dict(os.environ, PYTHONPATH=str(Path(path).resolve()))
            for seed in range(args.seeds):
                run_dir = Path(args.out) / version / f"seed{seed}"
                if (run_dir / "summary.npz").exists():
                    continue
                # never reuse the pickles of an incomplete earlier run
                shutil.rmtree(run_dir, ignore_errors=True)
                subprocess.run(
                    [sys.executable, __file__, f"--run-dir={run_dir}", f"--seed={seed}"]
                    + forwarded,
                    env=env,
                    check=True,
                )
    compare(args)


if __name__ == "__main__":
    main()
