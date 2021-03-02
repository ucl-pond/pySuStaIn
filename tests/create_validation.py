from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import pySuStaIn


def initialize_validation(seed):
    '''
    Initialize validation parameters shared across 
    all SuStaIn classes
    '''
    validation_params = {
        "sustain_kwargs": {
            "N_startpoints": 10,
            "N_S_max": 3,
            "N_iterations_MCMC": int(1e4),
            "dataset_name": "test",
            "use_parallel_startpoints": True,
            "seed": seed
        },
        "n_biomarkers": 4,
        "n_samples": 500,
        "n_subtypes": 3,
    }

    subtype_fractions = np.array([0.5, 0.30, 0.20])
    validation_params["sustain_kwargs"]["biomarker_labels"] = range(
        validation_params["n_biomarkers"]
    )
    validation_params["ground_truth_subtypes"] = np.random.RandomState(seed).choice(
        range(validation_params["n_subtypes"]), validation_params["n_samples"],
        replace=True, p=subtype_fractions
    ).astype(int)
    return validation_params


def create_new_validation(seed, sustain_classes):
    validation_params = initialize_validation(seed)
    # Extract params
    sustain_kwargs = validation_params["sustain_kwargs"]
    n_biomarkers = validation_params["n_biomarkers"]
    n_samples = validation_params["n_samples"]
    n_subtypes = validation_params["n_subtypes"]
    ground_truth_subj_ids = list(
        np.arange(1, validation_params["n_samples"]+1).astype('str')
    )
    ground_truth_subtypes = validation_params["ground_truth_subtypes"]
    # Main test loop
    for sustain_class in sustain_classes:
        # Define the output folder for this class
        sustain_kwargs["output_folder"] = Path.cwd() / f"{sustain_class.__name__}"
        # Path for results csv
        results_path = Path.cwd() / f"{sustain_class.__name__}_results.csv"
        # Delete old results if present
        if results_path.is_file():
            results_path.unlink()
        # Create the folder
        sustain_kwargs["output_folder"].mkdir(parents=True, exist_ok=True)
        # Class-specific setup for model
        sustain_model = sustain_class.test_sustain(
            n_biomarkers,
            n_samples,
            n_subtypes,
            ground_truth_subtypes,
            sustain_kwargs,
            seed=seed
        )
        # Run model
        try:
            (samples_sequence, samples_f, ml_subtype,
            prob_ml_subtype, ml_stage, prob_ml_stage,
            prob_subtype_stage) = sustain_model.run_sustain_algorithm()
        finally:
            # Remove saved files
            # TODO: Can remove then when save functionality changed
            shutil.rmtree(sustain_kwargs["output_folder"])
        # Collate results
        df = pd.DataFrame()
        df['subj_id'] = [int(i) for i in ground_truth_subj_ids]
        df['ml_subtype'] = ml_subtype
        df['prob_ml_subtype'] = prob_ml_subtype
        df['ml_stage'] = ml_stage
        df['prob_ml_stage'] = prob_ml_stage
        # Save results
        df.to_csv(
            Path.cwd() / f"{sustain_class.__name__}_results.csv",
            index=False
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Create new validation results"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Seed number used for validation"
    )
    parser.add_argument(
        "-c", "--sustainclass",
        type=str,
        default=None,
        choices=[i.__name__ for i in pySuStaIn.AbstractSustain.__subclasses__()],
        help="Name of single class to create new validation"
    )
    args = parser.parse_args()
    # Check if file already exists
    if (Path.cwd() / f"{args.sustainclass}_results.csv").is_file():
        # Require input for replacing existing validation results
        proceed = input("Warning, this will override existing validation results. Do you want to continue?")
    else:
        proceed = True
    if proceed:
        if args.sustainclass:
            class_dict = {
                i.__name__: i for i in pySuStaIn.AbstractSustain.__subclasses__()
            }
            create_new_validation(
                args.seed,
                [class_dict[args.sustainclass]]
            )
        else:
            create_new_validation(
                args.seed,
                pySuStaIn.AbstractSustain.__subclasses__()
            )
