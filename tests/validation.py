from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pySuStaIn

import create_validation


def test(seed, sustain_classes):
    validation_params = create_validation.initialize_validation(seed)
    # Extract params
    sustain_kwargs = validation_params["sustain_kwargs"]
    n_biomarkers = validation_params["n_biomarkers"]
    n_samples = validation_params["n_samples"]
    n_subtypes = validation_params["n_subtypes"]
    ground_truth_subj_ids = validation_params["ground_truth_subj_ids"]
    ground_truth_subtypes = validation_params["ground_truth_subtypes"]
    # Main test loop
    for sustain_class in sustain_classes:
        # Define the output folder for this class
        sustain_kwargs["output_folder"] = Path.cwd() / "temp"
        # Create the folder
        sustain_kwargs["output_folder"].mkdir(parents=True, exist_ok=True)
        # Class-specific setup for model
        sustain_model = sustain_class.test_sustain(
            n_biomarkers,
            n_samples,
            n_subtypes,
            ground_truth_subj_ids,
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
        # Want to load the results
        results = load_results(sustain_class)
        # and compare against the results output by sustain
        assert_frame_equal(df, results)
        print(f"{sustain_class.__name__} test passed!")
    print("Test passed!")


def load_results(sustain_class):
    return pd.read_csv(
        Path.cwd() / f"{sustain_class.__name__}_results.csv",
        index_col=None
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate SuStaIn output"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Seed number used for validation"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f", "--full",
        action="store_true",
        help="Flag to run test for all SuStaIn subclasses"
    )
    group.add_argument(
        "-c", "--sustainclass",
        type=str,
        default="MixtureSustain",
        choices=[i.__name__ for i in pySuStaIn.AbstractSustain.__subclasses__()],
        help="Name of single class to create new validation"
    )
    args = parser.parse_args()

    # Test all sustain subclasses
    if args.full:
        sustain_classes = pySuStaIn.AbstractSustain.__subclasses__()
    # Otherwise test a single class
    else:
        class_dict = {
            i.__name__: i for i in pySuStaIn.AbstractSustain.__subclasses__()
        }
        sustain_classes = [class_dict[args.sustainclass]]

    test(args.seed, sustain_classes)
