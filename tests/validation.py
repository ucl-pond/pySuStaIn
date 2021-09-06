from pathlib import Path
import time
import shutil

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pySuStaIn

import create_validation


def test(seed, sustain_classes, time_flag):
    validation_params = create_validation.initialize_validation(seed)
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
        print(f"Testing {sustain_class.__name__}\n")
        try:
            load_results(sustain_class)
        except FileNotFoundError as e:
            raise Exception(f"There is no results file for {sustain_class.__name__}. Please first run `create_validation.py`") from e
        # Define the output folder for this class
        sustain_kwargs["output_folder"] = Path.cwd() / "temp"
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
            if time_flag:
                start = time.time()
            (samples_sequence, samples_f, ml_subtype,
            prob_ml_subtype, ml_stage, prob_ml_stage,
            prob_subtype_stage) = sustain_model.run_sustain_algorithm()
            if time_flag:
                end = time.time() - start
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
        # Load the results
        results = load_results(sustain_class)
        # and compare against the results output by sustain
        assert_frame_equal(df, results)
        print(f"{sustain_class.__name__} test passed!")
        if time_flag:
            save_time(sustain_class, end)
    print("Test passed!")


def load_results(sustain_class):
    return pd.read_csv(
        Path.cwd() / f"{sustain_class.__name__}_results.csv",
        index_col=None
    )


def save_time(sustain_class, end):
    time_file = Path.cwd() / "times.csv"
    if time_file.is_file():
        df = pd.read_csv(time_file, index_col=None)
    else:
        time_file.touch()
        df = pd.DataFrame()

    df = df.append({
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "method": sustain_class.__name__,
        "time": end
    }, ignore_index=True)
    df.to_csv(time_file, index=False)


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
    parser.add_argument(
        "-t", "--time",
        action="store_true",
        help="Time the execution"
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
        default="mixturesustain",
        choices=[i.__name__.lower() for i in pySuStaIn.AbstractSustain.__subclasses__()] + [i.__name__.lower().replace("sustain", "") for i in pySuStaIn.AbstractSustain.__subclasses__()],
        help="Name of single class to create new validation"
    )
    args = parser.parse_args()
    # Test all sustain subclasses
    if args.full:
        sustain_classes = pySuStaIn.AbstractSustain.__subclasses__()
    # Otherwise test a single class
    else:
        # Allow less verbose selections
        if "sustain" in args.sustainclass:
            selected_class = args.sustainclass
        else:
            selected_class = args.sustainclass + "sustain"
        # Get all available subclasses
        class_dict = {
            i.__name__.lower(): i for i in pySuStaIn.AbstractSustain.__subclasses__()
        }
        sustain_classes = [class_dict[selected_class]]
    test(args.seed, sustain_classes, args.time)
