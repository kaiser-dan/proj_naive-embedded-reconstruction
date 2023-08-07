"""CLI to train a reconstruction classification model on an embedded remnant multiplex.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import pickle
import argparse

# --- Source code ---
from embmplxrec import classifiers
from embmplxrec import features
import embmplxrec.utils

# --- Globals ---


## Filepaths & templates
# * Relative to project root!
DIR_REMNANTS = os.path.join("data", "input", "remnants", "")
DIR_EMBEDDINGS = os.path.join("data", "interim", "caches", "")
DIR_MODELS = os.path.join("data", "interim", "models", "")
FILEPATH_TEMPLATE = "model_normalized-{normalize}_{basename}"
## Logging
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/train_model_{embmplxrec.utils.get_today(time=True)}.log")

# ================ FUNCTIONS ======================
# --- Command-line Interface ---
def check_file_exists():
    # Construct output filepath
    filepath_output_base = FILEPATH_TEMPLATE.format()
    filepath_output = os.path.isfile(DIR_MODELS, filepath_output_base)

    # Check if model already exists
    if os.path.isfile(filepath_output):
        logger.info(f"File '{filepath_output_base}' already exists! Skipping model training and evaluation.")
        return True

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Train reconstruction model and save to disk.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument(
        "filepath_remnants",
        type=str,
        help="Filepath of pickled remnant duplex.")
    parser.add_argument(
        "filepath_embeddings",
        type=str,
        help="Filepath of pickled remnant embeddings.")

    # Optional arguments, flag
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embedding vectors before calculating distance-based features.")
    parser.add_argument(
        "--save-reconstruction",
        dest="save_reconstruction",
        action="store_true",
        help="Save reconstructed multiplex to disk.")

    return parser

def verify_args(args):
    # - File input -
    if not os.path.isfile(args.filepath_remnants):
        raise FileNotFoundError(f"{args.filepath_remnants} not found!")
    if not os.path.isfile(args.filepath_embeddings):
        raise FileNotFoundError(f"{args.filepath_embeddings} not found!")

    return

def gather_args():
    # Initialize argparse
    parser = setup_argument_parser()

    # Gather command-line arguments
    args = vars(parser.parse_args())

    # Verify arguments are acceptable
    verify_args(args)

    return args

# --- Training model ---
def train_model(X_train, Y_train):
    # Train model
    


# --- Testing model ---
def test_model(model, X_test, Y_test, class_ = 0):
    pass



# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Check if file exists
    if check_file_exists():
        return

    # Separate directory and basename of input feature sets
    _, basename = os.path.split(args.filepath)

    # Load feature sets
    X_train, Y_train, X_test, Y_test = 

    # Train model
    model = train_model(X_train, Y_train)

    # Save model to disk
    model.save(filepath_output)
    # Evaluate model
    record = test_model(model, remnant_multiplex, remnant_embeddings, args.filepath_remnants)



if __name__ == "__main__":
    main()
