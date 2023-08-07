"""CLI to sample an observed training set from an duplex.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import argparse
import pickle

# --- Source code ---
from embmplxrec.remnants import observer
import embmplxrec.utils

# --- Globals ---
## Fixed parameters
# TODO: Find more robust way to track implemented sampling techniques
VALID_STRATEGIES = ["RANDOM"]

## Filepaths & templates
# * Relative to project root!
DIR_REMNANTS = os.path.join("data", "input", "remnants", "")
FILEPATH_TEMPLATE = "remnants_strategy-{strategy}_theta-{theta}_remrep-{rep}_{basename}"

## Logger
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/observe_remnant_{embmplxrec.utils.get_today(time=True)}.log"
)


# ================= FUNCTIONS =======================
# --- Command-line Interface ---
def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Observes a training sub-multiplex for reconstruction.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument(
        "filepath",
        type=str,
        help="Filepath of pickled duplex.")
    parser.add_argument(
        "theta",
        type=float, default=0.5,
        help="Relative size of training set, partial fraction of information.")

    # Optional arguments, flags
    parser.add_argument(
        "--strategy",
        default="RANDOM",
        const="RANDOM",
        nargs='?',
        choices=["RANDOM", "SNOWBALL"],
        help="Observation strategy of training edges.")
    parser.add_argument(
        "-r", "--repetition", dest="rep",
        type=int, default=1,
        help="Remnants repeition identifier.")

    return parser

def verify_args(args):
    # - File input -
    if not os.path.isfile(args.filepath):
        raise FileNotFoundError(f"{args.filepath} not found!")

    # - Numeric parameters -
    if args.theta < 0.0 or args.theta > 1.0:
        raise ValueError("Theta must be between 0.0 and 1.0 (inclusive)!")

    # - String parameters -
    if args.strategy.upper() not in VALID_STRATEGIES:
        raise NotImplementedError(f"{args.strategy} not implemented; choose from {VALID_STRATEGIES}")

    return

def gather_args():
    # Initialize argparse
    parser = setup_argument_parser()

    # Gather command-line arguments
    args = parser.parse_args()

    # Verify arguments are acceptable
    verify_args(args)

    return args

# --- Remnant observation ---
# TODO: Update when alternative observation strategies are supported
def observe_remnants(duplex, theta, strategy):
    logger.warning("`strategy` is currently ignored!")
    remnant_multiplex = observer.partial_information(duplex, theta)

    return remnant_multiplex

# ================= MAIN =======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Separate directory and basename of input edgelist file
    _, basename = os.path.split(args.filepath)

    # Construct filepath
    filepath_ = FILEPATH_TEMPLATE.format(
        strategy=args.strategy,
        theta=args.theta,
        rep=args.rep,
        basename=basename)
    filepath = os.path.join(DIR_REMNANTS, filepath_)

    # Check if file already exists
    if os.path.isfile(filepath):
        logger.info(f"File '{filepath_}' already exists! Skipping remnant observation.")
        return

    # Load edgelists and observe remnants
    with open(args.filepath, 'rb') as _fh:
        duplex = pickle.load(_fh)
    remnant_multiplex = observe_remnants(duplex, args.theta, args.strategy)

    # Save RemnantMultiplex to disk
    remnant_multiplex.save(filepath)


if __name__ == "__main__":
    main()
