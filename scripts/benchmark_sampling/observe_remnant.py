#!/usr/bin/env python
"""Script to sample an observed training set from an duplex.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import argparse
import pickle
from enum import Enum

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
from src.sampling.random import partial_information
from src.data.io import read_file

# --- Globals ---
## Exit status
class Status(Enum):
    OK = 0
    FILE = 1
    PARAM = 2
    OTHER = 3

## Fixed parameters
# TODO: Find more robust way to track implemented sampling techniques
VALID_STRATEGIES = ["RANDOM"]

## Filepaths & templates
FILEPATH_TEMPLATE: str = "theta-{theta}_strategy-{strategy}_remrep-{rep}_{edgelist_filepath}"


# ================= FUNCTIONS =======================
# --- Command-line Interface ---
def _setup_argument_parser():
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
        "--strategy", dest="strategy",
        type=str, default="RANDOM",
        help="Observation strategy of training edges.")
    parser.add_argument(
        "--repeat", dest="reps",
        type=int, default=1,
        help="Number of remnants to gather on given multiplex.")

    # Plain-text input
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Expect plain-text edgelist file.")
    parser.add_argument(
        "--layers", nargs=2,
        help="Layers of plain-text multiplex from which to induce a duplex."
    )

    return parser

def _verify_args(args):
    # - File input -
    if not os.path.isfile(args.filepath):
        raise FileNotFoundError(f"{args.filepath} not found!")

    # - Numeric parameters -
    if args.theta < 0.0 or args.theta > 1.0:
        raise ValueError("Theta must be between 0.0 and 1.0 (inclusive)!")
    if args.reps <= 0:
        raise ValueError("Number of repetitions must be a positive integer!")

    # - String parameters -
    if args.strategy.upper() not in VALID_STRATEGIES:
        raise NotImplementedError(f"{args.strategy} not implemented; choose from {VALID_STRATEGIES}")

    return

def gather_args():
    _parser = _setup_argument_parser()
    args = _parser.parse_args()

    # Verify integrity of arguments
    try:
        _verify_args(args)
    except FileNotFoundError as err:
        print(str(err), file=sys.stderr)
        quit(Status.FILE.value)
    except (ValueError, NotImplementedError) as err:
        print(str(err), file=sys.stderr)
        quit(Status.PARAM.value)
    except:
        print(str(err), file=sys.stderr)
        quit(Status.OTHER.value)

    return args

# --- File I/O ---
def load_duplex(filepath):
    with open(filepath, "rb") as _fh:
        duplex = pickle.load(_fh)

    return duplex

def save_data(data, filepath):
    # Initialize passing exist status
    status = Status.OK.value

    # Open binary file handle and dump object
    try:
        filehandle = open(filepath, "wb")
        pickle.dump(data, filehandle, pickle.HIGHEST_PROTOCOL)
    # Note exceptions if they occur
    except Exception as err:
        print(str(err), file=sys.stderr)
        status = Status.FILE.value
    # Regardless of status, close file stream
    finally:
        filehandle.close()

    # If non-OK status, force quit with status
    if status:
        quit(status)

    return

# --- Remnant observation ---
# TODO: Update when alternative observation strategies are supported
def observe_remnant(duplex, theta, strategy):
    G, H = duplex
    rem_G, rem_H, _, _ = partial_information(G, H, theta)

    return rem_G, rem_H

# ================= MAIN =======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Main logic
    edgelist_dir, edgelist_filepath = os.path.split(args.filepath)

    if args.plain:
        l1, l2 = [int(l) for l in args.layers]
        duplex = read_file(args.filepath)
        duplex = [duplex[l1], duplex[l2]]
        edgelist_filepath = os.path.splitext(edgelist_filepath)[0] + f"_l1-{l1}_l2-{l2}" + os.path.splitext(edgelist_filepath)[1]
    else:
        duplex = load_duplex(args.filepath)

    for rep in range(args.reps):
        filepath = os.path.join(
            edgelist_dir,
            "remnants_" + FILEPATH_TEMPLATE.format(
                theta=args.theta,
                strategy=args.strategy,
                rep=rep,
                edgelist_filepath=edgelist_filepath)
        )

        if os.path.isfile(filepath):
            print(f"File '{filepath}' already exists! Skipping")
            return

        rem_G, rem_H = observe_remnant(duplex, args.theta, args.strategy)

        save_data((rem_G, rem_H), filepath)

    return

if __name__ == "__main__":
    main()
