#!/usr/bin/env python
"""Script to embed a remnant and cache the embedding to disk.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import argparse
from enum import Enum

# --- Scientific computing ---

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
## Data

# --- Miscellaneous ---

# --- Globals ---
## Exit status
class Status(Enum):
    OK = 0
    FILE = 1
    PARAM = 2
    OTHER = 3


# ================ FUNCTIONS ======================
# --- Command-line Interface ---
def _setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Embed remnant duplex and save to disk.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument(
        "filepath",
        type=str,
        help="Filepath of pickled remnant duplex.")
    parser.add_argument(
        "EMBEDDING",
        choices=["N2V", "LE"],
        help="Embedding method.")
    parser.add_argument(
        "dimensions",
        type=int,
        help="Embedding dimension.")
    parser.add_argument(
        "--percomponent",
        action="store_true",
        help="Flag indicating component-wise embedding.")

    # Optional arguments, flag
    parser.add_argument(
        "-w", "--walklength", dest="walk_length",
        type=int, default=30,
        help="N2V Walk Length.")
    parser.add_argument(
        "--repeat", dest="reps",
        type=int, default=1,
        help="Number of embeddings to gather on given remnant duplex.")

    return parser

def _verify_args(args):
    # - File input -
    if not os.path.isfile(args.filepath):
        raise FileNotFoundError(f"{args.filepath} not found!")

    # - Numeric parameters -
    _fp_after_N = args.filepath.split("N-")[1]
    N = int(_fp_after_N.split("_")[0])
    if args.dimensions <= 0 or args.dimensions >= N:
        raise ValueError("The number of dimensions must be positive and not exceed N")
    if args.reps <= 0:
        raise ValueError("Number of repetitions must be a positive integer!")
    if args.walk_length <= 0 or args.walk_length >= N:
        raise ValueError("Walk length must be positive and not exceed N!")

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
    except ValueError as err:
        print(str(err), file=sys.stderr)
        quit(Status.PARAM.value)
    except:
        print(str(err), file=sys.stderr)
        quit(Status.OTHER.value)

    return args

# --- File I/O ---
# --- Embedding calculation ---

# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Main logic


if __name__ == "__main__":
    main()