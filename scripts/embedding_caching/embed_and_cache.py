#!/usr/bin/env python
"""Script to embed a remnant and cache the embedding to disk.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import pickle
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
from src.data import caches

# --- Miscellaneous ---

# --- Globals ---
## Exit status
class Status(Enum):
    OK = 0
    FILE = 1
    PARAM = 2
    OTHER = 3

## Filepaths & templates
CACHE_DIR: str = os.path.join(ROOT, "data", "input", "caches", "")
FILEPATH_TEMPLATE: str = "method-{method}_percomponent-{pc}_dim-{dim}_embrep-{rep}_{remnant_filepath}"


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
        "embedding",
        choices=["N2V", "LE", "ISOMAP"],
        help="Embedding method.")
    parser.add_argument(
        "dimensions",
        type=int,
        help="Embedding dimension.")

    # Optional arguments, flag
    parser.add_argument(
        "--percomponent", dest="per_component",
        action="store_true",
        help="Flag indicating component-wise embedding.")
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
    except Exception as err:
        print(str(err), file=sys.stderr)
        quit(Status.OTHER.value)

    return args

# --- File I/O ---
def load_remnant(filepath):
    with open(filepath, "rb") as _fh:
        remnant_duplex = pickle.load(_fh)

    return remnant_duplex


# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Main logic
    ## Load remnants
    remnants = load_remnant(args.filepath)

    for rep in range(args.reps):
        ## Apply embeddings and form cache
        cache = caches.build_cachedremnants(
            name=remnants[0].name,
            layers=(1, 2),
            remnants=remnants,
            embedder=args.embedding)

        ## Save cache to disk
        basename = FILEPATH_TEMPLATE.format(
            method=args.embedding,
            pc=args.per_component,
            dim=args.dimensions,
            rep=rep,
            remnant_filepath=os.path.basename(args.filepath)
        )
        cache.save(os.path.join(CACHE_DIR, basename))


if __name__ == "__main__":
    # ! >>> EMPTY FILENAME BUG HOT-FIX >>>
    if len(sys.argv[1]) < 3:
        quit()
    # ! <<< EMPTY FILENAME BUG HOT-FIX <<<

    main()