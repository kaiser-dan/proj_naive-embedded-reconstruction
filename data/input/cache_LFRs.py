"""Script to precompute and cache observational data on synthetic LFR benchmarks
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import argparse

import pickle

from enum import Enum
from itertools import product

# --- Scientific computing ---
import numpy as np

# --- Project source ---
# PATH adjustments
ROOT = "../../"
sys.path.append(f"{ROOT}/")
sys.path.append(f"{ROOT}/src/")

# Source code imports
from src.data import preprocessing, observations, benchmarks
from src.utils import parameters as params

# --- Miscellaneous ---
from tqdm import tqdm

# --- Globals ---
class Config(float, Enum):
    # Observations
    REPS = 1.0  # convert to int
    THETA_MIN = 0.05
    THETA_MAX = 0.95
    THETA_NUM = 37.0  # convert to int


# ================ FUNCTIONS ======================
def _setup_argument_parser():
    parser = argparse.ArgumentParser(description='Cache LFR benchmark observational data.')

    parser.add_argument(
        "EMBEDDING",
        choices=["N2V", "LE"],
        help="Embedding method.")
    parser.add_argument(
        "dimensions",
        type=int,
        help="Embedding dimension.")
    parser.add_argument(
        "-w", "--walklength", dest="walk_length",
        type=int, default=30,
        help="N2V Walk Length.")
    parser.add_argument(
        "-N", "--nodes", dest="N",
        type=int, default=1000,
        help="Number of nodes per layer of sampled duplex.")
    parser.add_argument(
        "-u", "--mu", dest="MU",
        type=float, default=0.1,
        help="Mixing parameter of each layer.")
    parser.add_argument(
        "-d", "--t1", dest="T1",
        type=float, default=2.1,
        help="Degree distribution power-law exponent.")
    parser.add_argument(
        "-c", "--t2", dest="T2",
        type=float, default=1.0,
        help="Community size distribution power-law exponent.")
    parser.add_argument(
        "-k", "--avgk", dest="AVG_K",
        type=float, default=6.0,
        help="Average degree.")
    parser.add_argument(
        "-m", "--maxk", dest="MAX_K",
        type=int, default=int(np.sqrt(1000)),
        help="Maximum degree.")
    parser.add_argument(
        "-p", "--prob", dest="PROB",
        type=float, default=1.0,
        help="Shuffling applied to break correlations.")

    return parser


def main():
    # >>> SETUP >>>
    # Setup argument parser
    print("Setting up argument parser...")
    parser = _setup_argument_parser()

    # Parse arguments according to parser
    print("Parsing arguments...")
    args = parser.parse_args()

    # Shared filename setup
    filename = "LFR_N-{}_mu-{}_t1-{}_t2-{}_kavg-{}_kmax-{}_prob-{}_dimensions-{}"
    filename = filename.format(*[args.N, args.MU, args.T1, args.T2, args.AVG_K, args.MAX_K, args.PROB, args.dimensions])
    # <<< SETUP <<<

    # >>> SAMPLING MODEL >>>
    # Generate network
    print("Generating benchmark topologies...")
    try:
        filepath = f"raw/synthetic/{filename}.edgelist"
        if os.path.isfile(filepath):
            raise FileExistsError(f"{filepath} already exists!")
        else:
            remnants = preprocessing.duplex_network(
                benchmarks.lfr_multiplex(
                    args.N,
                    args.T1, args.T2,
                    args.MU,
                    args.AVG_K, args.MAX_K,
                    1,  # min_community - ignored downstream
                    args.PROB)[0],
                1, 2
                )
            with open(filepath, "wb") as _fh:
                pickle.dump(remnants, _fh, pickle.HIGHEST_PROTOCOL)
    except FileExistsError as err:
        print(err, "Continuing to next step.")
        with open(filepath, "rb") as _fh:
            remnants = pickle.load(_fh)
    finally:
        print("Caching observations...")
    # <<< SAMPLING MODEL <<<

    # >>> CACHING EMBEDDINGS >>>
    # Setup caching hyperparameters
    reps = range(1, int(Config.REPS.value)+1)
    thetas = np.linspace(Config.THETA_MIN.value, Config.THETA_MAX.value, int(Config.THETA_NUM.value), endpoint=True)
    grid = list(product(thetas, reps))
    print(f"Discovered {len(grid)} parameter grid vertices")

    # Select embedder and parameters
    if args.EMBEDDING == "N2V":
        raise NotImplementedError("N2V LFR caching has not been refactored in this script yet!")
    elif args.EMBEDDING == "LE":
        parameters, hyperparameters, _ = params.set_parameters_LE(dimensions=args.dimensions)

        # TODO: Check if cache already exists
        for theta, rep in tqdm(grid, desc="Caching system across theta/rep"):
            observations.calculate_preprocessed_data(
                *remnants,
                filename, [1, 2],
                theta, rep,
                parameters, hyperparameters["embedding"],
                args.EMBEDDING,
                ROOT="preprocessed/"
            )
    # <<< CACHING EMBEDDINGS <<<

    print("Finished caching! Have a nice day.")


# ================ MAIN ======================
if __name__ == "__main__":
    main()
