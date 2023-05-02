"""Script to precompute and cache observational data for reconstruction experiments.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import argparse

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
from src.data import preprocessing, observations, dataio
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

    return parser


def main():
    # >>> SETUP >>>
    # Setup argument parser
    print("Setting up argument parser...")
    parser = _setup_argument_parser()

    # Parse arguments according to parser
    print("Parsing arguments...")
    args = parser.parse_args()

    # Shared system setup
    systems = [
        ("arxiv", 2, 6, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("arxiv", ROOT=ROOT).format(system="arxiv")), *[2, 6])),
        # ("celegans", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("celegans", ROOT=ROOT).format(system="celegans")), *[1, 2])),
        ("drosophila", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("drosophila", ROOT=ROOT).format(system="drosophila")), *[1, 2])),
        # ("london", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("london", ROOT=ROOT).format(system="london")), *[1, 2]))
    ]
    # <<< SETUP <<<

    # >>> CACHING EMBEDDINGS >>>
    # Setup caching hyperparameters
    reps = range(1, int(Config.REPS.value)+1)
    thetas = np.linspace(Config.THETA_MIN.value, Config.THETA_MAX.value, int(Config.THETA_NUM.value), endpoint=True)
    grid = list(product(systems, thetas, reps))
    print(f"Discovered {len(grid)} parameter grid vertices")

    # Select embedder and parameters
    if args.EMBEDDING == "N2V":
        raise NotImplementedError("N2V real system caching has not been refactored in this script yet!")
    elif args.EMBEDDING == "LE":
        parameters, hyperparameters, _ = params.set_parameters_LE(dimensions=args.dimensions, maxiter=100_000, tol=-4)

    for system_, theta, rep in tqdm(grid, desc="Caching observational data..."):
        system, l1, l2, (G, H) = system_

        observations.calculate_preprocessed_data(
            G, H,
            system, [l1, l2],
            theta, rep,
            parameters, hyperparameters["embedding"],
            EMBEDDING=args.EMBEDDING,
            ROOT="preprocessed/"
        )
    # <<< CACHING EMBEDDINGS <<<

    print("Finished caching! Have a nice day.")


# ==================== MAIN =======================
if __name__ == "__main__":
    main()
