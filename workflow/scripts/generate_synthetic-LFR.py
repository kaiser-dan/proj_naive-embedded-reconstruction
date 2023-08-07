"""CLI to sample an LFR duplex with specified parameters.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import argparse
import pickle

# --- Scientific computing ---
import numpy as np

# --- Source code ---
from embmplxrec.data.benchmarks import generate_multiplex_LFR
from embmplxrec.data.preprocessing import duplex_network
import embmplxrec.utils

# --- Globals ---
## Fixed parameters
MIN_COMMUNITY: int = 1
REPS: int = 10

## Filepaths & templates
ROOT = os.path.join(".", "")
DIR_EDGES = os.path.join(ROOT, "data", "input", "edgelists", "")
DIR_PARTITIONS = os.path.join(ROOT, "data", "input", "partitions", "")
FILEPATH_TEMPLATE: str = \
    "multiplex-LFR_N-{N}_T1-{t1}_T2-{t2}_kavg-{kavg}_kmax-{kmax}_mu-{mu:.1f}_prob-{prob}.pkl"

## Logging
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/generate_synthetic-LFR_{embmplxrec.utils.get_today(time=True)}.log"
)


# ================= FUNCTIONS =======================
# --- CLI parsing ---
def check_file_exists(args):
    # Check if files exist already
    filepath_ = FILEPATH_TEMPLATE.format(**args)
    filepath_edges = f"{DIR_EDGES}/{filepath_}"
    filepath_partitions = f"{DIR_PARTITIONS}/partitions_{filepath_}"
    if os.path.exists(filepath_edges): # or os.path.exists(filepath_partitions):
        logger.info(f"File '{filepath_}' already exists! Skipping LFR sampling.")
        return True, (filepath_edges, filepath_partitions)
    else:
        return False, (filepath_edges, filepath_partitions)

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Sample a single LFR duplex instance.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-N", "--nodes", dest="N",
        type=int, default=10000,
        help="Number of nodes per layer of sampled duplex.")
    parser.add_argument(
        "-u", "--mu", dest="mu",
        type=float, default=0.1,
        help="Mixing parameter of each layer.")
    parser.add_argument(
        "-d", "--t1", dest="t1",
        type=float, default=2.1,
        help="Degree distribution power-law exponent.")
    parser.add_argument(
        "-c", "--t2", dest="t2",
        type=float, default=1.0,
        help="Community size distribution power-law exponent.")
    parser.add_argument(
        "-k", "--avgk", dest="kavg",
        type=float, default=6.0,
        help="Average degree.")
    parser.add_argument(
        "-m", "--maxk", dest="kmax",
        type=int, default=int(np.sqrt(10000)),
        help="Maximum degree.")
    parser.add_argument(
        "-p", "--prob", dest="prob",
        type=float, default=1.0,
        help="Shuffling applied to break correlations.")

    return parser

def verify_args(args):
    if args["kavg"] >= args["kmax"]:
        raise ValueError("Average degree exceeds maximum degree!")

def gather_args():
    # Initialize argparse
    parser = setup_argument_parser()

    # Gather command-line arguments
    args = vars(parser.parse_args())

    # Append calculated parameters to args
    args["kmax"] = int(np.sqrt(args["N"]))

    # Verify arguments are acceptable
    verify_args(args)

    return args


# --- LFR sampling ---
def get_duplexes_and_partitions(args):
    # Generate naive LFR duplex
    D, sigma1, sigma2, _ = \
        generate_multiplex_LFR(
            args["N"],
            args["t1"], args["t2"],
            args["mu"],
            args["kavg"], args["kmax"],
            MIN_COMMUNITY,
            args["prob"],
            ROOT="./"
        )

    # Preprocess duplex
    G, H = duplex_network(D, 1, 2)

    duplex = (G, H)
    partition = (sigma1, sigma2)

    return duplex, partition

# ================= MAIN =======================
# TODO: Implement repetitive sampling
def main():
    # Parse CL arguments
    args = gather_args()

    # Check if file exists
    check, (filepath_edges, filepath_partitions) = check_file_exists(args)
    if check:
        return

    # Sample LFR duplexes
    duplexes, partitions = get_duplexes_and_partitions(args)

    # Save to disk
    with open(filepath_edges, 'wb') as _fh:
        pickle.dump(duplexes, _fh, pickle.HIGHEST_PROTOCOL)
        logger.debug("Synthetic duplex saved to file")

    with open(filepath_partitions, 'wb') as _fh:
        pickle.dump(partitions, _fh, pickle.HIGHEST_PROTOCOL)
        logger.debug("Synthetic duplex partitions saved to file")

if __name__ == "__main__":
    main()
