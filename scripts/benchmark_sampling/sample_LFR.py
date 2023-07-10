#!/usr/bin/env python
"""Script to sample an LFR duplex with specified parameters.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import argparse
import pickle

# --- Scientific computing ---
import numpy as np

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
from src.data.benchmarks import generate_multiplex_LFR
from src.data.preprocessing import duplex_network

# --- Globals ---
## Fixed parameters
MIN_COMMUNITY: int = 1
REPS: int = 10

## Filepaths & templates
DATA_EDGELISTS = os.path.join(*[ROOT, "data", "input", "edgelists", ""])
DATA_PARTITIONS = os.path.join(*[ROOT, "data", "input", "partitions", ""])
FILEPATH_TEMPLATE: str = \
    "name-LFR_N-{N}_T1-{t1}_T2-{t2}_kavg-{kavg}-kmax-{kmax}_mu-{mu:.1f}_prob-{prob}_rep-{rep}.pkl"


# ================= FUNCTIONS =======================
def _setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Sample a single LFR duplex instance.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

def _verify_args(args, kmax):
    if args.AVG_K >= kmax:
        raise ValueError("Average degree exceeds maximum degree!")

def get_duplexes_and_partitions(args, kmax, reps):
    duplexes = []
    partitions = []

    for _ in range(reps):
        # Generate naive LFR duplex
        D, sigma1, sigma2, _ = \
            generate_multiplex_LFR(
                args.N,
                args.T1, args.T2,
                args.MU,
                args.AVG_K, kmax,
                MIN_COMMUNITY,
                args.PROB,
                ROOT=ROOT
            )

        # Preprocess duplex
        G, H = duplex_network(D, 1, 2)

        duplexes.append((G, H))
        partitions.append((sigma1, sigma2))

    return duplexes, partitions

def save_data(args, kmax, rep, data, edgelist=True):
    if edgelist:
        DIR = DATA_EDGELISTS
        NAME = "edgelists_"
    else:
        DIR = DATA_PARTITIONS
        NAME = "partitions_"

    _error_flag = False
    try:
        filehandle = open(
            DIR + NAME + FILEPATH_TEMPLATE.format(
                N=args.N,
                t1=args.T1,
                t2=args.T2,
                kavg=args.AVG_K,
                kmax=kmax,
                mu=args.MU,
                prob=args.PROB,
                rep=rep
            ),
            "wb"
        )
        pickle.dump(data, filehandle, pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        sys.stderr.write(str(err)+"\n")
        _error_flag = True
    finally:
        filehandle.close()
        if _error_flag:
            print("Exiting with status 1!")
            quit(1)

# ================= MAIN =======================
def main():
    # Parse CL arguments
    _parser = _setup_argument_parser()
    args = _parser.parse_args()
    kmax = int(np.sqrt(args.N))

    # Verify integrity of arguments
    try:
        _verify_args(args, kmax)
    except ValueError as err:
        print(str(err))
        return
    except:
        print("Unknown error; exiting with status 1!")
        quit(1)

    # Sample LFR duplexes
    duplexes, partitions = get_duplexes_and_partitions(args, kmax, REPS)

    # Save each duplex to disk
    for rep in range(REPS):
        save_data(args, kmax, rep+1, duplexes[rep], True)
        save_data(args, kmax, rep+1, partitions[rep], False)

if __name__ == "__main__":
    main()
