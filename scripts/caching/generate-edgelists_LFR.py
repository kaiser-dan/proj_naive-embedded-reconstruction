#!/usr/bin/env python
"""Script to generate a corpus of LFR networks with varying mesoscale properties.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import pickle
from itertools import product

# --- Scientific computing ---
import numpy as np

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
DATA_EDGELISTS = os.path.join(*[ROOT, "data", "input", "edgelists", ""])
DATA_PARTITIONS = os.path.join(*[ROOT, "data", "input", "partitions", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
from src.data.benchmarks import generate_multiplex_LFR
from src.data.preprocessing import duplex_network

# --- Miscellaneous ---
from tqdm.auto import tqdm

# ================= MAIN =======================
def main():
    MIN_COMMUNITY: int = 1
    FILEPATH_TEMPLATE_EDGELISTS: str = \
        "egdelists_name-LFR_N-{N}_T1-{t1}_T2-{t2}_kavg-{kavg}-{kmax}_mu-{mu}_prob-{prob}_rep-{rep}.pkl"
    FILEPATH_TEMPLATE_PARTITIONS: str = \
        "partitions_name-LFR_N-{N}_T1-{t1}_T2-{t2}_kavg-{kavg}-{kmax}_mu-{mu}_prob-{prob}_rep-{rep}.pkl"

    _N = np.linspace(100, 10_000, num=10, dtype=int)
    _mu = np.linspace(0.1, 0.5, num=5)
    _t1 = np.linspace(1.0, 4.0, num=4, dtype=int)
    _t2 = np.linspace(2.1, 3.5, num=5)
    _kavg = np.linspace(6, 20, num=6, dtype=int)
    _prob = np.linspace(0, 1, num=5)
    _rep = np.arange(1, 11, dtype=int)

    parameter_grid = product(_N, _mu, _t1, _t2, _kavg, _prob, _rep)
    parameter_grid_size = len(_N)*len(_mu)*len(_t1)*len(_t2)*len(_kavg)*len(_prob)*len(_rep)

    for paramaters in tqdm(parameter_grid, total=parameter_grid_size, desc="Sampling LFRs...", colour="blue"):
        N, mu, t1, t2, kavg, prob, rep = paramaters
        kmax = int(np.sqrt(N))
        if kavg >= kmax:
            continue

        D, sigma1, sigma2, _ = \
            generate_multiplex_LFR(
                N,
                t1, t2,
                mu,
                kavg, kmax,
                MIN_COMMUNITY,
                prob,
                ROOT=ROOT
            )

        G, H = duplex_network(D, 1, 2)

        _error_flag = False
        try:
            fh = open(
                DATA_EDGELISTS + FILEPATH_TEMPLATE_EDGELISTS.format(
                    N=N, t1=t1, t2=t2, kavg=kavg, kmax=kmax, mu=mu, prob=prob, rep=rep
                ),
                "wb"
            )
            pickle.dump((G, H), fh, pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            sys.stderr.write(str(err)+"\n")
            _error_flag = True
        finally:
            fh.close()
            if _error_flag:
                quit(1)

        _error_flag = False
        try:
            fh = open(
                DATA_PARTITIONS + FILEPATH_TEMPLATE_PARTITIONS.format(
                    N=N, t1=t1, t2=t2, kavg=kavg, kmax=kmax, mu=mu, prob=prob, rep=rep
                ),
                "wb"
            )
            pickle.dump((sigma1, sigma2), fh, pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            sys.stderr.write(str(err)+"\n")
            _error_flag = True
        finally:
            fh.close()
            if _error_flag:
                quit(1)

    return

if __name__ == "__main__":
    main()
