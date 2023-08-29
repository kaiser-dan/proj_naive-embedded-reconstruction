"""
"""
# =========== SETUP ==========
import os
import time

import numpy as np
import networkx as nx
from tqdm import tqdm

import EMB


# Need to decide on resolution
# systematic review paper used 50ish to 10,000ish (logarithmic) over 10ish steps

# =========== FUNCTIONS ==========
def stopwatch(previous=None):
    time_ = time.time()
    if previous is not None:
        time_ -= previous
        print(time_)

    return time_


# ========== MAIN ===========
def main():
    times = dict()
    for N in np.geomspace(100, 10_000, num=10, endpoint=True):
        # 1 Generate synthetics
        D, _, _, _ = EMB.netsci.models.benchmarks.generate_duplex_LFR(
            N,
            2.1, 1.0, 0.1, 6, 100, 1, ROOT=os.path.join("..", "..", "")
        )
        D = dict(enumerate(EMB.netsci.models.preprocessing.make_layers_disjoint(*D.values())))

        # 2 Observe remnants
        remnants = EMB.remnants.observer.random_observations_multiplex(D, 0.3)

        # 3 Embed multiplex - timed
        timer_emb = stopwatch()
        vectors = EMB.embeddings.embed_multiplex_N2V(remnants, dimensions=128)
        timer_emb = stopwatch(timer_emb)

        # 4 Calculate features - timed
        timer_features = stopwatch()
        X
        timer_features = stopwatch(timer_features)

        # 5 Train model - timed
        timer_training = stopwatch()
        X
        timer_training = stopwatch(timer_training)

        # 6 Evaluate model - timed
        timer_eval = stopwatch()
        X
        timer_eval = stopwatch()

        # 7 Bookkeeping
        times[N] = {
            "emb": timer_emb,
            "features": timer_features,
            "training": timer_training,
            "eval": timer_eval,
            "total": timer_emb + timer_features + timer_training + timer_eval
        }

        print(times[N]["total"])

    return None


if __name__ == "":
    print("Beginning workflow profiling...")
    main()
    print("Finished profiling!")