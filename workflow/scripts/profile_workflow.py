"""
"""
# =========== SETUP ==========
import os
import time

import numpy as np
import networkx as nx
from tqdm import tqdm

import EMB

# =========== FUNCTIONS ==========
def stopwatch(previous=None, verbose=False):
    time_ = time.time()
    if previous is not None:
        time_ -= previous
        if verbose:
            print(time_)

    return time_


# ========== MAIN ===========
def main():
    times = dict()
    print("N,total_time")
    for N in np.geomspace(100, 10_000, num=10, endpoint=True):
        # 1 Generate synthetics
        D, _, _, _ = EMB.netsci.models.benchmarks.generate_duplex_LFR(
            int(N),
            2.1, 1.0, 0.1, 6, 100, 1,
            ROOT=os.path.join("..", "..", "")
        )
        D = dict(enumerate(EMB.netsci.models.preprocessing.make_layers_disjoint(*D.values())))

        # 2 Observe remnants
        remnant_multiplex = EMB.remnants.observer.random_observations_multiplex(D, 0.3)
        remnant_multiplex = EMB.remnants.observer.build_remnant_multiplex(D, remnant_multiplex)

        # 3 Embed multiplex - timed
        timer_emb = stopwatch()
        vectors = EMB.embeddings.embed_multiplex_Isomap(remnant_multiplex, dimensions=128)
        timer_emb = stopwatch(timer_emb)

        # 4 Calculate features - timed
        timer_features = stopwatch()
        testing_edges = remnant_multiplex[-1].edges()
        training_edges = EMB.netsci.all_edges(*remnant_multiplex.values()) - set(testing_edges)
        y_train = {edge: EMB.netsci.find_edge(edge, *D.values())[0] for edge in training_edges}
        y_test = {edge: EMB.netsci.find_edge(edge, *D.values())[0] for edge in testing_edges}
        remnant_multiplex = EMB.utils.cutkey(remnant_multiplex, -1)
        components = {
            label: sorted(nx.connected_components(graph))
            for label, graph in remnant_multiplex.items()
        }
        vectors = EMB.utils.cutkey(vectors, -1)
        for label, vectorset in vectors.items():
            node2id = dict()
            vectors[label] = EMB.embeddings.normalize_vectors(vectorset, components[label], node2id=node2id)
        training_distances = EMB.classifiers.get_distances_feature(vectors.values(), training_edges, training=True)
        testing_distances = EMB.classifiers.get_distances_feature(vectors.values(), testing_edges, training=False)
        training_degrees = EMB.classifiers.get_degrees_feature(remnant_multiplex.values(), training_edges, training=True)
        testing_degrees = EMB.classifiers.get_degrees_feature(remnant_multiplex.values(), testing_edges, training=False)
        X_train = np.array([training_distances, training_degrees]).transpose()
        X_test = np.array([testing_distances, testing_degrees]).transpose()
        timer_features = stopwatch(timer_features)

        # 5 Train model - timed
        timer_training = stopwatch()
        model = EMB.classifiers.train_model(X_train, list(y_train.values()))
        timer_training = stopwatch(timer_training)

        # 6 Evaluate model - timed
        timer_eval = stopwatch()
        accuracy, auroc, pr = EMB.classifiers.evaluate_model(model, X_test, list(y_test.values()))
        timer_eval = stopwatch(timer_eval)

        # 7 Bookkeeping
        times[N] = {
            "emb": timer_emb,
            "features": timer_features,
            "training": timer_training,
            "eval": timer_eval,
            "total": timer_emb + timer_features + timer_training + timer_eval
        }

        print(N,times[N]["total"])

    return None


if __name__ == "__main__":
    print("Beginning workflow profiling...")
    main()
    print("Finished profiling!")
