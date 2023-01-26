"""Experiment script to attempt logistic regression-based embedding reconstruction en masse."""
# ========== SET-UP ==========
# --- Imports ---
# Stdlib
import sys
import os
import random
from datetime import datetime
from IPython.display import display, Latex

# Scientific computing
import numpy as np
from sklearn.linear_model import LogisticRegression

# Network science
import networkx as nx
from node2vec import Node2Vec as N2V

# Data handling
import pandas as pd

# Project source
sys.path.append("../../src/")
import synthetic
import utils

# Miscellaneous
from tqdm.auto import tqdm

# --- Globals ---
metadata = {
    "PROJECT_ID": "EMB_ex19",
    "RESEARCHERS": "DK",
    "CURRENT_VERSION": "v1.1.1",
    "DATE": datetime.today().strftime("%Y%m%d")
}
TAG = "{PROJECT_ID}{CURRENT_VERSION}_{RESEARCHERS}_{DATE}".format(**metadata)


# ========== FUNCTIONS ==========
# --- Drivers ---
def main(system, l1, l2, **parameters):
    D = utils.read_file(f"../../data/input/raw/duplex_system={system}.edgelist")
    G, H = utils.duplex_network(D, l1, l2)

    parameters = _set_parameters(**parameters)

    records = []
    for theta in tqdm(np.linspace(parameters["theta_min"], parameters["theta_max"], parameters["theta_num"]), desc="theta"):
        for _ in tqdm(range(parameters["repeat"]), desc="Repetitions"):
            try:
                score = workflow(G, H, theta, parameters)


                record = {
                    "system": system,
                    "l1": l1,
                    "l2": l2,
                    "theta": theta,
                    "score": score
                }
                records.append(record)
            except:
                continue

    return records


def workflow(G, H, theta, parameters):
    # * Steps (2) thru (4) - Observe a priori information and calculate remnants
    R_G, R_H, testset, trainset = utils.partial_information(G, H, theta)

    # * Step (5) - Embed remnants
    E_G = get_representation(R_G, parameters)
    E_H = get_representation(R_H, parameters)

    # * Steps (6) and (7) - Calculate distances of nodes incident to edges in both embeddings
    distance_ratios_train = np.array([
        calculate_distance_ratio(edge, E_G, E_H)
        for edge in trainset
    ]).reshape(-1, 1)
    distance_ratios_test = np.array([
        calculate_distance_ratio(edge, E_G, E_H)
        for edge in testset
    ]).reshape(-1, 1)

    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Step (8) - Train a logistic regression
    model = LogisticRegression(random_state=37)
    model.fit(distance_ratios_train, labels_train)

    coef = model.coef_
    intercept = model.intercept_

    # * Step (9) - Predict testset with reconstruction
    score = model.score(distance_ratios_test, labels_test)

    return score


# --- Computations ---
def get_representation(G, parameters):
    """
    Embed a monoplex with node2vec. Wrapper from @Minsuk Kim.
    """
    # Create node2vec model
    n2v = N2V(G,
        dimensions = parameters["dimensions"],
        walk_length = parameters["walk_length"],
        num_walks = parameters["num_walks"],
        workers = parameters["workers"],
        quiet = parameters["quiet"],
    )

    # Embed topology under specified n2v model
    embedding = n2v.fit(
        window = parameters["window"],
        min_count = parameters["min_count"],
        batch_words = parameters["batch_words"],
    )

    # Format resultant vectors
    # ! NOTE: Currently broken for real networks
    # ! >>> Broken >>>
    # * Output format: Array with row [i] corresponding to embedded vector of node i
    # representation = np.array([
    #     embedding.wv['%d' % i]
    #     for i in range(G.number_of_nodes())
    # ])
    # ! <<< Broken <<<

    # ! >>> Hot fix >>>
    embedding = embedding.wv
    index_ = embedding.index_to_key
    vectors_ = embedding.vectors
    representation = {int(index_[idx]): vectors_[idx] for idx in range(G.number_of_nodes())}
    # ! <<< Hot fix <<<

    return representation


def calculate_distances(edge, E_G, E_H):
    # Retrieve nodes incident to edge
    i, j = edge

    # Calculate distance between incident nodes in both embeddings
    d_G = np.linalg.norm(E_G[i] - E_G[j])
    d_H = np.linalg.norm(E_H[i] - E_H[j])

    return d_G, d_H


def calculate_distance_ratio(edge, E_G, E_H):
    d_G, d_H = calculate_distances(edge, E_G, E_H)
    if d_H == 0 or d_G == 0:
        return 1
    else:
        return d_G / d_H


# -- Helpers ---
def _set_parameters(
    dimensions=128,
    walk_length=30,
    num_walks=100,
    workers=4,
    quiet=True,
    window=10,
    min_count=1,
    batch_words=4,
    penalty="l2",
    theta_min=0.05,
    theta_max=0.5,
    theta_num=10,
    repeat=5
        ):
    parameters = {
        # >>> Node2Vec embedding <<<
        "dimensions": dimensions,  # euclidean dimension to embedd
        "walk_length": walk_length,  # number of nodes in each walk
        "num_walks": num_walks,  # number of walks per node
        "workers": workers,  # for cpu parallel work
        "quiet": quiet,  # verbose printing
        "window": window,  # maximum distance between the current and predicted word within a sentence.
        "min_count": min_count,  # ignores all words with total frequency lower than this
        "batch_words": batch_words,  # [unsure]

        # >>> Logistic regression <<<
        "penalty": penalty,  # L2 regularization

        # >>> Simulations <<<
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters

# ========== MAIN ==========
if __name__ == "__main__":
    records = []
    for system in ["celegans", "london"]:
        for l1, l2 in [(1,2), (1,3), (2,3)]:
            print("="*16 + f" {system}-{l1}-{l2} " + "="*16)
            records.extend(main(system, l1, l2, workers=24, repeat=3, theta_num=6))

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"dataframe_{TAG}.csv", index_label="_UID")
