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
from scipy.sparse.linalg import eigsh

# Network science
import networkx as nx


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
    "PROJECT_ID": "EMB_ex20",
    "RESEARCHERS": "DK",
    "CURRENT_VERSION": "v1.1",
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
        np.log(calculate_distance_ratio(edge, E_G, E_H))
        for edge in trainset
    ]).reshape(-1, 1)
    distance_ratios_test = np.array([
        np.log(calculate_distance_ratio(edge, E_G, E_H))
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
def get_representation(remnant, parameters):
    # Book-keeping
    ## Indexing objects
    _nodes = sorted(remnant.nodes())  # * Force networkx indexing
    _nodes_reindexing = {node: idx for idx, node in enumerate(_nodes)}  # Allow for non-contiguous node indices

    ## Hyperparams
    dimension = np.array(parameters["dimension"])
    maxiter = len(_nodes)*parameters["maxiter_multiplier"]
    if parameters["tol_exp"] >= 0:
        tol = 0
    else:
        tol = 10**parameters["tol_exp"]
    # Calculate normalized Laplacian
    L_normalized = nx.normalized_laplacian_matrix(remnant, nodelist=_nodes)

    # Account for first eigenvalue correlated with degrees
    dimension += 1
    # Account for algebraic multiplicity of trivial eigenvalues equal to number of connected components
    num_components = nx.number_connected_components(remnant)

    dimension += num_components
    # Calculate eigenspectra
    eigenvalues, eigenvectors = eigsh(
            L_normalized, k=dimension,
            which="SM", maxiter=maxiter, tol=tol,
        )

    # * Ensure algebraic multiplcity of trivial eigenvalue matches num_components
    # TODO: Fix calculation
    # for idx, w_ in enumerate(w):
    #     trivial_ = sum([np.isclose(val, 0) for val in w_])
    #     components_ = num_components[idx]
    #     if trivial_ != components_:
    #         raise ValueError(
    #             f"""Number of components and algebraic multiplicity
    #             of trivial eigenvalue do not match in remnant layer {idx}
    #             Found {components_} components, {trivial_} near-0 eigenvalues
    #             {w_}
    #             """
    #             )
    # Retrieve eigenvectors and first non-trivial dimension-many components
    eigenvectors = np.array([
            vector[-parameters["dimension"]:]
            for vector in eigenvectors
        ])

    indexed_eigenvectors = {node: eigenvectors[idx] for idx, node in enumerate(_nodes)}

    return indexed_eigenvectors


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
    maxiter_multiplier=100,
    tol_exp=-8,
    penalty="l2",
    theta_min=0.05,
    theta_max=0.5,
    theta_num=10,
    repeat=5
        ):
    parameters = {
        # >>> LE embedding <<<
        "dimension": dimensions,
        "maxiter_multiplier": maxiter_multiplier,
        "tol_exp": tol_exp,

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
            records.extend(main(system, l1, l2, repeat=5, theta_num=11))

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"../../results/dataframes/dataframe_{TAG}.csv", index_label="_UID")
