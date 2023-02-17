"""Experiment script to verify base N2V results.

At the time of creation, fixed bugs in AUROC and N2V parallelism
brought preliminary N2V results into question.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
from sklearn.metrics import accuracy_score, roc_auc_score

# -- Network science ---

# -- Data handling ---
import pandas as pd

# --- Project source ---
# PATH adjustments
sys.path.append("../../src/")

# Primary modules
from data import dataio, preprocessing  # reading data, form duplex
from distance.score import likelihood
from distance.distance import embedded_edge_distance
from embed.N2V import N2V  # graph embeddings
from sampling.random import partial_information  # sample training set

# Utilities
from utils import parameters as params  # helpers for experiment parameters

# --- Miscellaneous ---
from datetime import datetime  # date metadata
from tqdm.auto import tqdm, trange  # progress bars
from tabulate import tabulate  # pretty dataframe printing

import warnings
warnings.filterwarnings("ignore")  # remove sklearn depreciation warning >.>

# ========== FUNCTIONS ==========
def experiment(system, layer_pair, parameters, hyperparameters):
    # >>> Book-keeping >>>
    record = {
        "system": system,
        "l1": layer_pair[0],
        "l2": layer_pair[1],
        "theta": parameters["theta"],
        "accuracy": -1.0,
        "auroc": -1.0,
    }  # initialize record
    # <<< Book-keeping <<<

    # >>> Procedure >>>
    # * Step (1) - Bring duplex into memory
    D = dataio.read_file(f"../../data/input/raw/duplex_system={system}.edgelist")
    G, H = preprocessing.duplex_network(D, *layer_pair)

    # * Steps (2) thru (4) - Observe a priori information and calculate remnants
    R_G, R_H, testset, _ = partial_information(G, H, parameters["theta"])

    # * Step (5) - Embed remnants
    E_G = N2V(R_G, parameters, hyperparameters)
    E_H = N2V(R_H, parameters, hyperparameters)

    # * Steps (6) and (7) - Calculate distances of nodes incident to edges in both embeddings
    distances = {}
    for edge in testset:
        d_G = embedded_edge_distance(edge, E_G)
        d_H = embedded_edge_distance(edge, E_H)

        distances[edge] = (d_G, d_H)

    # Get ground truth
    ground_truth = list(testset.values())

    # Get likelihoods
    likelihoods = {
        edge: likelihood(d_G, d_H)
        for edge, (d_G, d_H) in distances.items()
    }

    # Get hard classifications
    classifications = {
        edge: 1 if likelihood >= 0.5 else 0
        for edge, likelihood in likelihoods.items()
    }

    # * Step (8) - Evaluate performance
    record["accuracy"] = accuracy_score(ground_truth, list(classifications.values()))
    record["auroc"] = roc_auc_score(ground_truth, list(likelihoods.values()))
    # <<< Procedure <<<

    return record

# ========== MAIN ==========
def main(systems, parameters, hyperparameters, output_filehandle=None):
    # >>> Book-keeping >>>
    records = []  # initialize records

    # Build theta range
    thetas = params.build_theta_range(parameters["theta_min"], parameters["theta_max"], parameters["theta_num"])
    del parameters["theta_min"]
    del parameters["theta_max"]
    del parameters["theta_num"]
    # <<< Book-keeping <<<

    # >>> Experiment >>>
    # Loop over all desired induced duplexes
    for system, layers in tqdm(systems.items(), desc="Systems-level progress", position=0, colour="white"):
        for layer_pair in tqdm(layers, desc="Layer pairs", position=1, leave=False, colour="green"):
            # Loop over theta (primary variable)
            for theta in tqdm(thetas, desc="theta", position=2, leave=False, colour="yellow"):
                parameters["theta"] = theta
                # Repeat for statistics
                for _ in trange(hyperparameters["repeat"], desc="Repetitions", position=3, leave=False, colour="red"):
                    record = experiment(system, layer_pair, parameters, hyperparameters)
                    records.append(record)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = pd.DataFrame.from_records(records)
    if output_filehandle is not None:
        df.to_csv(output_filehandle, index_label="_UID")
    # <<< Post-processing <<<

    return df

if __name__ == "__main__":
    # >>> Experiment set-up >>>
    # Metadata
    metadata = {
        "PROJECT_ID": "EMB_ex23-verify",
        "RESEARCHERS": "DK",
        "CURRENT_VERSION": "v1.0",
        "DATE": datetime.today().strftime("%Y%m%d")
    }
    TAG = "{PROJECT_ID}{CURRENT_VERSION}_{RESEARCHERS}_{DATE}".format(**metadata)
    output_filehandle = f"../../results/dataframes/dataframe_{TAG}.csv"

    # Parameter ranges
    systems = {
        "arxiv": [(2, 6), (2, 7), (6, 7)],
        "celegans": [(1, 2), (1, 3), (2, 3)],
        "drosophila": [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        "london": [(1, 2), (1, 3), (2, 3)],
    }
    parameters, hyperparameters = params.set_parameters_N2V(theta_max=1.0)
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")
    df = main(systems, parameters, hyperparameters, output_filehandle)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # <<< Experiment <<<
