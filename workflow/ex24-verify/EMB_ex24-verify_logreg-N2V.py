"""Experiment script to verify base N2V results.

At the time of creation, fixed bugs in AUROC and N2V parallelism
brought preliminary N2V results into question.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# -- Network science ---

# -- Data handling ---
import pandas as pd

# --- Project source ---
# PATH adjustments
sys.path.append("../../src/")

# Primary modules
from classifiers import logreg  # logistic regression
from data import dataio, preprocessing  # reading data, form duplex
from distance.score import component_penalized_embedded_edge_distance_ratio, format_distance_ratios  # Feature calculations for logreg
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
    R_G, R_H, testset, trainset = partial_information(G, H, parameters["theta"])

    # * Step (5) - Embed remnants
    E_G = N2V(R_G, parameters, hyperparameters)
    E_H = N2V(R_H, parameters, hyperparameters)

    # * Steps (6) and (7) - Calculate distances of nodes incident to edges in both embeddings
    distance_ratios_train = -1 * np.ones(len(trainset))
    distance_ratios_test = -1 * np.ones(len(testset))
    for idx, edge in enumerate(trainset.keys()):
        distance_ratios_train[idx] = component_penalized_embedded_edge_distance_ratio(
            edge,
            R_G, R_H,
            E_G, E_H,
        )

    for idx, edge in enumerate(testset.keys()):
        distance_ratios_test[idx] = component_penalized_embedded_edge_distance_ratio(
            edge,
            R_G, R_H,
            E_G, E_H,
        )

    # Pre-processing distance ratios for sklearn models
    distance_ratios_train = format_distance_ratios(distance_ratios_train)
    distance_ratios_test = format_distance_ratios(distance_ratios_test)

    # Retrieve labels for sklearn model
    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Step (8) - Train a logistic regression
    try:
        model = logreg.train_fit_logreg(distance_ratios_train, labels_train)
    except ValueError:  # when only one class is available, happens for some london cases
        sys.stderr.write(">>> Only one train/test class available")
        return record

    # * Step (9) - Predict testset with reconstruction
    try:
        record["accuracy"] = logreg.get_model_accuracy(model, distance_ratios_test, labels_test)
        record["auroc"] = logreg.get_model_auroc(model, distance_ratios_test, labels_test)
        record["intercept"], record["coef"] = logreg.get_model_fit(model)
    except ValueError:  # only one class available, fricken London crap
        pass
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
        "PROJECT_ID": "EMB_ex24-verify",
        "RESEARCHERS": "DK",
        "CURRENT_VERSION": "v1.0",
        "DATE": datetime.today().strftime("%Y%m%d")
    }
    TAG = "{PROJECT_ID}{CURRENT_VERSION}_{RESEARCHERS}_{DATE}".format(**metadata)
    output_filehandle = f"../../results/dataframes/dataframe_{TAG}.csv"

    # Parameter ranges
    systems = {
        # "arxiv": [(2, 6), (2, 7), (6, 7)],
        "celegans": [(1, 2), (1, 3), (2, 3)],
        # "drosophila": [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        "london": [(1, 2), (1, 3), (2, 3)],
    }
    parameters, hyperparameters = params.set_parameters_N2V(theta_max=0.95)
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")
    df = main(systems, parameters, hyperparameters, output_filehandle)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # <<< Experiment <<<
