"""Experiment script to
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
from classifiers import logreg
from data import dataio, preprocessing
from distance.score import component_penalized_embedded_edge_distance_ratio
from embed import N2V, LE
from sampling.random import partial_information

# Utilities
from utils import parameters as params

# --- Miscellaneous ---
from datetime import datetime
from tqdm.auto import tqdm, trange
from tabulate import tabulate


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
        "coef": -1.0,
        "intercept": -1.0
    }  # initialize record
    # <<< Book-keeping <<<

    # >>> Procedure >>>
    # * Step (1) - # Bring duplex into memory
    D = dataio.read_file(f"../../data/input/raw/duplex_system={system}.edgelist")
    G, H = preprocessing.duplex_network(D, *layer_pair)

    # * Steps (2) thru (4) - Observe a priori information and calculate remnants
    R_G, R_H, testset, trainset = partial_information(G, H, parameters["theta"])

    # * Step (5) - Embed remnants
    E_G = LE.LE(R_G, parameters, hyperparameters)
    E_H = LE.LE(R_H, parameters, hyperparameters)

    # * Steps (6) and (7) - Calculate distances of nodes incident to edges in both embeddings
    distance_ratios_train = np.array([
        component_penalized_embedded_edge_distance_ratio(
            edge,
            R_G, R_H,
            E_G, E_H,
        )
        for edge in trainset
    ])
    distance_ratios_test = np.array([
        component_penalized_embedded_edge_distance_ratio(
            edge,
            R_G, R_H,
            E_G, E_H,
        )
        for edge in testset
    ])
    distance_ratios_train = np.log(distance_ratios_train)
    distance_ratios_test = np.log(distance_ratios_test)
    distance_ratios_train = distance_ratios_train.reshape(-1, 1)
    distance_ratios_test = distance_ratios_test.reshape(-1, 1)

    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Step (8) - Train a logistic regression
    model = logreg.train_fit_logreg(distance_ratios_train, labels_train)

    # * Step (9) - Predict testset with reconstruction
    record["accuracy"] = logreg.get_model_accuracy(model, distance_ratios_test, labels_test)
    record["auroc"] = logreg.get_model_auroc(model, distance_ratios_test, labels_test)
    record["intercept"], record["coef"] = logreg.get_model_fit(model)
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
                    try:
                        record = experiment(system, layer_pair, parameters, hyperparameters)
                        records.append(record)
                    except TypeError:
                        print(f"{system}({layer_pair[0]}, {layer_pair[1]}) - theta = {theta} \t remnant too sparse, k >= N")
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
        "PROJECT_ID": "EMB_ex21",
        "RESEARCHERS": "DK",
        "CURRENT_VERSION": "v2.1",
        "DATE": datetime.today().strftime("%Y%m%d")
    }
    TAG = "{PROJECT_ID}{CURRENT_VERSION}_{RESEARCHERS}_{DATE}".format(**metadata)
    output_filehandle = f"../../results/dataframes/dataframe_{TAG}.csv"

    # Parameter ranges
    systems = {
        # "arxiv": [(2, 6), (2, 7), (6, 7)],
        "celegans": [(1, 2), (1, 3), (2, 3)],
        # "drosophila": [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        # "london": [(1, 2), (1, 3), (2, 3)],
    }
    parameters, hyperparameters = params.set_parameters_LE(theta_num=10)
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    df = main(systems, parameters, hyperparameters, output_filehandle)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # <<< Experiment <<<
