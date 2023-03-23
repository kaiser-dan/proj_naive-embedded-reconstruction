"""Experiment script to explore one feature logistic regression reconstructions
from layer remnant embeddings

Broadly speaking, we have the following "workflow":

1. main() -> Sweep over _data_ (Systems \& induced duplexes)
2. experiment() -> For single data instance, sweep over _parameters_ (Theta, classifier parameters, etc.)
3. reconstruct() -> For single data instance and parameter instance, actually _do_ reconstruction task.
"""

"""
TDD

1. Load data
2. Fix feature
3. Experiment over PFI with given data and feature
4. Record performance and coefficients
5. Repeat for all fixtures
6. Repeat for all data
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np
from sklearn.metrics import roc_auc_score

# --- Network science ---

# --- Data handling ---

# --- Project source ---
# PATH adjustments
ROOT = "../../../../"
sys.path.append(f"{ROOT}/src/")

# Primary modules
## Data
from data import dataio
from data import preprocessing, postprocessing
from sampling.random import partial_information  # sample training set

## Embedding
from embed.N2V import N2V

## Classifiers
from classifiers import features  # feature set helpers
from classifiers import logreg  # logistic regression

## Utilities
from utils import parameters as params  # helpers for experiment parameters

# --- Miscellaneous ---
from time import perf_counter, time
from tqdm.auto import tqdm, trange  # progress bars
from tabulate import tabulate  # pretty dataframe printing

import warnings
warnings.filterwarnings("ignore")  # remove sklearn depreciation warnings

# ========== FUNCTIONS ==========
def reconstruct(G, H, feature_set, theta, parameters, hyperparameters, record):
    # >>> Book-keeping >>>
    # Start timers
    start_wall_time = time()  # start wall timer
    start_time = perf_counter()  # start process timer
    # <<< Book-keeping <<<

    # >>> Reconstruction procedure >>>
    R_G, R_H, testset, trainset = partial_information(G, H, theta)

    # * Calculate features
    # & Imbalance
    if "imb" in feature_set:
        hyperparameters["classifier"]["fit_intercept"] = True

    # & Embedding
    if "emb" in feature_set:
        E_G = N2V(R_G, parameters, hyperparameters)
        E_H = N2V(R_H, parameters, hyperparameters)

        distances_G_train, distances_H_train = features.get_distances((E_G, E_H), list(trainset.keys()))
        distances_G_test, distances_H_test = features.get_distances((E_G, E_H), list(testset.keys()))

        feature_distances_train = features.get_configuration_distances_feature(distances_G_train, distances_H_train)
        feature_distances_test = features.get_configuration_distances_feature(distances_G_test, distances_H_test)

    # & Degrees
    if "deg" in feature_set:
        src_degrees_train, tgt_degrees_train = features.get_degrees((R_G, R_H), list(trainset.keys()))
        src_degrees_test, tgt_degrees_test = features.get_degrees((R_G, R_H), list(testset.keys()))

        feature_degrees_train = features.get_configuration_probabilities_feature(src_degrees_train, tgt_degrees_train)
        feature_degrees_test = features.get_configuration_probabilities_feature(src_degrees_test, tgt_degrees_test)

    # * Format feature matrix
    if len(feature_set) == 1:
        if feature_set == {"imb"}:  # imbalance alone needs special formatting
            feature_matrix_train = np.array([0]*len(trainset)).reshape(-1,1)
            feature_matrix_test = np.array([0]*len(testset)).reshape(-1,1)
        elif feature_set == {"emb"}:
            feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
            feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
        elif feature_set == {"deg"}:
            feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
            feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif len(feature_set) == 2:
        if "imb" not in feature_set:
            feature_matrix_train = np.empty((len(trainset), 2))
            feature_matrix_train[:, 0] = feature_distances_train
            feature_matrix_train[:, 1] = feature_degrees_train

            feature_matrix_test = np.empty((len(testset), 2))
            feature_matrix_test[:, 0] = feature_distances_test
            feature_matrix_test[:, 1] = feature_degrees_test
        elif "emb" in feature_set:
            feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
            feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
        elif "deg" in feature_set:
            feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
            feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif len(feature_set) == 3:
        feature_matrix_train = np.empty((len(trainset), 2))
        feature_matrix_train[:, 0] = feature_distances_train
        feature_matrix_train[:, 1] = feature_degrees_train

        feature_matrix_test = np.empty((len(testset), 2))
        feature_matrix_test[:, 0] = feature_distances_test
        feature_matrix_test[:, 1] = feature_degrees_test

    # Retrieve labels for sklearn model
    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Train logistic regression classifier
    try:
        model = logreg.train_fit_logreg(feature_matrix_train, labels_train, hyperparameters["classifier"])
    except ValueError as err:  # when only one class is available, happens for some london cases
        print(feature_set)
        print(err)
        print(feature_matrix_train[:3])
        return record


    # * Reconstruct duplex with trained classifier
    try:
        accuracy = logreg.get_model_accuracy(model, feature_matrix_test, labels_test)
        auroc = logreg.get_model_auroc(model, feature_matrix_test, labels_test)
        aupr = logreg.get_model_aupr(model, feature_matrix_test, labels_test)
        intercept, coefs = logreg.get_model_fit(model)
    except ValueError:  # only one class available, fricken London crap
        return record

    # # >>> Post-processing >>>
    # # Stop timers
    end_time = perf_counter()
    end_wall_time = time()

    # Update record
    record.update({
        "theta": theta,
        "intercept": intercept,
        "coefficients": coefs,
        "accuracy": accuracy,
        "auroc": auroc,
        "aupr": aupr,
        "process_time": end_time - start_time,
        "wall_time": end_wall_time - start_wall_time
    })
    # <<< Post-processing <<<

    return record

def experiment(system, feature_set, layer_pair, parameters, hyperparameters, experiment_setup):
    # >>> Book-keeping >>>
    # Specify input data filehandle
    input_data = dataio.get_input_filehandle(system, ROOT=ROOT)

    # Initialize empty record
    empty_record = {
        "system": system,
        "l1": layer_pair[0],
        "l2": layer_pair[1],
        "features": feature_set,
        "theta": None,
        "intercept": None,
        "coefficients": None,
        "accuracy": None,
        "auroc": None,
        "aupr": None,
        "process_time": None,
        "wall_time": None
    }

    # Build theta range
    thetas = params.build_theta_range(experiment_setup)

    # Format CL progress bar
    progbar_thetas = tqdm(thetas, desc="theta", position=3, leave=False, colour="red")

    # Bring duplex into memory
    D = dataio.read_file(input_data.format(system=system))
    G, H = preprocessing.duplex_network(D, *layer_pair)

    # Initialize records (theta sweep)
    records = []
    # <<< Book-keeping <<<

    # >>> Sweep theta >>>
    for theta in progbar_thetas:
        record = reconstruct(G, H, feature_set, theta, parameters, hyperparameters, empty_record.copy())
        records.append(record)
    # <<< Sweep theta >>>

    return records

# ========== MAIN ==========
# TODO: Cleanup nested loops with Cartesian product on parameter basis
def main(systems, feature_sets, parameters, hyperparameters, experiment_setup, output_filehandle=None):
    # >>> Book-keeping >>>
    records = []  # initialize records
    # <<< Book-keeping <<<

    # >>> Experiment >>>
    # Loop over _data_
    ## Systems
    progbar_systems = tqdm(systems.items(), desc="Systems-level progress", position=0, colour="white")
    for system, layers in progbar_systems:
        ## Induced duplexes
        progbar_layers = tqdm(layers, desc="Layer pairs", position=1, leave=False, colour="green")
        for layer_pair in progbar_layers:
            ## Feature sets
            progbar_featuresets = tqdm(feature_sets, desc="Feature sets", position=2, leave=False, colour="yellow")
            for feature_set in progbar_featuresets:
                ## Repeat for statistics
                progbar_repetitions = trange(experiment_setup["repeat"], desc="Repetitions", position=2, leave=False, colour="red")
                for _ in progbar_repetitions:
                    records_ = experiment(system, feature_set, layer_pair, parameters, hyperparameters, experiment_setup)
                    records.extend(records_)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = postprocessing.df_from_records(records)
    # <<< Post-processing <<<

    dataio.save_df(df, output_filehandle)

    return

if __name__ == "__main__":
    # >>> Experiment set-up >>>
    output_filehandle, TAG = \
        dataio.get_output_filehandle(
            PROJECT_ID="EMB_ex29",
            CURRENT_VERSION="v1.1",
            ROOT=ROOT
        )

    # Parameter ranges
    systems = {
        "arxiv": [(2, 6)],
        # "celegans": [(1, 2)],
        "drosophila": [(1, 2)],
        # "london": [(1, 2)],
    }
    feature_sets = (
        # Single features
        {"imb"},
        {"emb"},
        {"deg"},
        # Feature pairs
        {"imb", "emb"},
        {"imb", "deg"},
        {"emb", "deg"},
        # All features
        {"imb", "emb", "deg"}
    )
    parameters, hyperparameters, experiment_setup = \
        params.set_parameters_N2V(
            fit_intercept=False,  # logreg
            theta_min=0.05, theta_max=0.95, theta_num=10, repeat=1  # other
        )
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")  # print stdout preface
    start_wall_time = time()  # start timers
    start_time = perf_counter()

    main(systems, feature_sets, parameters, hyperparameters, experiment_setup, output_filehandle)  # run simulations

    end_time = perf_counter()  # lap timers
    end_wall_time = time()
    print(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")  # print stdout postface
    print("\n", "="*60, "\n\n")
    # <<< Experiment <<<
