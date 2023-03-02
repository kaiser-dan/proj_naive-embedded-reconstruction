"""Experiment script to explore incrementally more sophisticated
feature sets to logistic regression with N2V embeddings.

Broadly speaking, we have the following "workflow":

1. main() -> Sweep over _data_ (Systems \& induced duplexes)
2. experiment() -> For single data instance, sweep over _parameters_ (Theta, classifier parameters, etc.)
3. reconstruct() -> For single data instance and parameter instance, actually _do_ reconstruction task.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

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
def reconstruct(G, H, theta, parameters, hyperparameters, record):
    # >>> Book-keeping >>>
    # Start timers
    start_wall_time = time()  # start wall timer
    start_time = perf_counter()  # start process timer
    # <<< Book-keeping <<<

    # >>> Reconstruction procedure >>>
    R_G, R_H, testset, trainset = partial_information(G, H, theta)

    # * Step (4) - Embed remnants
    E_G = N2V(R_G, parameters, hyperparameters)
    E_H = N2V(R_H, parameters, hyperparameters)

    # * Steps (5) - Calculate features
    vectors = (E_G, E_H)
    graphs = (R_G, R_H)

    ## Training features
    edges = list(trainset.keys())
    configuration_degrees = \
        features.get_configuration_probabilities_feature(
            *features.get_degrees(graphs, edges)
        )
    distances = features.get_distances(vectors, edges)
    feature_matrix_train = features.prepare_feature_matrix_confdeg_dist(configuration_degrees, distances)

    ## Test features
    edges = list(testset.keys())
    configuration_degrees = \
        features.get_configuration_probabilities_feature(
            *features.get_degrees(graphs, edges)
        )
    distances = features.get_distances(vectors, edges)
    feature_matrix_test = features.prepare_feature_matrix_confdeg_dist(configuration_degrees, distances)

    ## Retrieve labels for sklearn model
    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Step (6) - Train logistic regression classifier

    try:
        model = logreg.train_fit_logreg(feature_matrix_train, labels_train, hyperparameters["classifier"])
    except ValueError:  # when only one class is available, happens for some london cases
        return record

    # * Step (7) - Reconstruct duplex with trained classifier
    try:
        accuracy = logreg.get_model_accuracy(model, feature_matrix_test, labels_test)
        auroc = logreg.get_model_auroc(model, feature_matrix_test, labels_test)
        aupr = logreg.get_model_aupr(model, feature_matrix_test, labels_test)
    except ValueError:  # only one class available, fricken London crap
        return record

    # >>> Post-processing >>>
    # Stop timers
    end_time = perf_counter()
    end_wall_time = time()

    # Update record
    record.update({
        "theta": theta,
        "accuracy": accuracy,
        "auroc": auroc,
        "aupr": aupr,
        "process_time": end_time - start_time,
        "wall_time": end_wall_time - start_wall_time
    })
    # <<< Post-processing <<<

    return record

def experiment(system, layer_pair, parameters, hyperparameters, experiment_setup):
    # >>> Book-keeping >>>
    # Specify input data filehandle
    input_data = dataio.get_input_filehandle(system, ROOT=ROOT)

    # Initialize empty record
    empty_record = {
        "system": system,
        "l1": layer_pair[0],
        "l2": layer_pair[1],
        "theta": -np.inf,
        "accuracy": -np.inf,
        "auroc": -np.inf,
        "aupr": -np.inf,
        "process_time": np.inf,
        "wall_time": np.inf
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
        record = reconstruct(G, H, theta, parameters, hyperparameters, empty_record.copy())
        records.append(record)
    # <<< Sweep theta >>>

    return records

# ========== MAIN ==========
def main(systems, parameters, hyperparameters, experiment_setup, output_filehandle=None):
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
            ## Repeat for statistics
            progbar_repetitions = trange(experiment_setup["repeat"], desc="Repetitions", position=2, leave=False, colour="yellow")
            for _ in progbar_repetitions:
                records_ = experiment(system, layer_pair, parameters, hyperparameters, experiment_setup)
                records.extend(records_)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = postprocessing.df_from_records(records)
    # <<< Post-processing <<<

    # ! >>> DEBUG >>>
    print(tabulate(df[["system", "theta", "accuracy", "auroc", "aupr"]], headers='keys', tablefmt='psql'))
    # ! <<< DEBUG <<<

    dataio.save_df(df, output_filehandle)

    return

if __name__ == "__main__":
    # >>> Experiment set-up >>>
    output_filehandle, TAG = \
        dataio.get_output_filehandle(
            PROJECT_ID="EMB_ex28",
            CURRENT_VERSION="v1.0-FI",
            ROOT=ROOT
        )

    # Parameter ranges
    systems = {
        "arxiv": [(2, 6)],
        "celegans": [(1, 2)],
        "drosophila": [(1, 2)],
        "london": [(1, 2)],
    }
    parameters, hyperparameters, experiment_setup = \
        params.set_parameters_N2V(
            # N2V
            workers=48,
            # LogReg
            fit_intercept=True,
            # Other
            theta_min=0, theta_max=0.9, theta_num=10, repeat=5)
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")
    start_wall_time = time()
    start_time = perf_counter()

    main(systems, parameters, hyperparameters, experiment_setup, output_filehandle)

    end_time = perf_counter()
    end_wall_time = time()
    print(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")
    print("\n", "="*60, "\n\n")
    # <<< Experiment <<<
