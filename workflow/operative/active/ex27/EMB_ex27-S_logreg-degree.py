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
from data import benchmarks
from data import preprocessing, postprocessing
from sampling.random import partial_information  # sample training set

## Embedding

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
# ========== GLOBALS ===========
from enum import IntEnum
class Constants(IntEnum):
    GAMMA = 3
    KMIN = 3
    N = 1000
    KRANGE_MIN = 10
    KRANGE_MAX = 100
    KRANGE_NUM = 10

# ========== FUNCTIONS ==========
def reconstruct(G, H, theta, hyperparameters, record):
    # >>> Book-keeping >>>
    # Start timers
    start_wall_time = time()  # start wall timer
    start_time = perf_counter()  # start process timer
    # <<< Book-keeping <<<

    # >>> Reconstruction procedure >>>
    R_G, R_H, testset, trainset = partial_information(G, H, theta)

    # * Steps (5) - Calculate features
    # Training features
    src_degrees, tgt_degrees = features.get_degrees((R_G, R_H), list(trainset.keys()))
    feature_matrix_train = features.get_configuration_probabilities_feature(src_degrees, tgt_degrees)
    feature_matrix_train = np.array(feature_matrix_train).reshape(-1, 1)

    # Test features
    src_degrees, tgt_degrees = features.get_degrees((R_G, R_H), list(testset.keys()))
    feature_matrix_test = features.get_configuration_probabilities_feature(src_degrees, tgt_degrees)
    feature_matrix_test = np.array(feature_matrix_test).reshape(-1, 1)

    # Retrieve labels for sklearn model
    labels_train = list(trainset.values())
    labels_test = list(testset.values())

    # * Step (6) - Train logistic regression classifier
    try:
        model = logreg.train_fit_logreg(feature_matrix_train, labels_train, hyperparameters["classifier"])
    except ValueError:  # when only one class is available, happens for some london cases
        return record

    intercept, coefs = logreg.get_model_fit(model)
    try:
        assert_ = (intercept[0] != 0.0) if hyperparameters["classifier"]["fit_intercept"] else (intercept[0] == 0.0)
        assert assert_
    except AssertionError as err:
        print(err.message)

    # # * Step (7) - Reconstruct duplex with trained classifier
    try:
        accuracy = logreg.get_model_accuracy(model, feature_matrix_test, labels_test)
        auroc = logreg.get_model_auroc(model, feature_matrix_test, labels_test)
        aupr = logreg.get_model_aupr(model, feature_matrix_test, labels_test)
    except ValueError:  # only one class available, fricken London crap
        return record

    # # >>> Post-processing >>>
    # # Stop timers
    end_time = perf_counter()
    end_wall_time = time()

    # Update record
    record.update({
        "theta": theta,
        "intercept": intercept[0],
        "coefs": coefs[0][0],
        "accuracy": accuracy,
        "auroc": auroc,
        "aupr": aupr,
        "process_time": end_time - start_time,
        "wall_time": end_wall_time - start_wall_time
    })
    # <<< Post-processing <<<

    return record

def experiment(N, k1, k2, hyperparameters, experiment_setup):
    # >>> Book-keeping >>>
    # Initialize empty record
    empty_record = {
        "N": N,
        "k1": k1,
        "k2": k2,
        "theta": np.nan,
        "intercept": np.nan,
        "coefs": np.nan,
        "accuracy": np.nan,
        "auroc": np.nan,
        "aupr": np.nan,
        "process_time": np.nan,
        "wall_time": np.nan
    }

    # Solve for k_max from avg_k
    ## * See notes for derivation!
    k1_max = k1#1 / ((2/k1) - (1/3))
    k2_max = k2#1 / ((2/k2) - (1/3))

    # Build theta range
    thetas = params.build_theta_range(experiment_setup)

    # Format CL progress bar
    progbar_thetas = tqdm(thetas, desc="theta", position=3, leave=False, colour="red")

    # Sample specified duplex
    G_degs = [benchmarks.generate_power_law(Constants.GAMMA, Constants.KMIN, k1_max) for _ in range(N)]
    H_degs = [benchmarks.generate_power_law(Constants.GAMMA, Constants.KMIN, k2_max) for _ in range(N)]
    G_ = benchmarks.generate_configuration_model(G_degs)
    H_ = benchmarks.generate_configuration_model(H_degs)
    D = {1: G_, 2: H_}
    G, H = preprocessing.duplex_network(D, 1, 2)

    # Initialize records (theta sweep)
    records = []
    # <<< Book-keeping <<<

    # >>> Sweep theta >>>
    for theta in progbar_thetas:
        record = reconstruct(G, H, theta, hyperparameters, empty_record.copy())
        records.append(record)
    # <<< Sweep theta >>>

    return records

# ========== MAIN ==========
def main(N, k1s, k2s, hyperparameters, experiment_setup, output_filehandle=None):
    # >>> Book-keeping >>>
    records = []  # initialize records
    # <<< Book-keeping <<<

    # >>> Experiment >>>
    # Loop over _data_
    ## <K>_1
    progbar_k1s = tqdm(k1s, desc="<K>_1", position=0, colour="white")
    for k1 in progbar_k1s:
        ## Induced duplexes
        progbar_k2s = tqdm(k2s, desc="<K>_2", position=1, leave=False, colour="green")
        for k2 in progbar_k2s:
            ## Repeat for statistics
            progbar_repetitions = trange(experiment_setup["repeat"], desc="Repetitions", position=2, leave=False, colour="yellow")
            for _ in progbar_repetitions:
                records_ = experiment(N, k1, k2, hyperparameters, experiment_setup)
                records.extend(records_)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = postprocessing.df_from_records(records)
    # <<< Post-processing <<<

    # ! >>> DEBUG >>>
    print(tabulate(df[["k1", "k2", "theta", "accuracy", "auroc", "aupr"]], headers='keys', tablefmt='psql'))
    # ! <<< DEBUG <<<

    dataio.save_df(df, output_filehandle)

    return

if __name__ == "__main__":
    # >>> Experiment set-up >>>
    output_filehandle, TAG = \
        dataio.get_output_filehandle(
            PROJECT_ID="EMB_ex27-S",
            CURRENT_VERSION="v0.2",
            ROOT=ROOT
        )

    # Parameter ranges
    k1s = np.linspace(Constants.KRANGE_MIN, Constants.KRANGE_MAX, num=Constants.KRANGE_NUM)
    k2s = np.linspace(Constants.KRANGE_MIN, Constants.KRANGE_MAX, num=Constants.KRANGE_NUM)
    _, hyperparameters, experiment_setup = \
        params.set_parameters_N2V(
            # N2V
            workers=8,
            # LogReg
            fit_intercept=False, penalty="l2",
            # Other
            theta_min=0, theta_max=0.9, theta_num=10, repeat=20)
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")
    start_wall_time = time()
    start_time = perf_counter()

    main(Constants.N, k1s, k2s, hyperparameters, experiment_setup, output_filehandle)

    end_time = perf_counter()
    end_wall_time = time()
    print(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")
    print("\n", "="*60, "\n\n")
    # <<< Experiment <<<
