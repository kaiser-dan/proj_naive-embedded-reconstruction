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
from itertools import product

# --- Scientific computing ---

# --- Network science ---

# --- Data handling ---
import pandas as pd

# --- Project source ---
# PATH adjustments
ROOT = "../../../../"
sys.path.append(f"{ROOT}/src/")

# Primary modules
## Data
from data import dataio
from data import postprocessing
from data import observations

## Classifiers
from classifiers import features  # feature set helpers
from classifiers import logreg  # logistic regression

## Utilities
from utils import parameters as params  # helpers for experiment parameters

# --- Miscellaneous ---
from time import perf_counter, time
from tqdm.auto import tqdm  # progress bars

import warnings
warnings.filterwarnings("ignore")  # remove sklearn depreciation warnings

# ========== FUNCTIONS ==========
def main(
        system_layer_sets: set[tuple[str, int, int]],
        feature_sets: set[set[str]],
        hyperparameters: dict[str, object],
        experiment_setup: dict[str, object],
        output_filehandle: str = None) -> pd.DataFrame:
    # >>> Book-keeping >>>
    # Prepare recordbook
    records = []
    # <<< Book-keeping <<<

    # >>> Experiment >>>
    # Process parameter grid
    thetas = params.build_theta_range(experiment_setup)
    repetitions = range(1, experiment_setup["repeat"]+1)
    parameter_grid = list(product(system_layer_sets, feature_sets, thetas, repetitions))

    # Run experiment
    for parameter_grid_vertex in tqdm(parameter_grid, desc="Experiment"):
        system_layers, feature_set, theta, repetition = parameter_grid_vertex
        record = experiment(
            system_layers, feature_set,
            theta, repetition,
            hyperparameters
        )
        records.append(record)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = postprocessing.df_from_records(records)
    dataio.save_df(df, output_filehandle)
    # <<< Post-processing <<<


    return df


def experiment(
        system_layers: tuple[str, int, int],
        feature_set: set[str],
        theta: float,
        repetition: int,
        hyperparameters: dict[str, object]) -> dict[str, object]:
    # >>> Book-keeping >>>
    cache = observations.get_preprocessed_data(
        system_layers[0],
        (system_layers[1], system_layers[2]),
        theta, repetition,
        ROOT = ROOT + "data/input/preprocessed"
    )

    record = {
        "system": cache.system,
        "l1": cache.layers[0],
        "l2": cache.layers[1],
        "features": feature_set,
        "theta": cache.theta,
        "intercept": None,
        "coefficients": None,
        "accuracy": None,
        "auroc": None,
        "aupr": None,
    }

    if "imb" in feature_set:
        hyperparameters["classifier"]["fit_intercept"] = True
    # <<< Book-keeping <<<

    # >>> Calculations >>>
    # * Calculate features
    feature_distances_train = None
    feature_distances_test = None
    feature_degrees_train = None
    feature_degrees_test = None
    if "emb" in feature_set:
        # & Renormalize embeddings
        cache.renormalize()

        distances_G_train, distances_H_train = \
            features.get_distances(cache.embeddings, list(cache.observed_edges.keys()))
        distances_G_test, distances_H_test = \
            features.get_distances(cache.embeddings, list(cache.unobserved_edges.keys()))

        feature_distances_train = features.get_configuration_distances_feature(distances_G_train, distances_H_train)
        feature_distances_test = features.get_configuration_distances_feature(distances_G_test, distances_H_test)

    if "deg" in feature_set:
        src_degrees_train, tgt_degrees_train = \
            features.get_degrees(cache.remnants, list(cache.observed_edges.keys()))
        src_degrees_test, tgt_degrees_test = \
            features.get_degrees(cache.remnants, list(cache.unobserved_edges.keys()))

        feature_degrees_train = features.get_configuration_probabilities_feature(src_degrees_train, tgt_degrees_train)
        feature_degrees_test = features.get_configuration_probabilities_feature(src_degrees_test, tgt_degrees_test)

    # ^ Format feature matrix
    feature_matrix_train, feature_matrix_test = \
        features.format_feature_matrix(
            feature_set,
            len(cache.observed_edges), len(cache.unobserved_edges),
            feature_distances_train, feature_distances_test,
            feature_degrees_train, feature_degrees_test
        )
    labels_train, labels_test = features.get_labels(
        cache.observed_edges, cache.unobserved_edges
    )

    # * Train classifier
    try:
        model = logreg.train_fit_logreg(feature_matrix_train, labels_train, hyperparameters["classifier"])
    except ValueError:  # when only one class is available, happens for some london cases
        return record

    intercept, coefficients = logreg.get_model_fit(model)
    try:
        assert intercept[0] != 0
    except AssertionError as msg:
        print(msg)

    # * Reconstruct
    try:
        accuracy = logreg.get_model_accuracy(model, feature_matrix_test, labels_test)
        auroc = logreg.get_model_auroc(model, feature_matrix_test, labels_test)
        aupr = logreg.get_model_aupr(model, feature_matrix_test, labels_test)
    except ValueError:  # only one class available, fricken London crap
        return record
    # <<< Calculations <<<

    # >>> Post-processing >>>
    # Update record
    record.update({
        "intercept": intercept[0],
        "coefficients": coefficients[0],
        "accuracy": accuracy,
        "auroc": auroc,
        "aupr": aupr,
    })
    # <<< Post-processing <<<

    return record

# ========== MAIN ==========
if __name__ == "__main__":
    # >>> Experiment set-up >>>
    # Metadata
    output_filehandle, TAG = \
        dataio.get_output_filehandle(
            PROJECT_ID="EMB_ex29",
            CURRENT_VERSION="v2.0.3",
            ROOT=ROOT
        )

    # Parameter grid
    system_layer_sets = {
        # Large systems
        ("arxiv", 2, 6),
        ("drosophila", 1, 2),
        # Small systems
        ("celegans", 1, 2),
        ("london", 1, 2),
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
    _, hyperparameters, experiment_setup = \
        params.set_parameters_N2V(
            fit_intercept=False,  # logreg
            theta_min=0.05, theta_max=0.95, theta_num=11, repeat=10  # other
        )
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    print("\n", "="*30, TAG, "="*30, "\n\n")  # print stdout preface
    start_wall_time = time()  # start timers
    start_time = perf_counter()

    df = main(
        system_layer_sets, feature_sets,
        hyperparameters,
        experiment_setup, output_filehandle
    )  # run simulations

    end_time = perf_counter()  # lap timers
    end_wall_time = time()
    print(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")  # print stdout postface
    print("\n", "="*60, "\n\n")
    # <<< Experiment <<<
