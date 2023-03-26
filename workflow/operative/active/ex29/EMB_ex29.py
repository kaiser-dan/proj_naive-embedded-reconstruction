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
import numpy as np
from sklearn.metrics import roc_auc_score

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
from data import preprocessing, postprocessing
from data import observations
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
def main(
        system_layer_sets: set[tuple[str, int, int]],
        feature_sets: set[set[str]],
        parameters: dict[str, object], hyperparameters: dict[str, object],
        experiment_setup: dict[str, object],
        output_filehandle: str = None) -> pd.DataFrame:
    # >>> Book-keeping >>>
    # Prepare recordbook
    records = []
    record_ = {
        "system": None,
        "l1": None,
        "l2": None,
        "features": None,
        "theta": None,
        "intercept": None,
        "coefficients": None,
        "accuracy": None,
        "auroc": None,
        "aupr": None,
        "process_time": None,
        "wall_time": None
    }
    # <<< Book-keeping <<<

    # >>> Experiment >>>
    # Process parameter grid
    thetas = params.build_theta_range(experiment_setup)
    repetitions = range(1, experiment_setup["repeat"]+1)
    parameter_grid = product(system_layer_sets, feature_sets, thetas, repetitions)

    # Run experiment
    for parameter_grid_vertex in tqdm(parameter_grid, desc="Experiment", leave=False):
        system_layers, feature_set, theta, repetition = parameter_grid_vertex
        record = record_.copy()
        record.update(experiment(
            system_layers, feature_set,
            theta, repetition,
            parameters, hyperparameters
        ))
        records.extend(record)
    # <<< Experiment <<<

    # >>> Post-processing >>>
    df = postprocessing.df_from_records(records)
    # <<< Post-processing <<<

    dataio.save_df(df, output_filehandle)

    return df


def experiment(
        system_layers: tuple[str, int, int],
        feature_set: set[str],
        theta: float,
        repetition: int,
        parameters: dict[str, object],
        hyperparameters: dict[str, object]) -> dict[str, object]:
    # >>> Book-keeping >>>
    preprocessed_data = observations.get_preprocessed_data(
        system_layers[0],
        (system_layers[1], system_layers[2]),
        theta, repetition
    )
    # <<< Book-keeping <<<

    # >>> Calculations >>>

    # <<< Calculations <<<


    pass

# ========== MAIN ==========
if __name__ == "__main__":
    # >>> Experiment set-up >>>
    # Metadata
    output_filehandle, TAG = \
        dataio.get_output_filehandle(
            PROJECT_ID="EMB_ex29",
            CURRENT_VERSION="v2.0",
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

    df = main(
        system_layer_sets, feature_sets,
        parameters, hyperparameters,
        experiment_setup, output_filehandle
    )  # run simulations

    end_time = perf_counter()  # lap timers
    end_wall_time = time()
    print(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")  # print stdout postface
    print("\n", "="*60, "\n\n")
    # <<< Experiment <<<
