"""Experiment script to explore one feature logistic regression reconstructions from layer remnant embeddings

Broadly speaking, we have the following "workflow":

1. main() -> Sweep over _data_ (Systems \& induced duplexes)
2. experiment() -> For single data instance, sweep over _parameters_ (Theta, classifier parameters, etc.)
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys
from itertools import product

# --- Scientific computing ---
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- Network science ---

# --- Data handling ---
import pandas as pd

# --- Project source ---
# PATH adjustments
ROOT = "../../../../"
# ROOT = "./"
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
from tabulate import tabulate

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
        for penalty in [0.1]:
            record = experiment(
                system_layers, feature_set,
                theta, repetition,
                hyperparameters,
                penalty
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
        hyperparameters: dict[str, object],
        penalty: float=1e-12) -> dict[str, object]:
    # >>> Book-keeping >>>
    cache = observations.get_preprocessed_data(
        system_layers[0],
        (system_layers[1], system_layers[2]),
        theta, repetition,
        ROOT = ROOT + "data/input/preprocessed/synthetic"
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
        "log_penalty": np.log10(penalty)
    }
    # <<< Book-keeping <<<

    # >>> Calculations >>>
    # * Calculate features
    feature_distances_train = None
    feature_distances_test = None
    feature_degrees_train = None
    feature_degrees_test = None
    if "emb" in feature_set:
        # & Align centers
        cache.embeddings = cache.align_centers()

        # & Renormalize embeddings
        cache.embeddings = cache.renormalize()

        distances_G_train, distances_H_train = \
            features.get_distances(cache.embeddings, list(cache.observed_edges.keys()))
        distances_G_test, distances_H_test = \
            features.get_distances(cache.embeddings, list(cache.unobserved_edges.keys()))

        feature_distances_train = features.get_configuration_distances_feature(distances_G_train, distances_H_train, zde_penalty=penalty)
        feature_distances_test = features.get_configuration_distances_feature(distances_G_test, distances_H_test, zde_penalty=penalty)

    if "deg" in feature_set:
        src_degrees_train, tgt_degrees_train = \
            features.get_degrees(cache.remnants, list(cache.observed_edges.keys()))
        src_degrees_test, tgt_degrees_test = \
            features.get_degrees(cache.remnants, list(cache.unobserved_edges.keys()))

        feature_degrees_train = features.get_configuration_probabilities_feature(src_degrees_train, tgt_degrees_train)
        feature_degrees_test = features.get_configuration_probabilities_feature(src_degrees_test, tgt_degrees_test)

    # ^ Format feature matrix
    # ! >>> statsmodels package >>>
    # y, X, X_test = \
    #     features.format_feature_matrix_statsmodels(
    #         feature_set,
    #         len(cache.observed_edges), len(cache.unobserved_edges),
    #         cache.observed_edges, cache.unobserved_edges,
    #         feature_distances_train, feature_distances_test,
    #         feature_degrees_train, feature_degrees_test
    #     )
    #     _, labels_test = features.get_labels(
    #     cache.observed_edges, cache.unobserved_edges
    # )
    # # * Train classifier
    # try:
    #     model = sm.Logit(y, X)
    #     results = model.fit()
    # except ValueError:
    #     # ! Should be fixed, happens in extreme London cases (removed from caches)
    #     sys.stderr.write("ValueError")
    #     with open("debug-value.log", "a") as fh:
    #         print(record, file=fh)
    #     return record
    # except np.linalg.LinAlgError:
    #     # ! Convergence issues, may be amenable to hyperparameter-based fix?
    #     sys.stderr.write("np.linalg.LinAlgError")
    #     with open("debug-linalg.log", "a") as fh:
    #         print(record, file=fh)
    #     return record
    # except PerfectSeparationError:
    #     # ! One independent variable is perfect classifier (a good problem to have)
    #     sys.stderr.write("Perfect Separability!")
    #     with open("debug-PSE.log", "a") as fh:
    #         print(record, file=fh)
    #     return record

    # intercept = list(results.params)[0]
    # coefficients = list(results.params)[1:]
    # scores = results.predict(X_test)
    # classes = [1 if score >= 0.5 else 0 for score in scores]
    # pr_curve = precision_recall_curve(labels_test, scores)

    # # * Reconstruct
    # accuracy = accuracy_score(labels_test, classes)
    # auroc = roc_auc_score(labels_test, scores)
    # aupr = auc(pr_curve[1], pr_curve[0])
    # ! <<< statsmodels package <<<
    # ! >>> scikit-learn package >>>
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
    model = logreg.train_fit_logreg(feature_matrix_train, labels_train, hyperparameters["classifier"])
    intercept, coefficients = logreg.get_model_fit(model)
    intercept = intercept[0]
    coefficients = coefficients[0]
    try:
        accuracy = logreg.get_model_accuracy(model, feature_matrix_test, labels_test)
        auroc = logreg.get_model_auroc(model, feature_matrix_test, labels_test)
        aupr = logreg.get_model_aupr(model, feature_matrix_test, labels_test)
    except ValueError:
        return record
    # ! <<< scikit-learn package <<<
    # <<< Calculations <<<

    # >>> Post-processing >>>
    # Update record
    coef_emb = coefficients[0] if "emb" in feature_set else None
    if "deg" in feature_set and "emb" not in feature_set:
        coef_deg = coefficients[0]
    elif "deg" in feature_set and "emb" in feature_set:
        coef_deg = coefficients[1]
    else:
        coef_deg = None
    record.update({
        "intercept": intercept,
        "coefficients": coefficients,
        "coef_emb": coef_emb,
        "coef_deg": coef_deg,
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
            PROJECT_ID="EMB_ex32",
            CURRENT_VERSION="v2.0",
            ROOT=ROOT
        )

    # Parameter grid
    system_layer_sets = {
        # & Synthetic systems
        (f"LFR_mu-{mu}_prob-{prob}", 1, 2)
        for mu in [0.1, 0.2, 0.3, 0.4, 0.5]
        for prob in [0.0, 0.25, 0.5, 0.75, 1.0]
    }
    feature_sets = (
        # & Single features
        # {"imb"},
        # {"emb"},
        # {"deg"},
        # & Feature pairs
        {"imb", "emb"},
        {"imb", "deg"},
        # {"emb", "deg"},
        # & All features
        {"imb", "emb", "deg"},
    )
    _, hyperparameters, experiment_setup = \
        params.set_parameters_N2V(
            fit_intercept=True,  solver="newton-cholesky", penalty="l2", # logreg
            theta_min=0.05, theta_max=0.95, theta_num=37, repeat=1  # other
        )
    # <<< Experiment set-up <<<

    # >>> Experiment >>>
    sys.stderr.write("\n"+"="*30+TAG+"="*30+"\n\n")  # print stdout preface
    start_wall_time = time()  # start timers
    start_time = perf_counter()

    df = main(
        system_layer_sets, feature_sets,
        hyperparameters,
        experiment_setup, output_filehandle
    )  # run simulations

    end_time = perf_counter()  # lap timers
    end_wall_time = time()
    sys.stderr.write(f"Total process time: {(end_time - start_time):.4f} \t Total wall time: {(end_wall_time - start_wall_time):.4f}")  # print stdout postface
    sys.stderr.write("\n"+"="*60+"\n\n")
    # <<< Experiment <<<
