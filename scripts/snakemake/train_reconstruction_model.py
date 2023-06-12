#!/usr/bin/env python
"""Script to train a logistic regression-based duplex reconstruction model.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import pickle
from enum import Enum

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
## Classifiers
from src.classifiers.logreg import LogReg
from src.classifiers import features

## CachedEmbedding
from src.data.caches import CachedEmbeddings

# --- Globals ---
## Exit status
class Status(Enum):
    OK = 0
    FILE = 1
    OTHER = 2


# ================= FUNCTIONS =======================
# --- File I/O ---
def _verify_filepath(filepath: str, output: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    if os.path.isfile(output):
        raise FileExistsError(output)

# --- Features ---
def calculate_degree_feature(cache: CachedEmbeddings):
    rem_G, rem_H = cache.remnants
    edges = set(cache.observed_edges.keys())

    # Calculate degrees
    src_rem_G, tgt_rem_G = features.get_degrees(rem_G, edges)
    src_rem_H, tgt_rem_H = features.get_degrees(rem_H, edges)

    numerators = src_rem_G * tgt_rem_G
    others = src_rem_H * tgt_rem_H

    feature = features.as_configuration(numerators, others, transform=lambda x: x)

    return feature

def calculate_distance_feature(cache: CachedEmbeddings):
    vec_G, vec_H = cache.embeddings
    edges = set(cache.observed_edges.keys())

    # Calculate degrees
    distances_G = features.get_distances(vec_G, edges)
    distances_H = features.get_distances(vec_H, edges)

    feature = features.as_configuration(distances_G, distances_H)

    return feature

def calculate_features(cache: CachedEmbeddings):
    # Ensure vectors are normalized
    for vectors in cache.embeddings:
        if not vectors.aligned or not vectors.scaled:
            vectors.normalize()

    features_degrees = calculate_degree_feature(cache)
    features_distances = calculate_distance_feature(cache)

    feature_matrix = features.format_feature_matrix(features_degrees, features_distances)

    return feature_matrix


# ================= MAIN =======================
def main(filepath: str, output: str):
    # Verify CL arguments
    try:
        _verify_filepath(filepath, str)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        quit(Status.FILE.value)
    # ? Should FEE be a warning instead?
    except FileExistsError as err:
        print(err, file=sys.stderr)
        quit(Status.FILE.value)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)

    # Bring CachedEmbedding into scope
    try:
        filehandle = open(filepath, "rb")
        cache = pickle.load(filehandle)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)
    finally:
        filehandle.close()

    # Calculate features
    training_data = calculate_features(cache=cache)
    training_labels = features.get_labels(cache.observed_edges)

    # Build LogReg class instance
    model_type = "LogReg"
    features = ("intercept", "degrees", "distances")

    # TODO: Handle non-default parameters
    model = LogReg(
        model_type=model_type,
        features=features,
        experiment_params=dict(),
        training_data=training_data,
        training_labels=training_labels,
        logreg_parameters=dict(penalty='none')
    )

    # Save to disk
    model.save(output)


if __name__ == "__main__":
    filepath = sys.argv[1]
    output = sys.argv[2]

    main(filepath, output)