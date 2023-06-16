#!/usr/bin/env python
"""Script to apply a trained LogReg reconstruction model to a test set.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
import os
import pickle
from enum import Enum

# --- Scientific computing ---
import numpy as np

# --- Project source ---
# PATH adjustments
# ROOT = os.path.join(*["..", "..", ""])  # Relative to this file
ROOT = os.path.join(*["..", "..", "..", "..", ""])  # Relative to snakemake
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# Source code imports
## Classifiers
from src.classifiers import features, performance

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
def _verify_filepath(filepath: str, exists: bool):
    # File doesn't exist when it should
    if exists and not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    # File exists when it shouldn't
    if not exists and os.path.isfile(filepath):
        raise FileExistsError(filepath)

# --- Features ---
def calculate_degree_feature(cache: CachedEmbeddings):
    rem_G, rem_H = cache.remnants
    # ! >>> Broken >>>
    # ! Cached just the edges, not edge -> g.t. mapping
    # edges = set(cache.unobserved_edges.keys())
    # ! <<< HOT-FIX >>>
    edges = sorted(list(cache.unobserved_edges))
    # ! <<< BROKEN <<<


    # Calculate degrees
    src_rem_G, tgt_rem_G = features.get_degrees(rem_G.remnant, edges)
    src_rem_H, tgt_rem_H = features.get_degrees(rem_H.remnant, edges)

    numerators = src_rem_G * tgt_rem_G
    others = src_rem_H * tgt_rem_H

    feature = features.as_configuration(numerators, others, transform=lambda x: x)

    return feature

def calculate_distance_feature(cache: CachedEmbeddings):
    vec_G, vec_H = cache.embeddings
    # ! >>> Broken >>>
    # ! Cached just the edges, not edge -> g.t. mapping
    # edges = set(cache.unobserved_edges.keys())
    # ! <<< HOT-FIX >>>
    edges = sorted(list(cache.unobserved_edges))
    # ! <<< BROKEN <<<

    # Calculate degrees
    distances_G = features.get_distances(vec_G.vectors, edges)
    distances_H = features.get_distances(vec_H.vectors, edges)

    feature = features.as_configuration(distances_G, distances_H)

    return feature

def calculate_features(cache: CachedEmbeddings):
    # Ensure vectors are normalized
    for idx, vectors in enumerate(cache.embeddings):
        if not vectors.aligned or not vectors.scaled:
            components = cache.remnants[idx].get_components()
            vectors.normalize(components)

    features_degrees = calculate_degree_feature(cache)
    features_distances = calculate_distance_feature(cache)

    feature_matrix = features.format_feature_matrix(features_degrees, features_distances)

    return feature_matrix

# --- Model evaluation ---
def evaluate_model(model, test_data, test_labels):
    # Apply model to test data, forming reconstruction
    scores = model.get_scores(test_data)
    predictions = model.get_reconstruction(test_data)

    # Calculate performances
    accuracy = performance.performance(scores, predictions, test_labels, "accuracy")
    auroc = performance.performance(scores, predictions, test_labels, "AUC")
    pr = performance.performance(scores, predictions, test_labels, "PR")

    # Print to stdout (captured downstream)
    return f"{accuracy},{auroc},{pr}"


# ================= MAIN =======================
def main(
        cache_filepath: str,
        model_filepath: str,
        reconstruction_filepath: str,
        performance_filepath: str):
    # Verify CL arguments
    try:
        _verify_filepath(cache_filepath, True)
        _verify_filepath(model_filepath, True)
        _verify_filepath(reconstruction_filepath, False)
        _verify_filepath(performance_filepath, False)
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
        filehandle = open(cache_filepath, "rb")
        cache = pickle.load(filehandle)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)
    finally:
        filehandle.close()

    # Bring model into scope
    try:
        filehandle = open(model_filepath, "rb")
        model = pickle.load(filehandle)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)
    finally:
        filehandle.close()

    # Calculate features
    test_data = calculate_features(cache=cache)
    # ! >>> Broken >>>
    # ! Cached just the edges, not edge -> g.t. mapping
    # test_labels = features.get_labels(cache.unobserved_edges)
    # ! <<< HOT-FIX >>>
    test_labels = []
    for edge in sorted(list(cache.unobserved_edges)):
        if edge in cache.remnants[1].unknown_edges:
            test_labels.append(0)
        else:
            test_labels.append(1)
    test_labels = np.array(test_labels)
    # ! <<< BROKEN <<<

    # Evaluate model
    performance_output_string = evaluate_model(model, test_data, test_labels)

    # Save to disk
    np.save(reconstruction_filepath, model.get_reconstruction(test_data))
    with open(performance_filepath, "a") as _fh:
        _fh.write(performance_output_string)


if __name__ == "__main__":
    cache_filepath = sys.argv[1]  # CachedEmbedding filepath
    model_filepath = sys.argv[2]  # LogReg filepath
    reconstruction_filepath = sys.argv[3]  # Reconstruction filepath
    performance_filepath = sys.argv[4]  # Performance filepath

    main(cache_filepath, model_filepath, reconstruction_filepath, performance_filepath)