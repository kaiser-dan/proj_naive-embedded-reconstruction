# Input: CachedEmbedding filepath, LogReg filepath
# Output: Reconstruction filepath
#!/usr/bin/env python
"""Script to apply a trained LogReg reconstruction model to a test set.
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
def _verify_filepath(filepath: str, output: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    if os.path.isfile(output):
        raise FileExistsError(output)

# --- Features ---
def calculate_degree_feature(cache: CachedEmbeddings):
    rem_G, rem_H = cache.remnants
    edges = set(cache.unobserved_edges.keys())

    # Calculate degrees
    src_rem_G, tgt_rem_G = features.get_degrees(rem_G, edges)
    src_rem_H, tgt_rem_H = features.get_degrees(rem_H, edges)

    numerators = src_rem_G * tgt_rem_G
    others = src_rem_H * tgt_rem_H

    feature = features.as_configuration(numerators, others, transform=lambda x: x)

    return feature

def calculate_distance_feature(cache: CachedEmbeddings):
    vec_G, vec_H = cache.embeddings
    edges = set(cache.unobserved_edges.keys())

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
def main(cache_filepath: str, model_filepath: str, output: str):
    # Verify CL arguments
    try:
        _verify_filepath(cache_filepath, output)
        _verify_filepath(model_filepath, output)
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
        cache = pickle.load(cache_filepath)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)
    finally:
        filehandle.close()

    # Bring model into scope
    try:
        filehandle = open(model_filepath, "rb")
        model = pickle.load(model_filepath)
    except Exception as err:
        print(err, file=sys.stderr)
        quit(Status.OTHER.value)
    finally:
        filehandle.close()

    # Calculate features
    test_data = calculate_features(cache=cache)
    test_labels = features.get_labels(cache.unobserved_edges)

    # Apply model to test data, forming reconstruction
    scores = model.get_scores(test_data)
    predictions = model.get_reconstruction(test_data)

    accuracy = performance.performance(scores, predictions, test_labels, "accuracy")
    auroc = performance.performance(scores, predictions, test_labels, "AUC")
    pr = performance.performance(scores, predictions, test_labels, "PR")

    # Print to stdout (captured downstream)
    print(f"{accuracy}, {auroc}, {pr}")


if __name__ == "__main__":
    filepath = sys.argv[1]
    output = sys.argv[2]

    main(filepath, output)