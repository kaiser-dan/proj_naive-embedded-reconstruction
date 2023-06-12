"""Project source code for feature set creation and preprocessing.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys
import os

# --- Scientific computing ---
import numpy as np

# --- Project source ---
SRC = os.path.join(*["..", ""])
sys.path.append(SRC)
from src.distance.distance import embedded_edge_distance
from src.distance.score import scale_probability


# ========== FUNCTIONS ==========
# --- Helpers ---
def get_labels(edges):
    return list(edges.values())

# TODO: Add floating-point comparison safety for small floats
def safe_inverse(x, tolerance=sys.float_info.epsilon):
    x += tolerance
    if np.isclose(x, 0):
        raise ZeroDivisionError("Input with system precision is still too small!")
    else:
        return 1 / x

# --- Feature calculations ---
def get_degrees(graph, edges):
    src_degrees = []
    tgt_degrees = []

    for src, tgt in edges:
        src_degrees.append(graph.degree(src))
        tgt_degrees.append(graph.degree(tgt))

    return np.array(src_degrees), np.array(tgt_degrees)

def get_distances(vectors, edges):
    distances = [
        embedded_edge_distance(edge, vectors)
        for edge in edges
    ]

    return np.array(distances)

# --- Formatters ---
def as_configuration(
        *data,
        transform: function = safe_inverse,
        scale: function = scale_probability):
    # Gather passed in lists of features
    data = list(*data)

    # Apply transformation
    for idx in range(len(data)):
        data[idx] = list(map(transform, data[idx]))

    # Cast as configuration model
    configurations = []
    numerators = data[0]
    for idx in range(len(numerators)):
        configuration = numerators[idx] / sum([data_[idx] for data_ in data])
        configuration = scale(configuration)
        configurations.append(configuration)

    return configurations

def format_feature_matrix(*feature_vectors):
    # Get number of features and observations
    num_cols = len(*feature_vectors)
    num_rows = len(feature_vectors[0])
    dims = (num_rows, num_cols)

    # Instantiate empty feature matrix
    feature_matrix = np.empty(dims)

    # Add each feature to feature matrix, one column at a time
    for idx, feature in enumerate(feature_vectors):
        feature_matrix[:, idx] = feature

    return feature_matrix
