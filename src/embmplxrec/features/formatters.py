"""Formatting tools for pre-processing features in multiplex reconstruction setting.
"""
# ========== SET-UP ==========
# --- Network science ---
from networkx.exception import NetworkXError

# --- Scientific computing ---
import numpy as np

# --- Project source ---
import embmplxrec.features.distances as dists
import embmplxrec.features._functions as fcns

import embmplxrec.utils

# --- Miscellaneous ---
logger = embmplxrec.utils.logger.get_module_logger(
    name=__name__,
    file_level=10,
    console_level=30)


# ========== FUNCTIONS ==========
# --- Feature calculations ---
def get_degrees(graph, edges):
    src_degrees = []
    tgt_degrees = []

    for src, tgt in edges:
        try:
            src_degrees.append(graph.degree(src))
        except NetworkXError as err:
            logger.warning(f"Encountered NetworkXError ('{err}') - forcing degree 0")
            src_degrees.append(0)

        try:
            tgt_degrees.append(graph.degree(tgt))
        except NetworkXError as err:
            logger.warning(f"Encountered NetworkXError ('{err}') - forcing degree 0")
            tgt_degrees.append(0)

    return np.array(src_degrees), np.array(tgt_degrees)

def get_distances(vectors, edges):
    distances = [
        dists.embedded_edge_distance(edge, vectors)
        for edge in edges
    ]

    return np.array(distances)

# --- Formatters ---
def as_configuration(
        target_feature,
        *other_features,
        transform = fcns.inverse,
        scale = fcns.scale_probability):
    # Verify feature sets are valid and have same length
    N = len(target_feature)
    if any([len(feature) != N for feature in other_features]):
        raise ValueError("Some feature(s) do not share the same number of values as the given target feature!")

    # Apply transformation to each feature value
    target_feature = list(map(transform, target_feature))
    for idx, feature in enumerate(other_features):
        other_features[idx] = list(map(transform, feature))

    # Cast as configuration model
    ## Numerators as target (transformed) feature
    target_feature = np.array(target_feature)

    ## Denominators as sum of all (transformed) features
    denominators = np.sum([np.array(other_features)], axis=0)

    ## Apply division element-wise and scale
    configurations = list(map(scale, target_feature / denominators))

    return configurations

def format_feature_matrix(*feature_vectors):
    # Get number of features and observations
    num_cols = len(feature_vectors)
    num_rows = len(feature_vectors[0])
    dims = (num_rows, num_cols)

    # Instantiate empty feature matrix
    feature_matrix = np.empty(dims)

    # Add each feature to feature matrix, one column at a time
    for idx, feature in enumerate(feature_vectors):
        feature_matrix[:, idx] = feature

    return feature_matrix
