"""Formatting tools for pre-processing features in multiplex reconstruction setting.
"""
# ========== SET-UP ==========

__all__ = ["as_configuration", "format_feature_matrix"]

# --- Scientific computing ---
import numpy as np

# --- Project source ---
import embmplxrec.features._functions as fcns

# --- Miscellaneous ---
from . import LOGGER


# ========== FUNCTIONS ==========
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
    other_features = list(other_features)
    for idx, feature in enumerate(other_features):
        other_features[idx] = list(map(transform, feature))

    # Cast as configuration model
    ## Numerators as target (transformed) feature
    target_feature = np.array(target_feature)

    ## Denominators as sum of all (transformed) features
    denominators = np.resize(
        np.sum([np.array(other_features)], axis=0),
        (len(target_feature),)
    )
    denominators += target_feature

    ## Apply division element-wise
    configurations = target_feature / denominators

    # Scale
    configurations = [
        scale(p)
        for p in configurations
    ]

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
