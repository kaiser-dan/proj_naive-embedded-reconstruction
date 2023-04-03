"""Project source code for feature set creation and preprocessing.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Project source ---
sys.path.append("../")
from distance.distance import embedded_edge_distance
from distance.score import likelihood


# ========== FUNCTIONS ==========
# --- Calculations ---
# > Degrees >
def get_degrees(graphs, edges):
    # >>> Book-keeping >>>
    G, H = graphs  # alias input layer graphs
    # <<< Book-keeping <<<

    # >>> Degree calculations >>>
    src_degrees = [
        [G.degree(edge[0]) for edge in edges],
        [H.degree(edge[0]) for edge in edges]
    ]
    tgt_degrees = [
        [G.degree(edge[1]) for edge in edges],
        [H.degree(edge[1]) for edge in edges]
    ]
    # <<< Degree calculations

    return src_degrees, tgt_degrees

def get_configuration_probabilities_feature(src_degrees, tgt_degrees):
    # >>> Book-keeping >>>
    M = len(src_degrees[0])  # get number of observations in dataset
    configuration_probabilities = []  # initialize feature set
    # <<< Book-keeping <<<

    # >>> Calculate configuration probabilities >>>
    for idx in range(M):
        # numerator = k_i^G * k_j^G
        numerator = src_degrees[0][idx] * tgt_degrees[0][idx]

        # denominator = (k_i^G * k_j^G) + (k_i^H * k_j^H)
        denominator = numerator + (src_degrees[1][idx] * tgt_degrees[1][idx])

        probability = numerator / denominator
        probability = 2*probability - 1

        configuration_probabilities.append(probability)
    # <<< Calculate configuration probabilities <<<

    return configuration_probabilities
# < Degrees <

# > Distance >
def get_distances(vectors, edges):
    # >>> Book-keeping >>>
    G, H = vectors  # alias input layer graphs
    # <<< Book-keeping <<<

    # >>> Distance calculations >>>
    distances_G = [embedded_edge_distance(edge, G) for edge in edges]
    distances_H = [embedded_edge_distance(edge, H) for edge in edges]
    # <<< Distance calculations

    return distances_G, distances_H

def get_configuration_distances_feature(distances_G, distances_H, tilde=False, zde_penalty = 1e-9):
    # >>> Book-keeping >>>
    M = len(distances_G)  # get number of observations in dataset
    configuration_probabilities = []  # initialize feature set
    # <<< Book-keeping <<<

    # >>> Calculate configuration probabilities >>>
    for idx in range(M):
        # Base feature form
        if not tilde:
            probability = likelihood(distances_G[idx], distances_H[idx])
        else:
            probability = distances_G[idx] / (distances_H[idx] + zde_penalty)

        # Feature transformation
        if not tilde:
            probability = 2*probability - 1

        configuration_probabilities.append(probability)
    # <<< Calculate configuration probabilities <<<

    return configuration_probabilities
# < Distance <


# --- Formatters ---
def format_feature_matrix(
        feature_set, M_train, M_test,
        feature_distances_train=None,
        feature_distances_test=None,
        feature_degrees_train=None,
        feature_degrees_test=None,
):
    if feature_set == {"imb"}:
        feature_matrix_train = np.array([0]*M_train).reshape(-1,1)
        feature_matrix_test = np.array([0]*M_test).reshape(-1,1)
    elif feature_set == {"emb"}:
        feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
    elif feature_set == {"deg"}:
        feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif feature_set == {"imb", "emb"}:
        feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
    elif feature_set == {"imb", "deg"}:
        feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif feature_set == {"emb", "deg"} or feature_set == {"emb", "deg", "imb"}:
        feature_matrix_train = np.empty((M_train, 2))
        feature_matrix_train[:, 0] = feature_distances_train
        feature_matrix_train[:, 1] = feature_degrees_train

        feature_matrix_test = np.empty((M_test, 2))
        feature_matrix_test[:, 0] = feature_distances_test
        feature_matrix_test[:, 1] = feature_degrees_test

    return feature_matrix_train, feature_matrix_test


# --- Helpers ---
def get_labels(trainset, testset):
    labels_train = list(trainset.values())
    labels_test = list(testset.values())
    return labels_train, labels_test