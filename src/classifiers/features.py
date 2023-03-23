"""Project source code for feature set creation and preprocessing.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Project source ---
sys.path.append("../")
from distance.distance import embedded_edge_distance, component_penalized_embedded_edge_distance
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

def get_configuration_distances_feature(distances_G, distances_H):
    # >>> Book-keeping >>>
    M = len(distances_G)  # get number of observations in dataset
    configuration_probabilities = []  # initialize feature set
    # <<< Book-keeping <<<

    # >>> Calculate configuration probabilities >>>
    for idx in range(M):
        probability = likelihood(distances_G[idx], distances_H[idx])
        probability = 2*probability - 1

        configuration_probabilities.append(probability)
    # <<< Calculate configuration probabilities <<<

    return configuration_probabilities
# < Distance <


# --- Formatters ---
