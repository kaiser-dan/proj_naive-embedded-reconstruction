"""Depreciated project source code.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Project source ---
sys.path.append("./")
from distance.distance import embedded_edge_distance, component_penalized_embedded_edge_distance


# ========== FUNCTIONS ==========
def get_distances(vectors, edges):
    # >>> Book-keeping >>>
    G, H = vectors  # alias input layer embedded node vectors
    # <<< Book-keeping <<<

    # >>> Distance calculations >>>
    G_distances = [embedded_edge_distance(edge, G) for edge in edges]
    H_distances = [embedded_edge_distance(edge, H) for edge in edges]
    # <<< Distance calculations

    return G_distances, H_distances

def get_biased_distances(vectors, edges, components):
    # >>> Book-keeping >>>
    G, H = vectors  # alias input layer embedded node vectors
    G_components, H_components = components  # alias component mapping of remnants
    # <<< Book-keeping <<<

    # >>> Distance calculations >>>
    G_distances = [component_penalized_embedded_edge_distance(edge, G, G_components) for edge in edges]
    H_distances = [component_penalized_embedded_edge_distance(edge, H, H_components) for edge in edges]
    # <<< Distance calculations

    return G_distances, H_distances

def prepare_feature_matrix_confdeg_dist(configuration_degrees, distances):
    # >>> Book-keeping >>>
    N = len(configuration_degrees)
    NUM_FEATURES = 3

    G_distances, H_distances = distances

    feature_matrix = np.empty((N, NUM_FEATURES))
    # <<< Book-keeping <<<

    # >>> Format feature matrix >>>
    feature_matrix[:, 0] = configuration_degrees
    feature_matrix[:, 1] = G_distances
    feature_matrix[:, 2] = H_distances
    # <<< Format feature matrix <<<

    return feature_matrix
