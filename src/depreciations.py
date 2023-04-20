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


def embedded_edge_distance_ratio(
    edge, vectors_numerator, vectors_denominator,
        distance_=euclidean_distance):
    # Calculate edge distance in each remnant
    dist_numerator = embedded_edge_distance(edge, vectors_numerator, distance_)
    dist_denominator = embedded_edge_distance(edge, vectors_denominator, distance_)

    # Calculate distance ratio
    try:
        ratio = dist_numerator / dist_denominator
    except ZeroDivisionError:
        ratio = np.finfo(np.float64).max
    finally:
        return ratio


def component_penalized_embedded_edge_distance_ratio(
    edge,
    graph_numerator, graph_denominator,
    vectors_numerator, vectors_denominator,
        penalty=2**8, distance_=euclidean_distance):
    # >>> Book-keeping >>>
    src, tgt = edge  # identify nodes incident to edge
    # <<< Book-keeping <<<

    # >>> Score (feature) calculation >>>
    # Calculate edge distance in each remnant
    dist_numerator = component_penalized_embedded_edge_distance(edge, graph_numerator, vectors_numerator, penalty, distance_)
    dist_denominator = component_penalized_embedded_edge_distance(edge, graph_denominator, vectors_denominator, penalty, distance_)

    # Account for isolated component bias
    component_numerator_src = component(graph_numerator, src)
    component_numerator_tgt = component(graph_numerator, tgt)
    component_denominator_src = component(graph_denominator, src)
    component_denominator_tgt = component(graph_denominator, tgt)
    if (component_numerator_src == component_numerator_tgt) and (len(component_numerator_src) <= 15):
        dist_numerator += penalty
    if (component_denominator_src == component_denominator_tgt) and (len(component_denominator_src) <= 15):
        dist_denominator += penalty

    # Calculate distance ratio
    try:
        ratio = dist_numerator / dist_denominator
    except ZeroDivisionError:
        ratio = np.finfo(np.float64).max
    # <<< Score (feature) calculation <<<
    finally:
        return ratio
