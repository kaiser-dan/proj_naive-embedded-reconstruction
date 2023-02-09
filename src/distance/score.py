"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Scientific computing ---
import numpy as np

# --- Network science ---
from networkx import node_connected_component as component

# --- Project source code ---
from distance.distance import *

# ============= FUNCTIONS =================
# --- Likelihood models ---
# Basic convex models
def inverse_(x): return 1/x
def negexp_(x): return np.exp(-x)


# Sigmoid models
def logistic_(x): return 1 / (1 + negexp_(x))
def tanh_(x): return np.tanh(x)
def arctan_(x): return np.arctan(x)


# --- Drivers ---
# Probabilities
def likelihood(
    target_distance, *other_distances,
        likelihood_model = inverse_):
    # Calculate likelihood for specified layer (given distance)
    likelihood_of_target = likelihood_model(target_distance)

    # Normalize likelihood by likelihood over all layers
    likelihoods_of_others = [
        likelihood_model(dist) for dist in other_distances
    ]
    normalization_term = likelihood_of_target + sum(likelihoods_of_others)

    return likelihood_of_target / normalization_term

# Features
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
    elif (component_denominator_src == component_denominator_tgt) and (len(component_denominator_src) <= 15):
        dist_denominator += penalty

    # Calculate distance ratio
    try:
        ratio = dist_numerator / dist_denominator
    except ZeroDivisionError:
        ratio = np.finfo(np.float64).max
    # <<< Score (feature) calculation <<<
    finally:
        return ratio