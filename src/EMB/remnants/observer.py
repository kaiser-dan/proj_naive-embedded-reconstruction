"""Interface to multiplex partial observation samplers.
"""
# ============= SET-UP =================
__all__ = [
    'cumulative_remnant_multiplexes',
    'build_remnant_multiplex',
    'random_observations_multiplex','random_observations']

# --- Imports ---
import random

import networkx as nx

from . import LOGGER

# --- Globals ---
AGGREGATE_LABEL = -1


# =================== FUNCTIONS ===================
# * Assumes disjoint layers!
def cumulative_remnant_multiplexes(multiplex, thetas):
    # Initialize remnant multiplexes return struct
    remnant_multiplexes = {theta: None for theta in thetas}

    # Ensure thetas are sorted
    thetas = sorted(thetas)

    # Initialize previous observations as empty
    previous_observations = {label: set() for label in multiplex.keys()}

    # Cumulative observe over given thetas
    for theta in thetas:
        # Get observed multiplex at given theta
        # Utilize previous observations for cumulative observations
        observed_multiplex = random_observations_multiplex(multiplex, theta, previous_observations)

        # Update previous observations to newly observed layers
        previous_observations = observed_multiplex

        # Build remnant multiplex from observations
        remnant_multiplex = build_remnant_multiplex(multiplex, observed_multiplex)

        remnant_multiplexes[theta] = remnant_multiplex

    return remnant_multiplexes

def build_remnant_multiplex(multiplex, observed_multiplex):
    # Initialize remnant multiplex return struct
    remnant_multiplex = dict()

    # Initialize aggregate (test edges) "layer"
    aggregate = nx.Graph()

    # Retrieve all edges in the multiplex
    all_edges = _get_all_edges(multiplex)

    # Build each remnant layer
    for label, layer in multiplex.items():
        # Initialize remnant layer as empty graph with known nodes
        remnant_layer = nx.Graph()
        remnant_layer.add_nodes_from(layer.nodes())
        aggregate.add_nodes_from(layer.nodes())

        # Initialize remnant layer edges as all multiplex edges
        remnant_layer.add_edges_from(all_edges)

        # Remove edges from layer observed in _other_ layers
        for observed_label, observed_layer in observed_multiplex.items():
            if observed_label != label:
                remnant_layer.remove_edges_from(observed_layer)

        remnant_multiplex[label] = remnant_layer


    # Add aggregate edges (test edges) with obstructed label
    aggregate.add_edges_from(all_edges)
    for observed_layer in observed_multiplex.values():
        aggregate.remove_edges_from(observed_layer)
    remnant_multiplex[AGGREGATE_LABEL] = aggregate

    return remnant_multiplex

# TODO: Generalize to allow non-disjoint layers
# * Currently calculates layers as [U_l M^l / (U_k!=l Theta^k)]
def random_observations_multiplex(multiplex, theta, previous_observations = None):
    if previous_observations is None:
        previous_observations = {label: set() for label in multiplex.keys()}

    if multiplex.keys() != previous_observations.keys():
        raise ValueError("Multiplex and previous observations must share labels (keys).")

    observed_multiplex = {
        label: random_observations(
            multiplex[label], theta, previous_observations[label])
        for label in multiplex.keys()
    }

    return observed_multiplex

def random_observations(graph, theta, previous_observations = set()):
    # Specify viable edges to observe (precluding prior observations)
    sample_space = _get_sample_space(graph, previous_observations)

    # Adjust theta to account for relative size of previous observations
    adj_theta = _adjust_theta(graph, theta, previous_observations)

    # Randomly observe a proportion of edges
    observed = random.sample(
        tuple(sample_space),  # sampling set is depreciated
        k=int(adj_theta*graph.number_of_edges()))

    # Adjoin to previous observations
    observed = set(observed) | previous_observations

    return observed


# --- Helpers ---
def _adjust_theta(graph, theta, previous_observations):
    adjusted_theta = theta - (len(previous_observations) / graph.number_of_edges())
    LOGGER.debug(f"Adjusted theta from {theta:.2f} to {adjusted_theta:.2f}")

    return adjusted_theta

def _get_sample_space(graph, previous_observations):
    return set(graph.edges()) - previous_observations

def _get_all_edges(multiplex):
    edges = set()
    for layer in multiplex.values():
        edges.update(set(layer.edges()))

    return edges
