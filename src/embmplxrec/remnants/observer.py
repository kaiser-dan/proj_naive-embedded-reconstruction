"""Partial multiplex structural information sampling techniques.
"""
# ============= SET-UP =================

__all__ = ["partial_information", "random_observation"]

# --- Network science ---
import networkx as nx

# --- Scientific computing ---
from numpy.random import default_rng as rng_

# --- Source code ---
from embmplxrec.remnants.remnant import RemnantNetwork, RemnantMultiplex

from embmplxrec._types import MaybePrevs, Graphs, AbstractEdgesList, AbstractEdgesetLabeled

# --- Globals ---
from . import LOGGER
LOGGER.warning("Cumulative observations currently not implemented! Any cumulative training sets are ignored.")


# ============= FUNCTIONS =================
# --- Drivers ---
def get_observed(true_layers: Graphs, theta: float):
    observed = []
    for id, layer in enumerate(true_layers):
        observations = random_observation(layer, theta, id)
        observed.append(observations)

    return observed

def get_unobserved(aggregate_edges: AbstractEdgesetLabeled, observed: AbstractEdgesList):
    # Find edges which are in the observed set of _any_ layer
    all_observations = set()
    for observations in observed:
        all_observations.update(observations.keys())

    # Calculate unobserved edges as edges not observed to
    # originate from any layer
    unobserved = {
        edge: layer
        for edge, layer in aggregate_edges.items()
        if edge not in all_observations
    }

    return unobserved

def partial_information(
        graphs: Graphs,
        proportion_observed: float,
        previous_observations: MaybePrevs = None):
    # --- Book-keeping ---
    # Unpack parameters
    if previous_observations is not None:
        raise NotImplementedError("previous_observations is not yet implemented!")

    N = graphs[0].number_of_nodes()

    # --- Observations ---
    # Get all edges, from every layer
    all_edges = aggregate_layers(graphs)  # dict: edge -> true layer

    # Get observations in each layer
    observed = get_observed(true_layers=graphs, theta=proportion_observed)
    unobserved = get_unobserved(aggregate_edges=all_edges, observed=observed)

    # --- Form remnants ---
    remnant_layers = []
    for observations in observed:
        # Initialize empty graph
        remnant_layer = nx.Graph()
        remnant_layer.add_nodes_from(range(N))

        # Add edges observed edges in this layer (training edges)
        remnant_layer.add_edges_from(observations)

        # Add edges unknown to belong to this layer (testing edges)
        remnant_layer.add_edges_from(unobserved)

        # Construct RemnantNetwork object
        remnant_layer = RemnantNetwork(
            graph=remnant_layer,
            observed=observations,
            unobserved=unobserved,
            theta=proportion_observed
        )
        remnant_layers.append(remnant_layer)

    # Construct RemnantMultiplex object
    remnant_multiplex = RemnantMultiplex(
        layers=remnant_layers,
        labels=range(len(remnant_layers)),
        metadata={
            "theta": proportion_observed,
            "cumulative": False
        }
    )

    return remnant_multiplex

# --- Primary computations ---
def random_observation(graph, theta, id, cumulative_sample=None) -> dict[tuple[int, int], int]:
    # Initialize numpy random Generator
    rng = rng_()

    # Declare sample space for possible observations
    sample_space = tuple(graph.edges())

    # If sampling cumulatively, make sure not to repeat samples
    if cumulative_sample is not None:
        sample_space, theta = _adjust_observation_for_cumulative(sample_space, theta, cumulative_sample)

    # Efficiently draw sample of edge observations
    ## Utilizes sum_n(Bernoulli(p)) ~ Binomial(n, p)
    num_edges_observed = int(rng.binomial(len(sample_space), theta))
    observed_edges = rng.choice(
        sample_space,
        size=num_edges_observed,
        replace=False)

    return {
        tuple(edge): id
        for edge in observed_edges
    }

def aggregate_layers(graphs: list) -> dict[tuple[int, int], int]:
    all_edges = dict()  # dict: edge -> originating layer id
    for idx, graph in enumerate(graphs):
        all_edges.update({
            edge: idx for
            edge in set(graph.edges())
        })

    return all_edges

# --- Helpers ---
def _adjust_observation_for_cumulative(sample_space, theta, cumulative_sample):
    # Calculate edges still viable to be sampled
    adjusted_sample_space = tuple(set(sample_space) - set(cumulative_sample))

    # Calculate prior theta, infer proportion of samples for remaining observations
    theta_prior = len(cumulative_sample) / len(sample_space)
    adjusted_theta = theta - theta_prior

    return adjusted_sample_space, adjusted_theta