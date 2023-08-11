"""Partial multiplex structural information sampling techniques.
"""
# ============= SET-UP =================

__all__ = ["partial_information", "random_observation"]

# --- Scientific computing ---
from numpy.random import default_rng as rng_

# --- Source code ---
from embmplxrec.remnants.remnant import RemnantNetwork, RemnantMultiplex

# --- Aliases ---
from embmplxrec._types import MaybePrevs


# ============= FUNCTIONS =================
# --- Drivers ---
def partial_information(
        graphs,
        proportion_observed,
        previous_observations: MaybePrevs = None):
    # >>> Book-keeping >>>
    # Unpack parameters
    if previous_observations is None:
        previous_observations = [None]*len(graphs)

    # Initialize return structs
    observed = dict()
    unobserved = dict()
    # <<< Book-keeping <<<

    # Get observed edges and infer unobserved edges
    for idx in range(len(graphs)):
        current_graph = graphs[idx]
        current_previous_observations = previous_observations[idx]
        observed[idx] = random_observation(current_graph, proportion_observed, current_previous_observations)
        unobserved[idx] = set(current_graph.edges()) - observed[idx]

    # Construct RemnantNetwork and RemnantMultiplex objects
    idxs = range(len(graphs))
    layers = [
        RemnantNetwork(graphs[idx], observed[idx], unobserved[idx], proportion_observed)
        for idx in idxs
    ]
    remnant_multiplex = RemnantMultiplex(layers, list(idxs))

    return remnant_multiplex

# --- Primary computations ---
def random_observation(graph, theta, cumulative_sample=None):
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

    # Cast as set of tuples
    observed_edges = {tuple(edge) for edge in observed_edges}
    return observed_edges

# --- Helpers ---
def _adjust_observation_for_cumulative(sample_space, theta, cumulative_sample):
    # Calculate edges still viable to be sampled
    adjusted_sample_space = tuple(set(sample_space) - set(cumulative_sample))

    # Calculate prior theta, infer proportion of samples for remaining observations
    theta_prior = len(cumulative_sample) / len(sample_space)
    adjusted_theta = theta - theta_prior

    return adjusted_sample_space, adjusted_theta