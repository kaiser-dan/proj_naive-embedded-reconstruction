"""Project source code for random observed sub-multiplex simulation.
"""
# ============= SET-UP =================
# --- Standard library ---
import collections import deque

# --- Scientific computing ---

# --- Network science ---
import networkx as nx

# --- Project code ---
from utils.remnants import _build_remnants


# ============= FUNCTIONS =================

def balanced_partial_information(G1, G2, frac, seednodes=[0], search="bfs"):
    """
    balanced_partial_information Observe a priori subduplex of `G1+G2` with a balanced snowball from the `seednodes` nodes.

    AGH

    Parameters
    ----------
    G1 : nx.Graph
        Layer one
    G2 : nx.Graph
        Layer two
    frac : float
        Proportion of edges to observe in each layer
    seednodes : list, optional
        Nodes from which to sample nearby edges, by default [0]
    search : str, optional
        Snowball strategy for sampling, by default "bfs"

    Raises
    ------
    NotImplementedError
        Only supports 'bfs' and 'dfs' values for `search` kwarg.

    Returns
    -------
    dict
        Remnant of layer one under partial observations
    dict
        Remnant of layer two under partial observations
    dict
        Edges not observed a priori and their true origins
    """
    if search == "bfs":
        tree = nx.bfs_tree
    elif search == "dfs":
        tree = nx.dfs_tree
    else:
        raise NotImplementedError("Must use 'bfs' or 'dfs' for search kwarg!")

    Etrain = {}
    Etest = {}

    trees_G1 = [deque(tree(G1, node)) for node in seednodes]
    trees_G2 = [deque(tree(G2, node)) for node in seednodes]

    seen_G1 = {edge: False for edge in G1.edges()}
    seen_G2 = {edge: False for edge in G2.edges()}

    num_edges_G1 = int(G1.number_of_edges()*frac)
    num_edges_G2 = int(G2.number_of_edges()*frac)

    seen_G1 = _snowball_sample_edges(trees_G1, seen_G1, num_edges_G1)
    seen_G2 = _snowball_sample_edges(trees_G2, seen_G2, num_edges_G2)

    for edge, seen in seen_G1.items():
        if seen:
            Etrain[edge] = 1
        else:
            Etest[edge] = 1

    for edge, seen in seen_G2.items():
        if seen:
            Etrain[edge] = 0
        else:
            Etest[edge] = 0

    # Remnants
    rem_G1, rem_G2, Etest = _build_remnants(G1, G2, Etrain, Etest)

    return rem_G1, rem_G2, Etest


def _snowball_sample_edges(trees, seen, iterations):
    count = 0
    while count < iterations:
        for tree in trees:
            candidate_edge_observation = tree.popleft()
            if not seen[candidate_edge_observation]:
                seen[candidate_edge_observation] = True
                count += 1

    return seen
