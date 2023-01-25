# ============= SET-UP =================
# --- Standard library ---
import random  # for random shufflings
from collections import deque

# --- Network science ---
import networkx as nx  # General network tools


# =================== FUNCTIONS ===================
# --- File processing ---
def read_file(filename):
    G = {}

    with open(filename) as file:
        for line in file:
            data = line.strip().split()
            l = int(data[0])
            n = int(data[1])
            m = int(data[2])
            if l not in G:
                G[l] = nx.Graph()
            G[l].add_edge(n, m)

    return G


# --- Experimental setup ---
def duplex_network(G, l1, l2):
    G1 = G[l1].copy()
    G2 = G[l2].copy()

    # Delete common edges
    list_of_common_edges = []

    for e in G[l1].edges():
        if G[l2].has_edge(e[0], e[1]):
            list_of_common_edges.append([e[0], e[1]])

    for e in list_of_common_edges:
        G1.remove_edge(e[0], e[1])
        G2.remove_edge(e[0], e[1])

    # Delete nodes with zero degree
    list_of_nodes = []
    for n in G1.nodes():
        if G1.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.remove_node(n)

    list_of_nodes = []
    for n in G2.nodes():
        if G2.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G2.remove_node(n)

    ##create union of nodes
    list_of_nodes = []
    for n in G1.nodes():
        list_of_nodes.append(n)
    for n in G2.nodes():
        list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.add_node(n)
        G2.add_node(n)

    return G1, G2


def partial_information(G1, G2, frac):
    # Training/test sets
    Etest = {}
    Etrain = {}

    for e in G1.edges():
        if random.random() < frac:
            Etrain[e] = 1
        else:
            Etest[e] = 1

    for e in G2.edges():
        if random.random() < frac:
            Etrain[e] = 0
        else:
            Etest[e] = 0

    # Remnants
    rem_G1, rem_G2, Etest = _build_remnants(G1, G2, Etrain, Etest)

    return rem_G1, rem_G2, Etest, Etrain


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

# ----- Helpers -----
def _build_remnants(G1, G2, Etrain, Etest):
    # Remnants
    rem_G1 = nx.Graph()
    rem_G2 = nx.Graph()
    for n in G1:
        rem_G1.add_node(n)
        rem_G2.add_node(n)
    for n in G2:
        rem_G1.add_node(n)
        rem_G2.add_node(n)

    # Add aggregate edges we don't know about
    for e in Etest:
        rem_G1.add_edge(e[0], e[1])
        rem_G2.add_edge(e[0], e[1])

    # Add to remnant alpha the things known to be in alpha
    # So remnant alpha is unknown + known(alpha)
    for e in Etrain:
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])

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
