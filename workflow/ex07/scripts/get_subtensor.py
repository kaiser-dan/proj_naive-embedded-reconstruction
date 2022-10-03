# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
from os.path import abspath as ap
import pickle  # For serializing output
import random

# --- Network Science ---
import networkx as nx

# ============== FUNCTIONS ===========
def partial_information(G1, G2, frac):
    ##training/test sets
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

    ##remnants
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


def restrict_to_largest_component(remnant_G, remnant_H, test_set):
    remnant_G_adjusted = nx.Graph()
    remnant_H_adjusted = nx.Graph()
    remnant_G_adjusted.add_nodes_from(remnant_G.nodes())
    remnant_H_adjusted.add_nodes_from(remnant_H.nodes())

    maxcc_remnant_G = max(nx.connected_components(remnant_G), key=len)
    maxcc_remnant_H = max(nx.connected_components(remnant_H), key=len)

    edges_remnant_G_adjusted = set(remnant_G.subgraph(maxcc_remnant_G).edges())
    edges_remnant_H_adjusted = set(remnant_H.subgraph(maxcc_remnant_H).edges())
    remnant_G_adjusted.add_edges_from(edges_remnant_G_adjusted)
    remnant_H_adjusted.add_edges_from(edges_remnant_H_adjusted)

    test_set = {
        edge: gt_
        for edge, gt_ in test_set.items()
        if edge in edges_remnant_H_adjusted | edges_remnant_H_adjusted
    }

    return remnant_G_adjusted, remnant_H_adjusted, test_set


# ============== MAIN ===============
def main(duplex, parameters):
    # Observe duplex at each pfi
    remnant_G, remnant_H, test_set = partial_information(duplex[0], duplex[1], float(parameters["pfi"]))

    # Restrict to largest component, if specified
    if parameters["largest_component"]:
        remnant_G_adjusted, remnant_H_adjusted, test_set = restrict_to_largest_component(remnant_G, remnant_H, test_set)
    else:
        remnant_G_adjusted = remnant_G
        remnant_H_adjusted = remnant_H

    # Save to disk
    data = {
        "duplex": duplex,
        "remnant_duplex": (remnant_G_adjusted, remnant_H_adjusted),
        "test_set": test_set
    }
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(data, _fh, pickle.HIGHEST_PROTOCOL)

    return None


if __name__ == "__main__":
    # Load pickled duplex from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        duplex = pickle.load(_fh)

    # Run observation procedure
    main(duplex, snakemake.params)
