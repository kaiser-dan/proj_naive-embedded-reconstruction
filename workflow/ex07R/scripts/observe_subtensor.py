# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
from os.path import abspath as ap
import pickle  # For serializing output
import random
import snakemake

# --- Network Science ---
import networkx as nx

# ============== FUNCTIONS ===========
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


def partial_information(G, frac):
    G1, G2 = G

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
        rem_G2.add_edge(e[0], e[1])with open(snakemake.input[0], "rb") as _fh:
        duplex = pickle.load(_fh)
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])

    return rem_G1, rem_G2, Etest


# ============== MAIN ===============
def main(multiplex, params):
    # Restrict multiplex to duplex
    duplex = duplex_network(multiplex, params["alpha"], params["beta"])

    # Observe subtensor
    remnant_G, remnant_H, test_set = \
        partial_information(
            duplex,
            float(params["pfi"])
        )

    # Save to disk
    data = {
        "system": params["system"],
        "alpha": params["alpha"],
        "beta": params["beta"],
        "pfi": params["pfi"],
        "true_duplex": duplex,
        "remnant_duplex": (remnant_G, remnant_H),
        "test_set": test_set,
        "repetition": params["rep"]
    }

    return data


if __name__ == "__main__":
    # Load system from disk
    multiplex = read_file(snakemake.input["duplex_edgelist"])

    # Book-keeping system name
    params = snakemake.params
    params.update({"system": snakemake.wildcards["system"]})

    # Run observation procedure on system
    observation = main(multiplex, snakemake.params)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(observation, _fh, pickle.HIGHEST_PROTOCOL)
