# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output
import random

# --- Network Science ---
import networkx as nx

# ============== FUNCTIONS ===========
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

    ##observed
    obs_G1 = nx.Graph()
    obs_G2 = nx.Graph()
    for n in G1:
        obs_G1.add_node(n)
        obs_G2.add_node(n)
    for n in G2:
        obs_G1.add_node(n)
        obs_G2.add_node(n)


    # Add aggregate edges we don't know about
    for e in Etest:
        rem_G1.add_edge(e[0], e[1])
        rem_G2.add_edge(e[0], e[1])

    # Add aggregate edges we do know about _to observed_!
    for e, layer in Etrain.items():
        if layer == 1:
            obs_G1.add_edge(e[0], e[1])
        else:
            obs_G2.add_edge(e[0], e[1])

    # Add to remnant alpha the things known to be in alpha
    # So remnant alpha is unknown + known(alpha)
    for e in Etrain:
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])

    return rem_G1, rem_G2, Etest, obs_G1, obs_G2


# ============== MAIN ===============
def main(multiplex, params):
    # Restrict multiplex to duplex
    duplex = duplex_network(multiplex, 0, 1)

    # Observe subtensor
    remnant_G, remnant_H, test_set, observed_G, observed_H = \
        partial_information(
            duplex,
            float(params["pfi"])
        )

    # Save to disk
    data = {
        "true_duplex": duplex,
        "remnant_duplex": (remnant_G, remnant_H),
        "observed_duplex": (observed_G, observed_H),
        "test_set": test_set,
        "pfi": params["pfi"],
        "repetition": params["rep"]
    }

    # print(f"=============\n {params['pfi']}\n ==================")

    return data


if __name__ == "__main__":
    # Load duplex from disk
    with open(snakemake.input["multiplex"], "rb") as _fh:
        multiplex = pickle.load(_fh)


    # Run observation procedure on system
    params = dict(snakemake.params)
    observation = main(multiplex, params)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(observation, _fh, pickle.HIGHEST_PROTOCOL)
