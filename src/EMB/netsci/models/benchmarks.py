"""Common multiplex benchmark generation.
"""
# ============= SET-UP =================
__all__ = ['generate_network_LFR', 'generate_duplex_LFR']

# --- Imports ---
import os
import subprocess
import random

import numpy as np
import networkx as nx


# =================== FUNCTIONS ===================
# --- Single-layer ---
def generate_network_LFR(
        num_nodes: int,
        degree_heterogeneity: float,
        community_heterogeneity: float,
        community_mixing: float,
        degree_average: float,
        degree_max: float,
        id_adjustment: int = 1,
        ROOT = os.path.join("..", "")):

    # Generate LFR from binary
    subprocess.call(
        " ".join([
            f"{os.getcwd()}/{ROOT}/bin/LFR/benchmark.sh",  # symlink to binary
            "-N", str(num_nodes),
            "-t1", str(degree_heterogeneity),
            "-t2", str(community_heterogeneity),
            "-mu", str(community_mixing),
            "-k", str(degree_average),
            "-maxk", str(degree_max)
        ]),
        shell=True,
        stdout=open(os.devnull, 'w'),
        stderr=open(os.devnull, 'w')
    )

    # Format resultant network as networkx Graph
    edges_raw = np.loadtxt("network.dat")
    edges = []
    for idx in range(len(edges_raw)):
        src, tgt = edges_raw[idx]
        src = int(src) - id_adjustment
        tgt = int(tgt) - id_adjustment
        edges.append((src, tgt))
    graph = nx.Graph(edges)

    # Format resultant node partition as [node -> community] mapping
    comms_raw = np.loadtxt("community.dat")
    comms = dict()
    for idx in range(len(comms_raw)):
        node, comm = comms_raw[idx]
        node = int(node) - id_adjustment
        comm = int(comm)
        comms[node] = comm

    return graph, comms


# --- Multiplex ---
def generate_multiplex_configuration():
    raise NotImplementedError("Not yet ported from previous code base!")

def generate_multiplex_BA_overlapping():
    raise NotImplementedError("Not yet ported from previous code base!")

def generate_duplex_LFR(
        num_nodes: int,
        degree_heterogeneity: float,
        community_heterogeneity: float,
        community_mixing: float,
        degree_average: float,
        degree_max: float,
        probability_relabel: float = 1.0,
        id_adjustment: int = 1,
        ROOT = os.path.join("..", "")):
    # Generate LFR layer from which to build duplex
    graph, comms = generate_network_LFR(
        num_nodes, degree_heterogeneity, community_heterogeneity,
        community_mixing, degree_average, degree_max, id_adjustment, ROOT)

    # Create [community -> nodes] mapping
    # TODO: Reimplement with default_dict
    communities = dict()
    for node, community in comms.items():
        if community not in communities:
            communities[community] = [node]
        else:
            communities[community].append(node)

    # Alias [node -> community] mapping of original LFR
    sigma_1 = comms

    # Shuffle community labels
    node_to_new_comm = dict()
    for community in communities.values():
        tmp = community.copy()
        random.shuffle(tmp)
        for node_idx in range(len(community)):
            old_comm_label = community[node_idx]
            new_comm_label = tmp[node_idx]
            node_to_new_comm[old_comm_label] = new_comm_label

    # Apply community label shuffling to [node -> community] mapping
    tmp_sigma_2 = dict()
    for node in sigma_1.keys():
        new_comm_label = node_to_new_comm[node]
        tmp_sigma_2[new_comm_label] = sigma_1[node]

    # Initialize duplex of LFR layers
    D = dict()
    D[1] = graph.copy()
    D[2] = nx.Graph()

    # Ensure same node set in both layers
    D[2].add_nodes_from(D[1].nodes())

    # Add edges to second layer _with label shuffling_
    for edge in D[1].edges():
        src, tgt = edge
        src = node_to_new_comm[src]
        tgt = node_to_new_comm[tgt]
        D[2].add_edge(src, tgt)

    # TODO: Finish refactor
    # Break community correlation between layers
    nodes = list(D[2].nodes())
    new_labels = dict()
    H = D[2].copy()
    for node in D[2]:
        new_labels[node] = node
    for node in new_labels:
        if random.random() < probability_relabel:
            new_node = random.choice(nodes)
            tmp = new_labels[node]
            new_labels[node] = new_labels[new_node]
            new_labels[new_node] = tmp

    D[2] = nx.Graph()
    for n in H:
        m = new_labels[n]
        D[2].add_node(m)
    for e in H.edges():
        n = new_labels[e[0]]
        m = new_labels[e[1]]
        D[2].add_edge(n, m)

    sigma_2 = dict()
    for n in tmp_sigma_2:
        m = new_labels[n]
        sigma_2[m] = tmp_sigma_2[n]

    return D, sigma_1, sigma_2, community_mixing
