"""[Docstring here].
"""
# =========== SETUP ==========
# --- Imports ---
import os
import shelve

import numpy as np
import networkx as nx

import EMB

# --- Aliases ---
# Functions
jn = os.path.join

# Pathing
ROOT = jn("..", "..", "")
FIGURES = jn(ROOT, "results", "figures", "")
DATA = jn(ROOT, "data", "")
TMP = jn(ROOT, ".tmp", "")

# --- Globals ---
Ns = np.arange(1000, 5100, 100, dtype=int)

# =========== FUNCTIONS ==========
def get_graphs():
    fp = jn(TMP, "graphs.shelf")
    if not os.path.isfile(fp):
        with shelve.open(fp) as shelf:
            for N in Ns:
                G = nx.barabasi_albert_graph(N, 3)
                shelf[N] = G

    with shelve.open(fp) as shelf:
        graphs = shelf.values()

    return graphs

def embed_graph(graph, embedding):
    match embedding:
        case "N2V":
            vectors = EMB.embeddings.embed_N2V(graph, dimensions=128)
        case "LE":
            vectors = EMB.embeddings.embed_LE(graph, k=128)
        case "Isomap":
            vectors = EMB.embeddings.embed_Isomap(graph, dimensions=128)
        case "HOPE":
            vectors = EMB.embeddings.embed_HOPE(graph, dimensions=128)
        case _:
            raise NotImplementedError

    fp = jn(TMP, f"vectors_{embedding}.shelf")
    with open(fp) as shelf:
        shelf[graph.number_of_nodes()] = vectors



vectors_n2v = dict()  # N -> vectors
vectors_le = dict()  # N -> vectors
vectors_isomap = dict()  # N -> vectors
vectors_hope = dict()  # N -> vectors
edge_sets = dict()  # N -> edges
for N in Ns:
    print(f"Embedding N={N}...")

    G = nx.erdos_renyi_graph(N, 2/N)
    edge_sets[N] = G.edges()

    vectors_n2v[N] = EMB.embeddings.embed_N2V(G, dimensions=128)
    vectors_le[N] = EMB.embeddings.embed_LE(G, k=128)
    vectors_isomap[N] = EMB.embeddings.embed_Isomap(G, dimensions=128)
    vectors_hope[N] = EMB.embeddings.embed_HOPE(G, dimensions=128)
