"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Scientific computations ---
import numpy as np

# --- Network science ---
from networkx import node_connected_component as component


# ============= FUNCTIONS =================
# --- Metrics ---
def euclidean_distance(x, y): return np.linalg.norm(x - y)
def cosine_similarity(x, y): return np.arccos(np.dot(x, y) / (np.norm(x) * np.norm(y)))
def poincare_disk_distance(x, y): raise NotImplementedError("Hyperbolic distance not yet implemented!")


# --- Drivers ---
def embedded_edge_distance(edge, X, Y, distance_=euclidean_distance):
    src, tgt = edge

    x = X[src] - X[tgt]
    y = Y[src] - Y[tgt]

    return distance_(x, y)


def component_penalized_embedded_edge_distance(edge, graph, X, Y, penalty=2**8, distance_=euclidean_distance):
    src, tgt = edge

    x = X[src] - X[tgt]
    y = Y[src] - Y[tgt]

    dist = distance_(x, y)

    if component(graph, src) != component(graph, tgt):
        dist += penalty

    return dist
