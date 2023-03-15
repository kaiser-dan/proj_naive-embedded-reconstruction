"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Scientific computations ---
import numpy as np

# --- Network science ---
from networkx import connected_components


# ============= FUNCTIONS =================
# --- Metrics ---
def euclidean_distance(x, y): return np.linalg.norm(x - y)
def cosine_similarity(x, y): return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
def poincare_disk_distance(x, y): raise NotImplementedError("Hyperbolic distance not yet implemented!")


# --- Drivers ---
def embedded_edge_distance(edge, vectors, distance_=euclidean_distance):
    src, tgt = edge

    return distance_(vectors[src], vectors[tgt]) + 1e-16

def component_penalized_embedded_edge_distance(edge, vectors, components, penalty=2**8, distance_=euclidean_distance):
    src, tgt = edge

    try:
        dist = distance_(vectors[src], vectors[tgt]) + 1e-16
    except ValueError:  # Dimension mismatch when per-component embedding applied
        dist = 1e-16

    if components[src] != components[tgt]:
        dist += penalty

    return dist



# --- Helpers ---
def get_component_mapping(graph):
    mapping = {}
    components = connected_components(graph)

    for component_id, component_nodes in enumerate(components):
        for node in component_nodes:
            mapping[node] = component_id

    return mapping