"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys

# --- Network science ---
from networkx import connected_components

# --- Source code ---
from distance.distance import _metrics

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon

# ============= FUNCTIONS =================
def embedded_edge_distance(
        edge, vectors,
        metric=_metrics.euclidean_distance):
    src, tgt = edge  # unpack edge
    distance = metric(vectors[src], vectors[tgt])
    distance += SYSTEM_PRECISION

    return distance

def component_penalized_embedded_edge_distance(
        edge, vectors, components,
        penalty=2**8,
        metric=_metrics.euclidean_distance):
    src, tgt = edge  # unpack edge

    distance = metric(vectors[src], vectors[tgt])
    distance += SYSTEM_PRECISION

    if components[src] != components[tgt]:
        distance += penalty

    return distance

# --- Helpers ---
def get_component_mapping(graph):
    mapping = {}  # node -> component
    components = connected_components(graph)  # [[nodes in component], ..., [nodes in component]]

    # Enumerate over components, associating included nodes to that component
    for component_id, component_nodes in enumerate(components):
        for node in component_nodes:
            mapping[node] = component_id

    return mapping