"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Network science ---
from networkx import connected_components

# --- Source code ---
from distance import _metrics

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon

# ============= FUNCTIONS =================
def embedded_edge_distance(
        edge, vectors,
        metric=_metrics.euclidean_distance):
    src, tgt = edge  # unpack edge

    try:
        distance = metric(vectors[src], vectors[tgt])
    except KeyError as err:  # * unknown cause of string keys may occur
        # Attempt type fix
        if isinstance(src, str) or isinstance(tgt, str): 
            src, tgt = int(src), int(tgt)
            distance = metric(vectors[src], vectors[tgt])
        # Consider as vacuous distance
        else:
            distance = 999

    
    distance += SYSTEM_PRECISION

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
