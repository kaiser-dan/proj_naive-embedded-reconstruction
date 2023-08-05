"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import os

# --- Scientific computing ---
import numpy as np

# --- Network science ---
from networkx import connected_components

# --- Source code ---
from distance import _metrics

sys.path.append(os.path.join("..", "utils", ""))
import utils

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon
logger = utils.get_module_logger(name=__name__, file_level=0, console_level=30)

# ============= FUNCTIONS =================
def embedded_edge_distance(
        edge, vectors,
        metric=_metrics.euclidean_distance):
    src, tgt = edge  # unpack edge

    try:
        distance = metric(vectors[src], vectors[tgt])
    except KeyError as err:  # * unknown cause of string keys may occur
        logger.warning(f"Found KeyError on {err} - checking types...")
        # Attempt type fix
        if isinstance(src, str) or isinstance(tgt, str):
            logger.warning("Found string type for node index! Converting to integer and retrying")
            src, tgt = int(src), int(tgt)
            distance = embedded_edge_distance((src,tgt), vectors, metric)
        # ? Consider as vacuous distance
        else:
            logger.critical(f"Types are as expected; rethrowing err {err}")
            # ? distance = 999

    logger.debug("Adding system precision to avoid ZeroDivisionErrors")
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
