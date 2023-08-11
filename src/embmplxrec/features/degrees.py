"""Functions to calculate degree feature in the reconstruction context.
"""
# ============= SET-UP =================

__all__ = ["get_degrees"]

# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon
from . import LOGGER

# ============= FUNCTIONS =================
# --- Drivers ---
def get_degrees(graph, edges):
    src_degrees = [
        safe_degree(graph, edge[0])
        for edge in edges
    ]
    tgt_degrees = [
        safe_degree(graph, edge[1])
        for edge in edges
    ]

    return np.array(src_degrees) * np.array(tgt_degrees)


# --- Helpers ---
def safe_degree(graph, node, default=0):
    if node in graph:
        return graph.degree(node)
    else:
        LOGGER.warning(f"Node '{node}' not present in graph - substituting default value for degree ({default})")
        return default