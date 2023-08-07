"""Functions to calculate degree feature in the reconstruction context.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Source code ---
import embmplxrec.utils

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    file_level=10,
    console_level=30)

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
        logger.warning(f"Node '{node}' not present in graph - substituting default value for degree ({default})")
        return default