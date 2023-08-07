"""Functions to calculate distances between embedded node vectors in the reconstruction context.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Source code ---
from embmplxrec.features import _functions
import embmplxrec.utils

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    file_level=10,
    console_level=30)

logger.warning("[DEPRECIATION] Epislon addition to vector distance will be depreciated in an upcoming version!")

# ============= FUNCTIONS =================
# --- Drivers ---
def get_distances(vectors, edges):
    distances = [
        embedded_edge_distance(edge, vectors)
        for edge in edges
    ]

    return np.array(distances)

# --- Computations ---
def embedded_edge_distance(
        edge, vectors,
        metric=_functions.euclidean_distance):
    src, tgt = edge  # unpack edge

    try:
        distance = metric(vectors[src], vectors[tgt])
    except KeyError as err:  # * unknown cause of string keys may occur
        logger.warning(f"Found KeyError on '{err}' - checking types...")
        # Attempt type fix
        if isinstance(src, str) or isinstance(tgt, str):
            logger.warning("Found string type for node index! Converting to integer and retrying")
            src, tgt = int(src), int(tgt)
            distance = embedded_edge_distance((src,tgt), vectors, metric)  # recurse to recheck errors
        else:
            logger.error(f"Types are valid; rethrowing error")
            raise err

    logger.debug("Adding system precision to avoid ZeroDivisionErrors")

    distance += SYSTEM_PRECISION

    return distance
