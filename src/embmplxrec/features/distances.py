"""Functions to calculate distances between embedded node vectors in the reconstruction context.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys

# --- Source code ---
from embmplxrec.features import _functions
import embmplxrec.utils

# --- Globals ---
SYSTEM_PRECISION = sys.float_info.epsilon
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    file_level=10,
    console_level=30)

# ============= FUNCTIONS =================
def embedded_edge_distance(
        edge, vectors,
        metric=_functions.euclidean_distance):
    src, tgt = edge  # unpack edge

    try:
        distance = metric(vectors[src], vectors[tgt])
    except KeyError as err:  # * unknown cause of string keys may occur
        logger.warning(f"Found KeyError on {err} - checking types...")
        # Attempt type fix
        if isinstance(src, str) or isinstance(tgt, str):
            logger.warning("Found string type for node index! Converting to integer and retrying")
            src, tgt = int(src), int(tgt)
            distance = embedded_edge_distance((src,tgt), vectors, metric)  # recurse to recheck errors
        # ? Consider as vacuous distance
        else:
            logger.critical(f"Types are as expected; rethrowing err {err}")
            # ? distance = 999

    logger.debug("Adding system precision to avoid ZeroDivisionErrors")
    distance += SYSTEM_PRECISION
    logger.warning("Epislon addition to vector distance will be depreciated in an upcoming version!")

    return distance
