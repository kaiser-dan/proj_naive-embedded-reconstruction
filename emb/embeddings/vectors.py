"""Vector pre-processing utility.
"""
# ============= SET-UP =================
__all__ = ["normalize_vectors"]

# --- Imports ---
import numpy as np

from loguru import logger as LOGGER


# =================== FUNCTIONS ===================
# --- Normalization ---
def normalize_vectors(vectors, components, node2id=dict()):
    vectors = align_vectors(vectors, components, node2id=node2id)
    assert len(vectors) > 0

    vectors = scale_vectors(
        vectors, components
    )  # already reindexed, do not pass node2id!
    assert len(vectors) > 0

    return vectors


@LOGGER.catch
def align_vectors(vectors, components, node2id=dict()):
    aligned_vectors = dict()

    for component in components:
        component_vectors = [vectors[node2id.get(node, node)] for node in component]
        center = _get_center_of_mass(component_vectors)
        for node in component:
            aligned_vectors[node] = vectors[node2id.get(node, node)] - center

    return aligned_vectors


@LOGGER.catch
def scale_vectors(vectors, components, node2id=dict()):
    scaled_vectors = dict()

    for component in components:
        component_vectors = [vectors[node2id.get(node, node)] for node in component]
        component_norm = _get_component_norm(component_vectors)
        LOGGER.debug(f"Component: {len(component)} nodes, {component_norm} total norm")

        # If component has 0 total norm, must be a collection of zero vectors.
        # Cast norm as nonzero to avoid ZDE
        # and let the 0 _vector components_ propogate.
        if component_norm == 0:
            LOGGER.info(
                f"Encountered zero total norm component (component = {component})"
            )
            component_norm = 1

        for node in component:
            scaled_vectors[node] = (1 / component_norm) * np.array(
                vectors[node2id.get(node, node)]
            )

    return scaled_vectors


# --- Helpers ---
def _get_center_of_mass(vectors):
    # Assumes vectors is array-like of array-likes (e.g., a matrix)
    return np.mean(vectors, axis=0)


def _get_component_norm(vectors):
    return np.sum([np.linalg.norm(vector) for vector in vectors])
