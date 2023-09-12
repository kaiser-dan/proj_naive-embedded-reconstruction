"""Vector pre-processing utility.
"""
# ============= SET-UP =================
__all__ = ['normalize_vectors']

# --- Imports ---
import numpy as np

from . import LOGGER


# =================== FUNCTIONS ===================
# --- Normalization ---
def normalize_vectors(vectors, components, node2id=dict()):
    vectors = align_vectors(vectors, components, node2id=node2id)
    assert len(vectors) > 0
    vectors = scale_vectors(vectors, components)  # already reindexed, do not pass node2id!
    assert len(vectors) > 0

    return vectors

def align_vectors(vectors, components, node2id=dict()):
    aligned_vectors = dict()

    for component in components:
        try:
            component_vectors = []
            for node in component:
                node_relabeled = node2id.get(node, node)
                vec = vectors[node_relabeled]
                component_vectors.append(vec)
        except (IndexError, KeyError) as err:
            LOGGER.error(f"{err}")
            LOGGER.debug(f"Component node: {node}")
            LOGGER.debug(f"node_relabeled: {node_relabeled}")
            # LOGGER.debug(f"vec: {vec}")
            raise err
        except Exception as err:
            LOGGER.critical(f"Previously unencountered error: {err}")
            raise err

        center = _get_center_of_mass(component_vectors)

        for node in component:
            aligned_vectors[node] = vectors[node2id.get(node, node)] - center

    return aligned_vectors

def scale_vectors(vectors, components, node2id=dict()):
    scaled_vectors = dict()

    for component in components:
        try:
            component_vectors = [vectors[node2id.get(node, node)] for node in component]
        except (IndexError, KeyError) as err:
            LOGGER.error(f"{err}")
            LOGGER.debug(f"Component nodes: {component}")
            LOGGER.debug(f"node2id: {node2id}")
            LOGGER.debug(f"vectors: {vectors}")
            raise err
        except Exception as err:
            LOGGER.critical(f"Previously unencountered error: {err}")
            raise err

        component_norm = _get_component_norm(component_vectors)

        # If component has 0 total norm, must be
        # a collection of zero vectors.
        # Cast norm as nonzero to avoid ZDE and let the
        # 0's travel through.
        if component_norm == 0:
            LOGGER.debug(f"Encountered zero total norm component (component = {component})")
            component_norm = 1

        for node in component:
            # ! DEBUG
            normed_vector = [coordinate / component_norm for coordinate in vectors[node2id.get(node, node)]]
            LOGGER.debug(f"node: {node} => normed_vector: {normed_vector}")

            scaled_vectors[node] = normed_vector
        LOGGER.debug(f"Current return 'scaled_vectors' = {scaled_vectors}")

    return scaled_vectors


# --- Helpers ---
def _get_center_of_mass(vectors):
    # Assumes vectors is array-like of array-likes (e.g., a matrix)
    return np.mean(vectors, axis=0)

def _get_component_norm(vectors):
    return np.sum([np.linalg.norm(vector) for vector in vectors])
