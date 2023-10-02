"""Feature calculations for multiplex reconstruction features.
"""
# ============= SET-UP =================
__all__ = ["get_distances_feature", "get_degrees_feature", "get_edge_to_layer"]

# --- Imports ---
import numpy as np
from networkx.exception import NetworkXError

from EMB import netsci

from . import LOGGER

# --- Globals ---
EPSILON = 1e-32  # desired maximum precision


# =================== FUNCTIONS ===================
def get_distances_feature(vectorsets, edgeset, training=False):
    distances = []
    for vectors in vectorsets:
        dists_ = [inverse_vector_distance(vectors, edge, training) for edge in edgeset]
        distances.append(dists_)

    return _get_normalized_feature(distances)


def get_degrees_feature(graphs, edgeset, training=True):
    degree_products = []
    for graph in graphs:
        degprods_ = [degree_product(graph, edge, training) for edge in edgeset]
        degree_products.append(degprods_)

    return _get_normalized_feature(degree_products)


def inverse_vector_distance(vectors, edge, training=True):
    try:
        v_i = vectors[edge[0]]
        v_j = vectors[edge[1]]
    # Handle non-existent vector errors that arise from disconnected observations in training set
    # If error occurs as expected, can be proxied with theory d(i,j) = inf if no path i->j
    except (KeyError, IndexError) as err:
        LOGGER.info(f"Vector access error: {err}; checking error catch protocol...")
        LOGGER.debug(f"Edge = {edge}")
        LOGGER.debug(f"Vectors keys = {vectors.keys()}")
        if training:
            LOGGER.info(
                "Function call flagged as training set, continuing with d = 1/epsilon"
            )
            return 1 / EPSILON
        else:
            LOGGER.error("Function call flagged as testing set, rethrowing error")
            raise err
    # Handle other errors
    except Exception as err:
        LOGGER.critical(f"Previously unencountered error: {err}")
        raise err

    # Cast as numpy arrays to facilitate L2-norm
    if not isinstance(v_i, np.ndarray):
        v_i = np.array(v_i)
    if not isinstance(v_j, np.ndarray):
        v_j = np.array(v_j)

    # Compute L2-norm
    d = np.linalg.norm(v_i - v_j)

    # If vectors within specific distance, force distance to be system min
    # This avoids downstream ZeroDivisionErrors and OverflowErrors
    if d <= EPSILON:
        d = EPSILON

    return 1 / d


# ? Do we need exception catchers?
def degree_product(graph, edge, training=True):
    try:
        k_i = graph.degree(edge[0])
        k_j = graph.degree(edge[1])
    # Handle non-existent vector errors that arise from disconnected observations in training set
    # If error occurs as expected, can be proxied with theory k_i = 0 if i not in G
    except NetworkXError as err:
        LOGGER.info(f"Degree access error: {err}; checking error catch protocol...")
        LOGGER.debug(f"Edge = {edge}")
        LOGGER.debug(f"src in G? {edge[0] in graph}")
        LOGGER.debug(f"tgt in G? {edge[1] in graph}")
        if training:
            LOGGER.info(
                "Function call flagged as training set, proceeding with null degree"
            )
            return 0
        else:
            LOGGER.error("Function call flagged as testing set, rethrowing error")
            raise err
    # Handle other errors
    except Exception as err:
        LOGGER.critical(f"Previously unencountered error: {err}")
        raise err

    return k_i * k_j


def get_edge_to_layer(edges, graphs):
    edge_to_layer = dict()
    for edge in edges:
        layer = netsci.find_edge(edge, *graphs)
        if len(layer) > 1:
            LOGGER.info(
                "Edge found in more than one layer; taking origin as lexigraphic minimum of layer ids"
            )
        layer = layer[0]
        edge_to_layer[edge] = layer

    return edge_to_layer


# --- Helpers ---
def _get_normalized_feature(feature_components, scaling=lambda x: 2 * x - 1):
    normalized_feature = []
    for idx in range(len(feature_components[0])):
        numerator = feature_components[0][idx]
        denominator = 0
        for feature_component in feature_components:
            denominator += feature_component[idx]

        val = numerator / denominator
        val = scaling(val)

        normalized_feature.append(val)

    return np.array(normalized_feature)
