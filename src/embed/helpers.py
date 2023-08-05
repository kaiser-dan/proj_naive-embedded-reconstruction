"""Helpers for dealing with graph embeddings and their outputs.
"""
# ============= SET-UP =================
# --- Scientific computing ---
import numpy as np

# --- Network science ---
import networkx as nx

# ============= FUNCTIONS =================
# --- Dealing with node labels ---
def reindex_nodes(graph):
    """Reindex graph nodes to contiguous range [0, N-1].

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    dict
        node label -> new node label
    """
    reindexed_nodes = {
        index: new_index
        for new_index, index in enumerate(sorted(graph.nodes()))
    }
    return reindexed_nodes


def get_contiguous_vectors(model):
    """Get vectors from graph embbedding accounting for potential non-contiguous node ids.

    Parameters
    ----------
    model : word2vec model

    Returns
    -------
    dict
        node label -> embedded vector
    """
    # Retrieve 'raw' vectors
    vectors = model.vectors

    # Retrieve word2vec internal hash of node ids to vector indices
    node_labels = model.index_to_key

    # Map node ids into corresponding vector
    # This accounts for graphs with non-consecutive node ids
    embedding = {
        int(node_label): vectors[node_index]
        for node_index, node_label in enumerate(node_labels)
    }

    return embedding

# --- Converting data structs ---
def dict_to_matrix(D):
    # * Assumes contiguous keys from 0
    num_rows = len(D)
    num_cols = len(D[0])

    M = np.empty((num_rows, num_cols))

    for row_idx, row in D.items():
        M[row_idx] = row

    return M

def matrix_to_dict(M):
    D = {
        row_idx: row
        for row_idx, row in enumerate(M)
    }
    return D

# --- Dealing with components ---
def get_components(graph):
    return [
        graph.subgraph(component).copy()
        for component in nx.connected_components(graph)
    ]