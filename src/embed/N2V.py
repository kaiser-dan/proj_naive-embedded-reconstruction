"""Project source code for applying Node2Vec embedding.

Wrapper for node2vec package with
additional post-processing for non-consecutive node ids.
"""
# ============= SET-UP =================
# --- Network science ---
from node2vec import Node2Vec


# ============= FUNCTIONS =================
def N2V(graph, parameters, hyperparameters):
    """Embed `graph` using node2vec.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for node2vec walk generation.
    hyperparameters : dict
        Keyword arguments for word2vec fitting on node2vec-generated walks.

    Returns
    -------
    dict
        Map of node ids to embedded vectors.
    """
    # Sample random walks
    embedding_model = Node2Vec(graph, **parameters)

    # Embed walks with word2vec and retrieve model
    embedding_model = embedding_model.fit(**hyperparameters)
    embedding_model = embedding_model.wv

    # Retrieve resultant vectors
    vectors = embedding_model.vectors

    # Retrieve word2vec internal hash of node ids to vector indices
    node_labels = embedding_model.index_to_key

    # Map node ids into corresponding vector
    # This accounts for graphs with non-consecutive node ids
    embedding = {
        int(node_label): vectors[node_index]
        for node_index, node_label in enumerate(node_labels)
    }

    return embedding
