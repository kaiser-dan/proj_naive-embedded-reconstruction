"""Project source code for applying Node2Vec embedding.

Wrapper for node2vec package with
additional post-processing for non-consecutive node ids.
"""
# ============= SET-UP =================
# --- Scientific computing ---

# --- Network science ---
import networkx as nx
from node2vec import Node2Vec

# --- Miscellaneous ---
from embed.helpers import get_contiguous_vectors

# ============= FUNCTIONS =================
def N2V(graph: nx.Graph, parameters: dict, hyperparameters: dict):
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
    embedding = get_contiguous_vectors(embedding_model)

    return embedding


def N2V_per_component(graph: nx.Graph, parameters: dict, hyperparameters: dict):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # >>> Embedding >>>
    # Retrieve each component as a graph
    component_subgraphs = sorted(
        [
            graph.subgraph(component).copy()
            for component in nx.connected_components(graph)
        ],
        key=len,reverse=True
    )

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(N2V(component_subgraph, parameters, hyperparameters))

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector
    # <<< Embedding <<<

    return vectors