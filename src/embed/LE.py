"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
# --- Scientific computing ---
from scipy.sparse.linalg import eigsh  # eigensolver
from scipy.linalg import eigh  # eigensolver for dense matrices

# --- Network science ---
import networkx as nx

# --- Miscellaneous ---
from embed.helpers import reindex_nodes, get_components

# ============= FUNCTIONS =================
# --- Driver ---
def LE(graph, parameters, hyperparameters, per_component: bool = False):
    """Embed `graph` using Laplacian eigenmaps.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for LE parameter selection.
    hyperparameters : dict
        Keyword arguments for ARPACK convergence parameters.
    per_component: bool
        Embed each graph component separately, by default False.

    Returns
    -------
    dict
        Map of node ids to embedded vectors.

    """
    # >>> Book-keeping >>>
    # TODO Fill in notes
    _dispatch = _LE  # default embedding sub-method

    # ! >>> Temp NCV fix >>>
    if hyperparameters.get("ncv") is not None:
        del hyperparameters["ncv"]
    # ! <<< Temp NCV fix <<<

    node_index = reindex_nodes(graph)  # relabeling node labels -> contiguous node labels

    vectors = dict()  # output struct, node label -> vector
    # <<< Book-keeping <<<

    # >>> Dispatch >>>
    if per_component:
        return _LE_per_component(graph, parameters, hyperparameters)

    if parameters["k"] >= graph.number_of_nodes():
        _dispatch = _LE_dense

    eigenvectors = _dispatch(graph, parameters, hyperparameters)
    # <<< Dispatch <<<

    # >>> Post-processing >>>
    # Apply node reindexing
    for node, node_adjusted in node_index.items():
        vectors[node] = eigenvectors[node_adjusted]
    # <<< Post-processing <<<

    return vectors


# --- Main computations ---
def _LE(graph, parameters, hyperparameters):
    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigsh(L, **parameters, **hyperparameters)

    return eigenvectors


def _LE_dense(graph, parameters, hyperparameters):
    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))

    # Densify matrix
    L = L.toarray()

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigh(L)

    return eigenvectors


def _LE_per_component(graph, parameters, hyperparameters):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # Retrieve each component as a graph
    component_subgraphs = get_components(graph)

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(
            LE(component_subgraph, parameters, hyperparameters)
        )

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector

    return vectors