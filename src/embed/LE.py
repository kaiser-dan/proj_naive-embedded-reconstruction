"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
# --- Scientific computing ---
from numpy import ndarray

from scipy.sparse.linalg import eigsh  # eigensolver
from scipy.linalg import eigh  # eigensolver for dense matrices

# --- Network science ---
import networkx as nx

# --- Miscellaneous ---
from embed.helpers import reindex_nodes, get_components, matrix_to_dict
from embed.embedding import Embedding


# ============= FUNCTIONS =================
# --- Driver ---
def LE(
        graph: nx.Graph,
        parameters: dict, hyperparameters: dict,
        per_component: bool = False,
        nodelist: list|None = None):
    """Embed `graph` using Laplacian eigenmaps.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for LE parameter selection.
    hyperparameters : dict
        Keyword arguments for ARPACK convergence parameters.
    per_component: bool, optional
        Embed each graph component separately, by default False.
    nodelist: list|None, optional
        Node order for normalized laplacian matrix presentation, by default sorted index.

    Returns
    -------
    Embedding
        Embedding class instance.

    """
    # >>> Book-keeping >>>
    _dispatch = _LE  # default embedding sub-method

    # ! >>> Temp NCV fix >>>
    if hyperparameters.get("ncv") is not None:
        del hyperparameters["ncv"]
    # ! <<< Temp NCV fix <<<

    node_index = reindex_nodes(graph)  # relabeling node labels -> contiguous node labels

    # Homogenize node sorting for adjacency/Laplacian matrices
    if nodelist is None:
        nodelist = sorted(graph.nodes())

    vectors = dict()  # output struct, node label -> vector
    # <<< Book-keeping <<<

    # >>> Dispatch >>>
    if per_component:
        return _LE_per_component(graph, parameters, hyperparameters, nodelist)

    if parameters["k"] >= graph.number_of_nodes():
        _dispatch = _LE_dense

    eigenvectors = _dispatch(graph, parameters, hyperparameters, nodelist)
    # <<< Dispatch <<<

    # >>> Post-processing >>>
    # Converting type
    if type(eigenvectors) == ndarray:
        eigenvectors = matrix_to_dict(eigenvectors)

    # Remove first coordinate of eigenvectors (proportional to node degree)
    eigenvectors = {
        node: vector[1:]
        for node, vector in eigenvectors.items()
    }

    # Apply node reindexing
    for node, node_adjusted in node_index.items():
        vectors[node] = eigenvectors[node_adjusted]
    # <<< Post-processing <<<

    embedding = Embedding(vectors, "LE" if not per_component else "LE-PC")

    return embedding


# --- Main computations ---
def _LE(graph, parameters, hyperparameters, nodelist):
    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=nodelist)

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigsh(L, **parameters, **hyperparameters)

    return eigenvectors


def _LE_dense(graph, parameters, hyperparameters, nodelist):
    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=nodelist)

    # Densify matrix
    L = L.toarray()

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigh(L)

    return eigenvectors


def _LE_per_component(graph, parameters, hyperparameters, nodelist):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # Retrieve each component as a graph
    component_subgraphs = get_components(graph)

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        component_nodelist = [node for node in nodelist if node in component_subgraph.nodes()]
        vectors_per_component.append(
            LE(component_subgraph, parameters, hyperparameters, nodelist=component_nodelist)
        )

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector

    return vectors
