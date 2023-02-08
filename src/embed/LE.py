"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
# --- Scientific computing ---
from scipy.sparse.linalg import eigsh  # eigensolver

# --- Network science ---
import networkx as nx


# ============= FUNCTIONS =================
# --- Helpers ---
def _reindex_nodes(graph):
    reindexed_nodes = {
        index: new_index
        for new_index, index in enumerate(sorted(graph.nodes()))
    }  # Allow for non-contiguous node indices
    return reindexed_nodes


# --- Driver ---
def LE(graph, parameters, hyperparameters):
    """Embed `graph` using Laplacian eigenmaps.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for LE parameter selection.
    hyperparameters : dict
        Keyword arguments for ARPACK convergence parameters.

    Returns
    -------
    ~~np.array~~
    dict
        Map of node ids to embedded vectors (as rows).

    """
    # >>> Book-keeping >>>
    reindexed_nodes = _reindex_nodes(graph)  # fix networkx indexing
    vectors = dict()
    # ! >>> BROKEN >>>
    # ! Non-contiguous indexing in some real remnants is causing
    # ! indexing errors with arrays - generalizing to a dict instead
    # ! All downstream analyses are able to proceed
    # vectors = np.zeros(
    #     (graph.number_of_nodes(), parameters["k"])  # needs k for scipy.sparse.linalg.eigsh
    # )  # initialize embedded vectors
    # ! <<< BROKEN <<<
    # <<< Book-keeping <<<

    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph.nodes()))

    # Compute the eigenspectra of the normalized Laplacian matrix
    # ! >>> BROKEN >>>
    #_, eigenvectors = \
    #    eigsh(L, **parameters, **hyperparameters)
    # ! <<< BROKEN <<<
    # ! >>> HOTFIX >>>
    _, eigenvectors = \
        eigsh(
            L, k=parameters["k"],
            maxiter=hyperparameters["maxiter"],
            tol=hyperparameters["tol"],
            ncv=hyperparameters["NCV"]*graph.number_of_nodes()
        )
    # ! <<< HOTFIX <<<

    # Apply node reindexing (thanks networkx :/)
    for index, new_index in reindexed_nodes.items():
        vectors[index] = eigenvectors[new_index]

    return vectors


def LE_per_component(graph, parameters, hyperparameters):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    # <<< Book-keeping <<<

    # Retrieve each component as a graph
    component_subgraphs = [
        graph.subgraph(component).copy()
        for component in nx.connected_components(graph)
    ]

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(
            LE(component_subgraph, parameters, hyperparameters)
        )

    return vectors_per_component
