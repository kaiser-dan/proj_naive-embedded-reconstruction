"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
__all__ = ["embed_LE", "embed_multiplex_LE"]

# --- Scientific computing ---
from scipy.sparse.linalg import eigsh  # eigensolver

# --- Network science ---
import networkx as nx

from . import LOGGER

# ============= FUNCTIONS =================
# ? Apply contiguous inverse mapping before return?
def embed_LE(graph, nodelist=None, **kwargs):
    if nodelist == None:
        nodelist = sorted(graph.nodes())

    if len(set(graph.nodes()) & set(nodelist)) / len(set(graph.nodes()) | set(nodelist)) < 1:
        graph.add_nodes_from(nodelist)

    # Calculate normalized Laplacian matrix
    try:
        L = nx.normalized_laplacian_matrix(graph, nodelist=nodelist)
    except:
        LOGGER.critical()

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigsh(L,
        which="SM", maxiter=100*L.shape[0], tol=1e-4,
        **kwargs)

    return eigenvectors[:, 1:]

def embed_multiplex_LE(multiplex, **kwargs):
    nodelist = sorted(multiplex[1].nodes())
    vectors = {
        label: embed_LE(graph, nodelist=nodelist, **kwargs)
        for label, graph in multiplex.items()
    }

    return vectors