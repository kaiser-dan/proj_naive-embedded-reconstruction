"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
__all__ = ["embed_LE", "embed_multiplex_LE"]

# --- Scientific computing ---
from scipy.sparse.linalg import eigsh  # eigensolver

# --- Network science ---
import networkx as nx


# ============= FUNCTIONS =================
# ? Apply contiguous inverse mapping before return?
def embed_LE(graph, **kwargs):
    nodelist = sorted(graph.nodes())

    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=nodelist)

    # Compute the eigenspectra of the normalized Laplacian matrix
    _, eigenvectors = eigsh(L,
        which="SM", maxiter=100*L.shape[0], tol=1e-4,
        **kwargs)

    return eigenvectors

def embed_multiplex_LE(multiplex, **kwargs):
    vectors = {
        label: embed_LE(graph, **kwargs)
        for label, graph in multiplex.items()
    }

    return vectors