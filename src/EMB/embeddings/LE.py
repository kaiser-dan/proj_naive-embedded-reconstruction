"""Project source code for applying Laplacian Eigenmap embedding.
"""
# ============= SET-UP =================
__all__ = ["embed_LE", "embed_multiplex_LE"]

# --- Scientific computing ---
from scipy.sparse.linalg import eigsh  # eigensolver
from scipy.linalg import eigh  # dense eigensolver (remnants of small networks)

# --- Network science ---
import networkx as nx

from . import LOGGER


# ============= FUNCTIONS =================
# ? Apply contiguous inverse mapping before return?
def embed_LE(graph, nodelist=None, **kwargs):
    if nodelist is None:
        nodelist = sorted(graph.nodes())

    num_shared_nodes = len(set(graph.nodes()) & set(nodelist))
    num_total_nodes = len(set(graph.nodes()) | set(nodelist))
    if num_shared_nodes / num_total_nodes < 1:
        graph.add_nodes_from(nodelist)

    LOGGER.debug(
        f"Number of nodes present = {graph.number_of_nodes()} (nx) or {len(nodelist)} (nodelist)"
    )

    # Calculate normalized Laplacian matrix
    L = nx.normalized_laplacian_matrix(graph, nodelist=nodelist)

    # Ensure valid dimension
    kwargs["k"] = min(kwargs["k"], graph.number_of_nodes())

    # Compute the eigenspectra of the normalized Laplacian matrix
    try:
        _, eigenvectors = eigsh(
            L, which="SM", maxiter=100 * L.shape[0], tol=1e-4, **kwargs
        )
    except TypeError:
        LOGGER.info("Encountered type error, retrying with dense eigensolver...")
        _, eigenvectors = eigh(L.toarray())
    except Exception:
        LOGGER.critical("Previously unencountered error!")

    return eigenvectors[:, 1:]


def embed_multiplex_LE(multiplex, **kwargs):
    nodelist = set()
    for graph in multiplex.values():
        nodelist.update(graph.nodes())
    nodelist = sorted(list(nodelist))

    vectors = {
        label: embed_LE(graph, nodelist=nodelist, **kwargs)
        for label, graph in multiplex.items()
    }

    return vectors
