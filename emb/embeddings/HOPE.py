"""Project source code for applying HOPE embedding.
"""
# ============= SET-UP =================

__all__ = ["embed_HOPE", "embed_multiplex_HOPE"]

# --- Scientific computing ---
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing

# --- Network science ---
import networkx as nx

from loguru import logger as LOGGER

# ============= CLASSES =================
class HOPE_:
    """
    Ou, Mingdong, et al. Asymmetric transitivity preserving graph embedding.
    Proceedings of the 22nd ACM SIGKDD international conference on Knowledge
    discovery and data mining. ACM, 2016.

    Implementation borrowed from https://github.com/THUDM/cogdl/blob/master/cogdl/models/emb/hope.py
    with modification
    This implementation use Katz similarity of the nodes, which is claimed to
    be the best in the paper

    HOPE use the numpy linear algebra lib, which by default uses multiple threads.
    To disable it, export OPENBLAS_NUM_THREADS=1 or MKL_NUM_THREADS=1
    in shell depending on the backend of the numpy installation.
    """
    def __init__(self, dimensions=128, beta=0.01):
        self.dimension = dimensions
        self.beta = beta

    def train(self, G):
        self.G = G
        self.id2node = dict(zip(range(len(G)), G))

        adj = nx.adjacency_matrix(self.G).todense()
        n = adj.shape[0]
        # The author claim that Katz has superior performance in related tasks
        # S_katz = (M_g)^-1 * M_l = (I - beta*A)^-1 * beta*A = (I - beta*A)^-1 * (I - (I -beta*A))
        #        = (I - beta*A)^-1 - I
        katz_matrix = np.asarray((np.eye(n) - self.beta * np.mat(adj)).I - np.eye(n))
        self.embeddings_matrix = self._get_embedding(katz_matrix, self.dimension)
        #center_point = self.embeddings_matrix.mean(axis=0) # centering the coordinates
        #self.embeddings_matrix -= center_point # centering the coordinates
        self.embeddings = {
            int(self.id2node[i]): self.embeddings_matrix[i] for i in range(len(self.id2node))
        }

        return self.embeddings

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut and vt
        ut, s, vt = sp.linalg.svds(matrix, int(dimension / 2))
        emb_matrix_1, emb_matrix_2 = ut, vt.transpose()

        emb_matrix_1 = emb_matrix_1 * np.sqrt(s)
        emb_matrix_2 = emb_matrix_2 * np.sqrt(s)
        emb_matrix_1 = preprocessing.normalize(emb_matrix_1, "l2")
        emb_matrix_2 = preprocessing.normalize(emb_matrix_2, "l2")
        features = np.hstack((emb_matrix_1, emb_matrix_2))
        return features


# ============= FUNCTIONS =================
def embed_HOPE(graph, **kwargs):
    # Verify 0 < k < num_nodes
    if kwargs.get("dimensions", np.inf) >= graph.number_of_nodes():
        LOGGER.warning("Encountered graph with n <= dimensions; forcing dimensions = n-1.")
        kwargs["dimensions"] = graph.number_of_nodes() - 1
    vectors = HOPE_(**kwargs).train(graph)

    return vectors

def embed_multiplex_HOPE(multiplex, **kwargs):
    vectors = {
            label: embed_HOPE(graph, **kwargs)
            for label, graph in multiplex.items()
        }

    return vectors
