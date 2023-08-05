"""Project source code for applying HOPE embedding.
"""
# ============= SET-UP =================
# --- Standard library ---
import os

# --- Scientific computing ---
import numpy as np
import scipy.sparse as sp
from sklearn import preprocessing

# --- Network science ---
import networkx as nx

# --- Project source code ---
from embed.helpers import reindex_nodes, get_components, matrix_to_dict
from embed.embedding import Embedding


# ============= CLASSES =================
class HOPE:
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
    def __init__(self, dimension=128, beta=0.01):
        self.dimension = dimension
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
            str(self.id2node[i]): self.embeddings_matrix[i] for i in range(len(self.id2node))
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
# --- Driver ---
def HOPE(
        graph: nx.Graph,
        parameters: dict, hyperparameters: dict = dict(),
        per_component: bool = False):
    """Embed `graph` using Isomap.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for HOPE parameter selection.
    hyperparameters : dict
        <NOT IMPLEMENTED FOR HOPE>
    per_component: bool, optional
        Embed each graph component separately, by default False.

    Returns
    -------
    Embedding
        Embedding class instance.
    """
    # >>> Book-keeping >>>
    # Declare sub-method
    _dispatch = _HOPE if not per_component else _HOPE_per_component

    # Relabel nodes contiguously
    node_index = reindex_nodes(graph)

    # Initialize return struct
    vectors = dict()  # node label -> vector
    # <<< Book-keeping <<<

    # Get vectors
    vectors = _dispatch(graph, parameters)

    # Converting type
    if eigenvectors is np.ndarray:
        eigenvectors = matrix_to_dict(eigenvectors)

    # Apply node reindexing
    for node, node_adjusted in node_index.items():
        vectors[node] = eigenvectors[node_adjusted]

    # Construct Embedding instance
    embedding = Embedding(vectors, "HOPE" if not per_component else "HOPE-PC")

    return embedding

# --- Primary computations ---
def _HOPE(graph, parameters):
    vectors = HOPE(**parameters).train(graph)

    return vectors

def _HOPE_per_component(graph, parameters):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # Retrieve each component as a graph
    component_subgraphs = get_components(graph)

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(
            _HOPE(component_subgraph, parameters)
        )

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector

    return vectors
