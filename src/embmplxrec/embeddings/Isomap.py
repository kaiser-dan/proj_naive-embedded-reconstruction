"""Project source code for applying Isomap embedding.
"""
# ============= SET-UP =================
# --- Standard library ---
import os

# --- Scientific computing ---
from numpy import ndarray
from sklearn import manifold

# --- Network science ---
import networkx as nx

# --- Project source code ---
from embmplxrec.hypercomparison import networks
from embmplxrec.embeddings.helpers import reindex_nodes, get_components, matrix_to_dict
from embmplxrec.embeddings.embedding import Embedding


# ============= CLASSES =================
"""
Implementations of Isomap
Metric MDS + shortest path length matrix of a network
"""
class MDSBase:
    def __init__(self, dimensions):
        self.dimension = dimensions

    def train(self, network):
        self.network = network
        self.network.generate_shortest_path_length_matrix()
        self.embeddings_matrix = self._get_embedding()

        if len(self.embeddings_matrix) == 0:
            self.embeddings = {}
        else:
            self.embeddings = {
                self.network.id2node[i]: self.embeddings_matrix[i] for i in range(len(self.network.id2node))
            }
        return self.embeddings


class Isomap_(MDSBase):
    def __init__(self, dimensions=2):
        super(Isomap_, self).__init__(dimensions)

    def _get_embedding(self):
        return manifold.MDS(
            self.dimension, dissimilarity="precomputed").fit_transform(
                self.network.shortest_path_length_matrix)

# ============= FUNCTIONS =================
# --- Driver ---
def Isomap(
        graph: nx.Graph,
        parameters: dict, hyperparameters: dict = dict(),
        per_component: bool = False):
    """Embed `graph` using Isomap.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for Isomap parameter selection.
    hyperparameters : dict
        <NOT IMPLEMENTED FOR ISOMAP>
    per_component: bool, optional
        Embed each graph component separately, by default False.

    Returns
    -------
    Embedding
        Embedding class instance.
    """
    # >>> Book-keeping >>>
    # Declare sub-method
    _dispatch = _Isomap if not per_component else _Isomap_per_component

    # Relabel nodes contiguously
    node_index = reindex_nodes(graph)

    # Initialize return struct
    vectors_return = dict()  # node label -> vector
    # <<< Book-keeping <<<

    # Get vectors
    vectors = _dispatch(graph, parameters)

    # Converting type
    if vectors is ndarray:
        vectors = matrix_to_dict(vectors)

    # Apply node reindexing
    # ! >>> BROKEN >>>
    #for node, node_adjusted in node_index.items():
    #    vectors_return[node] = vectors[node_adjusted]
    # ! --- HOT-FIX ---
    vectors_return = vectors
    # ! <<< BROKEN <<<

    # Construct Embedding instance
    embedding = Embedding(vectors_return, "Isomap" if not per_component else "Isomap-PC")

    return embedding

# --- Primary computations ---
def _Isomap(graph, parameters):
    # Cast as NetworkBase class object for hypercomparison
    G_ = networks.NetworkBase()
    G_.G = graph

    vectors = Isomap_(**parameters).train(G_)

    return vectors

def _Isomap_per_component(graph, parameters):
    # >>> Book-keeping >>>
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # <<< Book-keeping <<<

    # Retrieve each component as a graph
    component_subgraphs = get_components(graph)

    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(
            _Isomap(component_subgraph, parameters)
        )

    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector

    return vectors
