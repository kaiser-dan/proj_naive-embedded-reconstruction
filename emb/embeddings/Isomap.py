"""Project source code for applying Isomap embedding.
"""
# ============= SET-UP =================
__all__ = ["embed_Isomap", "embed_multiplex_Isomap"]

# --- Standard library ---
# import sys
import warnings
import time

# --- Scientific computing ---
import numpy as np
from sklearn import manifold

# --- Network science ---
import networkx as nx


# --- Misc ---
# from rich import print
from loguru import logger as LOGGER

LOGGER.add(".logs/emb-debug_isomap.log", level='DEBUG')

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============= CLASSES =================
class NetworkBase:
    """
    Base class for networks, provides some basic operations
    """

    def __init__(self):
        pass

    def get_largest_component(self, directed=False):
        """
        Return the largest connected component subgraph for undirected networks
        Return the largest weakly connected component subgraph for directed
        networks
        """
        if directed:
            # for networkx version older than 2.1
            # return max(nx.weakly_connected_component_subgraphs(self.G), key=len)
            return self.G.subgraph(
                max(nx.weakly_connected_components(self.G), key=len)
            ).copy()
        else:
            # return max(nx.connected_component_subgraphs(self.G), key=len)
            return self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()

    def convert_to_undirected(self):
        """
        Convert the graph to undirected.
        """
        self.G = self.G.to_undirected()

    def remove_selfloops(self):
        """
        Remove selfloops edges.
        """
        self.G.remove_edges_from(self.G.selfloop_edges())

    def remove_parallel_edges(self):
        """
        Remove parallel edges.
        """
        self.G = nx.Graph(self.G)

    def generate_shortest_path_length_matrix(self):
        if hasattr(self, "shortest_path_length_matrix"):
            return

        LOGGER.info("Precomputing shortest path lengths")
        sw = time.perf_counter()

        number_of_nodes = len(self.G.nodes)
        self.index_nodes()
        diameter = nx.diameter(self.G)
        self.shortest_path_length_matrix = (diameter + 1) * np.ones(
            (number_of_nodes, number_of_nodes)
        )
        shortest_path_length_dict = dict(nx.shortest_path_length(self.G))
        LOGGER.debug(
            f"Shortest path length dictionary computed ({(time.perf_counter() - sw):.3f} s)"
        )

        LOGGER.debug("Filling data structure")
        sw = time.perf_counter()
        for node1, path_nbrs in shortest_path_length_dict.items():
            for node2, path_length in path_nbrs.items():
                self.shortest_path_length_matrix[node1][node2] = path_length
        LOGGER.debug(f"Data structure filled ({(time.perf_counter() - sw):.3f} s)")

        LOGGER.info("Shortest path length matrix computed")

    def index_nodes(self):
        if not hasattr(self, "id2node"):
            self.id2node = dict(enumerate(self.G.nodes))

        if not hasattr(self, "node2id"):
            self.node2id = {value: key for key, value in self.id2node.items()}

    def dump_network(self, path):
        nx.write_edgelist(self.G, path)


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

        LOGGER.debug(f"Geodesic distances: {self.network.shortest_path_length_matrix}")

        LOGGER.info("Embedding geodesic distances")
        sw = time.perf_counter()
        self.embeddings_matrix = self._get_embedding()
        LOGGER.info("Embedding completed")
        LOGGER.debug(f"Embedding time = {(time.perf_counter() - sw):.3f} s")

        LOGGER.debug("Filling data structure")
        sw = time.perf_counter()
        if len(self.embeddings_matrix) == 0:
            self.embeddings = {}
        else:
            self.embeddings = {
                self.network.id2node[i]: self.embeddings_matrix[i]
                for i in range(len(self.network.id2node))
            }
        LOGGER.debug(f"Data structure filled ({(time.perf_counter() - sw):.3f} s)")

        return self.embeddings

        LOGGER.debug(f"Data structure filled ({(time.perf_counter() - sw):.3f} s)")


class Isomap(MDSBase):
    def __init__(self, dimensions=2):
        super(Isomap, self).__init__(dimensions)

    def _get_embedding(self):
        return manifold.MDS(self.dimension, dissimilarity="precomputed").fit_transform(
            self.network.shortest_path_length_matrix
        )


# ============= FUNCTIONS =================
# --- Driver ---
def embed_Isomap(graph, **kwargs):
    """Embed `graph` using Isomap.

    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    """
    # Cast as NetworkBase class object for hypercomparison
    G_ = NetworkBase()
    G_.G = graph

    return Isomap(**kwargs).train(G_)


def embed_multiplex_Isomap(multiplex, **kwargs):
    vectors = {
        label: embed_Isomap(graph, **kwargs) for label, graph in multiplex.items()
    }

    return vectors
