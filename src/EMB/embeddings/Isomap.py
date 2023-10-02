"""Project source code for applying Isomap embedding.
"""
# ============= SET-UP =================
__all__ = ["embed_Isomap", "embed_multiplex_Isomap"]

# --- Standard library ---
import warnings

# --- Scientific computing ---
import numpy as np
from sklearn import manifold

# --- Network science ---
import networkx as nx


# --- Globals ---
from . import LOGGER
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
            #return max(nx.weakly_connected_component_subgraphs(self.G), key=len) #for networkx version older than 2.1
            return self.G.subgraph(max(nx.weakly_connected_components(self.G), key=len)).copy()
        else:
            #return max(nx.connected_component_subgraphs(self.G), key=len)
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
        if hasattr(self, 'shortest_path_length_matrix'):
            return

        number_of_nodes = len(self.G.nodes)
        self.index_nodes()

        self.shortest_path_length_matrix = np.zeros((
            number_of_nodes, number_of_nodes
        ))
        shortest_path_length_dict = dict(nx.shortest_path_length(self.G))

        for nodeid1 in range(number_of_nodes):
            for nodeid2 in range(number_of_nodes):
                try:
                    l = shortest_path_length_dict[self.id2node[nodeid1]][self.id2node[nodeid2]]
                except KeyError as err:
                    LOGGER.debug(f"KeyError in path calculations {self.id2node[nodeid1]} -> {self.id2node[nodeid2]}; forcing path distance as N+1")
                    l = self.G.number_of_nodes()+1
                    self.shortest_path_length_matrix[nodeid1][nodeid2] = l
                except Exception as err:
                    LOGGER.error(err)
                    raise err

    def index_nodes(self):
        if not hasattr(self, 'id2node'):
            self.id2node = dict(enumerate(self.G.nodes))

        if not hasattr(self, 'node2id'):
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
        self.embeddings_matrix = self._get_embedding()

        if len(self.embeddings_matrix) == 0:
            self.embeddings = {}
        else:
            self.embeddings = {
                self.network.id2node[i]: self.embeddings_matrix[i] for i in range(len(self.network.id2node))
            }
        return self.embeddings


class Isomap(MDSBase):
    def __init__(self, dimensions=2):
        super(Isomap, self).__init__(dimensions)

    def _get_embedding(self):
        return manifold.MDS(
            self.dimension, dissimilarity="precomputed").fit_transform(
                self.network.shortest_path_length_matrix)

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
        label: embed_Isomap(graph, **kwargs)
        for label, graph in multiplex.items()
    }

    return vectors
