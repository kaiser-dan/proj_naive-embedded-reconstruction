"""Source code specifying Remnant class.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import pickle
from typing import Union

# --- Scientific computation ---
import networkx as nx

# ============= CLASSES =================
class Remnant:
    """Class for organizing layer remnant data.

    Data
    ----
    remnant : nx.Graph
        Graph object containing remnant topology.
    known_edges : set[tuple[int, int]]
        Collection of edges in the training set.
    unknown_edges : set[tuple[int, int]]
        Collection of edges in the test set.
    theta : Union[None, float]
        Relative size of training set. `None` if unspecificed/unknown.
    observation_strategy : Union[None, str]
        Method describing strategy for gathering training set. `None` is unspecified/unknown.
    name : Union[None, str]
        System name. `None` if unspecified/unknown.

    Methods
    -------
    save(filepath: str, only_graph: bool)
        Saves object to the given filepath. `only_graph` will save only an edgelist.
    """
    def __init__(
            self,
            remnant: nx.Graph,
            known_edges: set[tuple[int, int]],
            unknown_edges: set[tuple[int, int]],
            theta: Union[None, float] = None,
            observation_strategy: Union[None, str] = None,
            name: Union[None, str] = None):
        # Data assignment
        self.remnant = remnant
        self.known_edges = known_edges
        self.unknown_edges = unknown_edges
        self._theta = theta
        self._observation_strategy = observation_strategy
        self._name = name

        return

    # --- Properties ---
    # NOTE: All properties are read-only
    @property
    def theta(self):
        """Partial fraction of observation, i.e., relative size of training set"""
        return self._theta

    @property
    def observation_strategy(self):
        """Method of observation training set"""
        return self._observation_strategy

    @property
    def name(self):
        """System name of remnant"""
        return self._name

    # --- Private methods ---
    def __str__(self):
        str_ = \
        f"""
        ================
        Remnant Instance
        ----------------
        Graph with {self.remnant.number_of_nodes()} nodes and {self.remnant.number_of_edges()} edges.

        Metadata:
            name: {self.name}
            observation_strategy: {self.observation_strategy}
            theta: {self.theta}
        ================
        """
        return str_

    def __eq__(self, other):
        return self.remnant == other.remnant

    # --- Public methods ---
    def save(self, filepath: str, only_graph: bool = False):
        save_remnant(self, filepath, only_graph)

# ============= FUNCTIONS =================
def save_remnant(remnant: Remnant, filepath: str, only_graph: bool = False):
    if only_graph:
        nx.write_edgelist(remnant.remnant, filepath)
    else:
        try:
            fh = open(filepath, "wb")
            pickle.dump(remnant, fh, pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            sys.stderr.write(f"{err}\n Error serializing Remnant instance!")
        finally:
            fh.close()

def build_remnants(
        G: nx.Graph,
        H: nx.Graph,
        training_edges: dict[tuple[int, int], int],
        test_edges: dict[tuple[int, int], int],
        theta: Union[None, float] = None,
        observation_strategy: Union[None, str] = None,
        name: Union[None, str] = None):
    known_edges_G = {edge for edge, layer in training_edges.items() if layer == 1}
    known_edges_H = {edge for edge, layer in training_edges.items() if layer == 0}
    unknown_edges_G = {edge for edge, layer in test_edges.items() if layer == 1}
    unknown_edges_H = {edge for edge, layer in test_edges.items() if layer == 0}

    if name is None:
        name = "None"

    R_G = Remnant(G, known_edges_G, unknown_edges_G, theta, observation_strategy, name+"_layer-1")
    R_H = Remnant(H, known_edges_H, unknown_edges_H, theta, observation_strategy, name+"_layer-2")

    return R_G, R_H