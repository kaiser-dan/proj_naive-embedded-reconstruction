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

    get_components(descending: bool)
        Returns connected components of `remnant`. Will be sorted in descending order according to size if `descending` is True.
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

    def get_components(self, descending: bool = True):
        return sorted(nx.connected_components(self.remnant), key=len, reverse=descending)

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

def build_remnant(raw_graph: nx.Graph,
        training_edges,
        test_edges,
        layer_id,
        theta,
        observation_strategy,
        name):
    # >>> Book-keeping >>>
    # Set non-None name for filepath downstream
    if name is None:
        name = f"None_layer-{layer_id}"
    # <<< Book-keeping <<<

    # Calculate remnant graph
    ## Add nodes
    remnant_graph = nx.Graph()
    remnant_graph.add_nodes_from(raw_graph)

    ## Get edges
    known_edges = {
        edge
        for edge, layer in training_edges.items()
        if layer == layer_id
    }
    unknown_edges = set(test_edges.keys())
    remnant_edges = known_edges | unknown_edges

    ## Add edges
    remnant_graph.add_edges_from(remnant_edges)

    # Build Remnants object
    remnant = Remnant(
        remnant_graph,
        known_edges,
        unknown_edges,
        theta,
        observation_strategy,
        name
    )

    return remnant

def build_remnants(
        G, H,
        training_edges, test_edges,
        theta, observation_strategy=None, name=None):
    R_G = build_remnant(
        G,
        training_edges, test_edges, 1,
        theta, observation_strategy, name
    )
    R_H = build_remnant(
        H,
        training_edges, test_edges, 0,
        theta, observation_strategy, name
    )

    return R_G, R_H