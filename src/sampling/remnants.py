"""
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
    def __init__(
            self,
            remnant: nx.Graph,  # remnant graph, implicitly some A \cup \Theta_l
            known_edges: set[tuple[int, int]],  # set of edges considered known a priori, implicitly \Theta_l
            unknown_edges: set[tuple[int, int]],  # set of edges _not_ known a priori, implicitly A
            name: Union[None, str] = None):  # name of remnant, if applicable
        # Data assignment
        self.remnant = remnant
        self.known_edges = known_edges
        self.unknown_edges = unknown_edges

        self._name = name

        return


    # --- Properties ---
    @property
    def name(self):  # name is read-only
        return self._name


    # --- Private methods ---


    # --- Public methods ---
    # > I/O >
    def save(self, filepath: str, only_graph: bool = False):
        save_remnant(self, filepath, only_graph)


class ExperimentRemnant(Remnant):
    def __init__(
            self,
            remnant: nx.Graph,
            known_edges: set[tuple[int, int]],
            unknown_edges: set[tuple[int, int]],
            theta: Union[None, float] = None,
            observation_strategy: Union[None, str] = None,
            name: Union[None, str] = None):
        # Inheritance initialization
        super().__init__(self, remnant, known_edges, unknown_edges, name)

        # Data assignment
        self._theta = theta
        self._observation_strategy = observation_strategy

        return

    # --- Properties ---
    @property
    def theta(self):  # read-only
        """Partial fraction of observation, i.e., relative size of training set"""
        return self._theta

    @property
    def observation_strategy(self):  # read-only
        """Method of observation training set"""
        return self._observation_strategy


    # --- Private methods ---


    # --- Public methods ---

# ============= FUNCTIONS =================
def save_remnant(remnant: Union[Remnant, ExperimentRemnant], filepath: str, only_graph: bool = False):
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