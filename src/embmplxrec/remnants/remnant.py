"""Classes for handling remnant multiplexes in the multiplex reconstruction setting.
"""
# ============= SET-UP =================
# --- Standard library ---
import pickle
from dataclasses import dataclass, field
from typing import List

# --- Scientific computation ---
import networkx as nx

# --- Source code ---
from embmplxrec.data import io

# --- Aliases ---
from embmplxrec._types import AbstractEdges


# ============= CLASSES =================
@dataclass(eq=False, frozen=True)
class RemnantNetwork:
    graph: nx.Graph
    observed: AbstractEdges
    unobserved: AbstractEdges
    theta: float
    metadata: dict = field(default_factory=dict)

    # --- Public methods ---
    def save(self, filepath: str, safe_save: bool = True):
        if safe_save:
            io.safe_save(self, filepath)
        else:
            with open(filepath, 'wb') as _fh:
                pickle.dump(self, _fh, pickle.HIGHEST_PROTOCOL)

@dataclass(eq=False) #, frozen=True)
class RemnantMultiplex:
    # Specified data
    layers: List[RemnantNetwork]
    labels: List[int]
    metadata: dict = field(default_factory=dict)
    # Inferred data
    labels_to_layers: dict = field(init=False, repr=False)  # label (int) -> layer (RemnantNetwork)
    observed: AbstractEdges = field(init=False, repr=False)
    unobserved: AbstractEdges = field(init=False, repr=False)

    # --- Private methods ---
    def __post_init__(self):
        # Construct labels <-> layers
        self._build_labels_to_layers()

        # Aggregate observed and unobserved edge sets
        self._build_edge_sets()

    def _build_labels_to_layers(self):
        self.labels_to_layers = dict()
        # TODO: Check for label contiguity
        for idx, label in enumerate(self.labels):
            self.labels_to_layers[label] = self.layers[idx]

    def _build_edge_sets(self):
        self.observed = set()
        self.unobserved = set()
        for layer in self.layers:
            self.observed.update(layer.observed)
            self.unobserved.update(layer.unobserved)

    def _check_nodes(self):
        node_counts = [
            layer.graph.number_of_nodes()
            for layer in self.layers
        ]
        num_nodes_baseline = node_counts[0]

        assert [num_nodes == num_nodes_baseline for num_nodes in node_counts].all()

    # --- Public methods ---
    def save(self, filepath: str, safe_save: bool = True):
        if safe_save:
            io.safe_save(self, filepath)
        else:
            with open(filepath, 'wb') as _fh:
                pickle.dump(self, _fh, pickle.HIGHEST_PROTOCOL)