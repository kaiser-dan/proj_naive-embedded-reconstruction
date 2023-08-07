# --- Standard library ---
import sys
import os

# --- Network science ---
import networkx as nx

# --- Test utility ---
import pytest
from dataclasses import FrozenInstanceError  # exception we test capturing

# --- Project source code ---
from embmplxrec.remnants import remnant

# ========== Fixtures ==========
# --- Test Fixtures ---
@pytest.fixture
def simple_remnant():
    G = nx.path_graph(10)
    observed = {
        (0, 1),
        (1, 2),
        (2, 3)
    }
    unobserved = set(G.edges()) - observed
    theta = 0.5

    R = remnant.RemnantNetwork(
        graph=G,
        observed=observed,
        unobserved=unobserved,
        theta=theta
    )

    return R

@pytest.fixture
def simple_mplx():
    # A complete four node aggregate
    # A path graph in one layer
    # An "x" in the other
    perimeter = nx.Graph()
    perimeter.add_nodes_from(range(4))
    perimeter.add_edges_from([
        (0,1),
        (1,2),
        (2,3),
        (3,0)
    ])

    interior = nx.Graph()
    interior.add_nodes_from(range(4))
    interior.add_edges_from([
        (0,2),
        (1,3),
    ])

    R1 = remnant.RemnantNetwork(perimeter, {(0,1),(1,2)}, {(2,3),(3,0)}, 0.5)
    R2 = remnant.RemnantNetwork(interior, {(0,2)}, {(1,3)}, 0.5)

    M = remnant.RemnantMultiplex([R1, R2], [1, 2])

    return M


# ========== Test suite ==========
# --- RemnantNetwork object ---
class TestRemnantNetwork:
    # Attributes
    def test_edges(self, simple_remnant):
        expected = set(simple_remnant.graph.edges())  # expectation of train \cup test

        obs = simple_remnant.observed
        unobs = simple_remnant.unobserved
        actual = obs | unobs  # evaluating train \cup test

        assert actual == expected

    # Methods
    def test_save(self):
        pass

    def test_safe_save(self):
        pass

    # Dataclass boilerplate
    def test_frozen(self, simple_remnant):
        with pytest.raises(FrozenInstanceError):
            simple_remnant.theta = 0.0

    def test_default_factories(self, simple_remnant):
        assert simple_remnant.metadata == dict()


# --- RemnantMultiplex object ---
class TestRemnantMultiplex:
    # Attributes
    def test_labels_to_layer(self, simple_mplx):
        expected = {1: simple_mplx.layers[0], 2: simple_mplx.layers[1]}
        actual = simple_mplx.labels_to_layers
        assert actual == expected

    # Methods
    def test_save(self):
        pass

    def test_safe_save(self):
        pass

    # Dataclass boilerplate
    def test_default_factories(self, simple_mplx):
        assert simple_mplx.metadata == dict()
