# --- Test utility ---
import pytest
from dataclasses import FrozenInstanceError  # exception we test capturing

# --- Globals ---
import fixtures
simple_remnant = fixtures.simple_remnant
simple_mplx = fixtures.simple_mplx

# ========== Test suite ==========
# --- RemnantNetwork object ---
class TestRemnantNetwork:
    # Attributes
    def test_edges(self, simple_remnant):
        expected = set(simple_remnant.graph.edges())  # expectation of train \cup test

        obs = simple_remnant.observed.keys()
        unobs = simple_remnant.unobserved.keys()
        actual = obs | unobs  # evaluating train \cup test

        assert actual == expected

    # Methods
    # def test_fixed_edge_sequence(self, simple_remnant):
    #     pass

    # Dataclass boilerplate
    def test_frozen(self, simple_remnant):
        with pytest.raises(FrozenInstanceError):
            simple_remnant.theta = 0.0

    # TODO: Fix test
    def test_default_factories(self, simple_remnant):
        pass
        # assert simple_remnant.metadata == dict()


# --- RemnantMultiplex object ---
class TestRemnantMultiplexSimple:
    # Attributes
    def test_labels_to_layer(self, simple_mplx):
        expected = {0: simple_mplx.layers[0], 1: simple_mplx.layers[1]}
        actual = simple_mplx.labels_to_layers
        assert actual == expected

    def test_empty_observed_intersection(self, simple_mplx):
        expected = set()
        actual = simple_mplx.layers[0].observed.keys() & simple_mplx.layers[1].observed.keys()

        assert actual == expected

    def test_nonempty_unobserved_intersection(self, simple_mplx):
        expected = set()
        actual = simple_mplx.layers[0].unobserved.keys() & simple_mplx.layers[1].unobserved.keys()

        assert actual != expected

    def test_empty_unobserved_observed_intersection(self, simple_mplx):
        expected = set()
        test_edges = simple_mplx.layers[0].unobserved.keys() & simple_mplx.layers[1].unobserved.keys()
        train_edges = simple_mplx.layers[0].observed.keys() | simple_mplx.layers[1].observed.keys()
        actual = train_edges & test_edges

        assert actual == expected

    # Dataclass boilerplate
    # TODO: Fix test
    def test_default_factories(self, simple_mplx):
        pass
        # assert simple_mplx.metadata == dict()

