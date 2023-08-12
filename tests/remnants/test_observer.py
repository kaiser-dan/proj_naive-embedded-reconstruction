# --- Logging ---
import logging

# --- Source code ---
from embmplxrec.remnants import observer

# --- Globals ---
import fixtures
simple_remnant = fixtures.simple_remnant
simple_mplx = fixtures.simple_mplx
formed_mplx = fixtures.formed_mplx

logging.basicConfig(level=logging.DEBUG)

# ========== Test suite ==========
class TestObserverSimple:
    def test_random_observation_empty(self, simple_remnant):
        actual = observer.random_observation(simple_remnant.graph, 0.0, 0).keys()
        expected = set()

        assert actual == expected

    def test_random_observation_full(self, simple_remnant):
        actual = observer.random_observation(simple_remnant.graph, 1.0, 0).keys()
        expected = set(simple_remnant.graph.edges())

        assert actual == expected

class TestObserverFormed:
    # Attributes
    def test_empty_observed_intersection(self, formed_mplx):
        expected = set()
        actual = formed_mplx.layers[0].observed.keys() & formed_mplx.layers[1].observed.keys()

        assert actual == expected

    def test_nonempty_unobserved_intersection(self, formed_mplx):
        unexpected = set()
        actual = formed_mplx.layers[0].unobserved.keys() & formed_mplx.layers[1].unobserved.keys()

        assert actual != unexpected

    def test_empty_unobserved_observed_intersection(self, formed_mplx):
        expected = set()
        test_edges = formed_mplx.layers[0].unobserved.keys() & formed_mplx.layers[1].unobserved.keys()
        train_edges = formed_mplx.layers[0].observed.keys() | formed_mplx.layers[1].observed.keys()
        actual = train_edges & test_edges

        assert actual == expected

    def test_correct_unobserved_per_layer(self, formed_mplx):
        for R in formed_mplx.layers:
            expected = set(R.graph.edges()) - R.observed.keys()
            actual = set(R.unobserved.keys())

            assert actual == expected

    def test_correct_observed(self, formed_mplx):
        expected = formed_mplx.layers[0].observed.keys() | formed_mplx.layers[1].observed.keys()
        actual = set(formed_mplx.observed.keys())

        assert actual == expected

    def test_correct_unobserved(self, formed_mplx):
        expected = formed_mplx.layers[0].unobserved.keys() & formed_mplx.layers[1].unobserved.keys()
        actual = set(formed_mplx.unobserved.keys())

        assert actual == expected
