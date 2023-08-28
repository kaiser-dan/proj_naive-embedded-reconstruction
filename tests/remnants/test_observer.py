import pytest

import networkx as nx

from EMB.remnants import observer

# --- Fixtures ---
from fixtures import simple_duplex

# =========== TESTS ===========
class TestObservationHelpers:
    G = nx.path_graph(10)
    A = nx.path_graph(101)
    B = {(node, node+1) for node in range(10)}
    def test_adjust_theta(self):
        theta = 0.6
        expected = 0.5
        actual = observer._adjust_theta(self.A, theta, self.B)

        assert actual == expected

    def test_get_sample_space(self):
        previous = {(0,1), (8,9)}

        expected = {(node, node+1) for node in range(1,8)}
        actual = observer._get_sample_space(self.G, previous)

        assert actual == expected

    def test_get_sample_space_nonoverlappingprev(self):
        previous = {(37,40)}

        expected = {(node, node+1) for node in range(0,9)}
        actual = observer._get_sample_space(self.G, previous)

        assert actual == expected

    def test_get_sample_space_emptyprev(self):
        previous = set()

        expected = {(node, node+1) for node in range(9)}
        actual = observer._get_sample_space(self.G, previous)

        assert actual == expected

    def test_get_all_edges(self):
        expected = {(node, node+1) for node in range(100)}
        actual = observer._get_all_edges({1:self.G,2:self.A})
        assert actual == expected

class TestObserver:
    G = nx.path_graph(11)

    @pytest.mark.parametrize('theta', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    def test_basic(self, theta):
        expected = int(10*theta)
        actual = len(observer.random_observations(self.G, theta))

        assert actual == expected

    def test_empty(self):
        expected = 0
        actual = len(observer.random_observations(self.G, 0.0))

        assert actual == expected

    def test_full(self):
        expected = 10
        actual = len(observer.random_observations(self.G, 1.0))

        assert actual == expected

class TestCumulativeObserver:
    G = nx.path_graph(11)

    def test_basic_cumulative(self):
        observations = observer.random_observations(self.G, 0.6, {(0,1)})

        expected = 6
        actual = len(observations)

        assert actual == expected

    def test_monotonicity(self):
        observations = observer.random_observations(self.G, 0.6, {(0,1)})

        assert {(0,1)}.issubset(observations)

class TestObserverMultiplex:
    def test_random_observations_multiplex_core(self, simple_duplex):
        obs_mplx = observer.random_observations_multiplex(simple_duplex, 0.5)
        assert isinstance(obs_mplx, dict)

    def test_random_observations_multiplex_subsetinclusion(self, simple_duplex):
        obs_mplx = observer.random_observations_multiplex(simple_duplex, 0.5)

        l1_subset = all([edge in simple_duplex[1].edges() for edge in obs_mplx[1]])
        l2_subset = all([edge in simple_duplex[2].edges() for edge in obs_mplx[2]])
        assert l1_subset and l2_subset

    def test_build_remnant_multiplex(self, simple_duplex):
        obs_mplx = observer.random_observations_multiplex(simple_duplex, 0.5)

        rmplx = observer.build_remnant_multiplex(simple_duplex, obs_mplx)

        assert rmplx.get(-1, None) is not None

    def test_build_remnant_multiplex_sharedagg(self, simple_duplex):
        obs_mplx = observer.random_observations_multiplex(simple_duplex, 0.5)

        rmplx = observer.build_remnant_multiplex(simple_duplex, obs_mplx)

        agg = rmplx[-1]
        agg_edges = set(agg.edges())

        l1_inclusion = agg_edges.issubset(rmplx[1].edges())
        l2_inclusion = agg_edges.issubset(rmplx[2].edges())

        assert l1_inclusion and l2_inclusion

class TestCumulativeObserverMultiplex:
    def test_random_observations_multiplex_monotonic(self, simple_duplex):
        obs_mplx = observer.random_observations_multiplex(simple_duplex, 0.5)
        obs_mplx2 = observer.random_observations_multiplex(simple_duplex, 0.7, obs_mplx)

        l1_monotonic = obs_mplx[1].issubset(obs_mplx2[1])
        l2_monotonic = obs_mplx[2].issubset(obs_mplx2[2])

        assert l1_monotonic and l2_monotonic

    def test_cumulative_remnant_multiplxes_core(self, simple_duplex):
        rmplxs = observer.cumulative_remnant_multiplexes(simple_duplex, [0.1, 0.5, 0.9])

        assert set(rmplxs.keys()) == {0.1, 0.5, 0.9}