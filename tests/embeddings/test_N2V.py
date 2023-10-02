import pytest

import networkx as nx

from EMB.embeddings import N2V

# =========== TESTS ===========
class TestN2VHelpers:
    def test_check_kwargs(self):
        expected = {
            "dimensions": None,
        }
        actual = N2V._check_kwargs(dimensions=None)

        assert actual == expected

    def test_check_kwargs_nopanic(self):
        expected = {
            "dimensions": None,
        }
        actual = N2V._check_kwargs(dimensions=None, stuff=1)

        assert actual == expected

    def test_check_kwargs_panic(self):
        with pytest.raises(ValueError):
            N2V._check_kwargs(_panic=True, dimensions=None, stuff=1)


class TestN2V:
    A = nx.erdos_renyi_graph(50, 0.005)
    B = nx.erdos_renyi_graph(50, 0.005)
    D = {1: A, 2: B}
    def test_N2V_core(self):
        vectors = N2V.embed_N2V(self.A)

        expected = [128]*50  # dimension of return vectors

        actual = [len(vector) for vector in vectors.values()]

        assert actual == expected

    def test_N2V_multiplex_core(self):
        vectors = N2V.embed_multiplex_N2V(self.D)

        expected = [128]*50  # dimension of return vectors

        actual1 = [len(vector) for vector in vectors[1].values()]
        actual2 = [len(vector) for vector in vectors[2].values()]

        assert (actual1 == expected) and (actual2 == expected)
