import networkx as nx

from emb.netsci.models import preprocessing
# =========== TESTS ===========
class TestNetsciPreprocessingUnit:
    def test_trim_inactive_nodes_empty(self):
        A = nx.Graph()
        A.add_edges_from({(0,1)})

        preprocessing.trim_inactive_nodes(A)

        expected = [0,1]
        actual = list(A.nodes())

        assert actual == expected

    def test_trim_inactive_nodes_full(self):
        A = nx.Graph()
        A.add_nodes_from({0,1,2,3,4})

        preprocessing.trim_inactive_nodes(A)

        expected = []
        actual = list(A.nodes())

        assert actual == expected

    def test_trim_inactive_nodes_partial(self):
        A = nx.Graph()
        A.add_nodes_from({0,1,2,3,4})
        A.add_edges_from({(0,1), (1,2), (2,4)})

        preprocessing.trim_inactive_nodes(A)

        expected = [0,1,2,4]
        actual = list(A.nodes())

        assert actual == expected

    def test_get_common_nodes_empty(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_nodes_from({0,1,2,3,4})
        B.add_nodes_from({0,1,2,3,4})

        expected = {0,1,2,3,4}
        actual = preprocessing.get_all_nodes(A, B)

        assert actual == expected

    def test_get_common_nodes_full(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_nodes_from({0,1,2})
        B.add_nodes_from({3,4})

        expected = {0,1,2,3,4}
        actual = preprocessing.get_all_nodes(A, B)

        assert actual == expected

    def test_get_common_nodes_partial(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_nodes_from({0,1,2})
        B.add_nodes_from({2,3,4})

        expected = {0,1,2,3,4}
        actual = preprocessing.get_all_nodes(A, B)

        assert actual == expected

    def test_get_common_edges_empty(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })
        B.add_edges_from({
            (1,5),
            (1,9),
            (37,3)
        })

        expected = set()
        actual = preprocessing.get_shared_edges(A, B)

        assert actual == expected

    def test_get_common_edges_full(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })
        B.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })

        expected = {(0,1), (1,2), (2,3)}
        actual = preprocessing.get_shared_edges(A, B)

        assert actual == expected

    def test_get_common_edges_partial(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })
        B.add_edges_from({
            (1,5),
            (1,2),
            (37,3)
        })

        expected = {(1,2)}
        actual = preprocessing.get_shared_edges(A, B)

        assert actual == expected

class TestNetsciPreprocessingIntegrated:
    def test_make_layers_disjoint_empty(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })
        B.add_edges_from({
            (1,5),
            (1,9),
            (37,3)
        })

        expected = [
            {(0,1),(1,2),(2,3)},
            {(1,5),(1,9),(37,3)}
        ]
        actual = [set(g.edges()) for g in preprocessing.make_layers_disjoint(A, B)]

        assert actual == expected

    def test_make_layers_disjoint_full(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })
        B.add_edges_from({
            (0,1),
            (1,2),
            (2,3)
        })

        expected = [
            set(),
            set()
        ]
        actual = [set(g.edges()) for g in preprocessing.make_layers_disjoint(A, B)]

        assert actual == expected

    def test_make_layers_disjoint_partial(self):
        A = nx.Graph()
        B = nx.Graph()
        A.add_edges_from({
            (0,1),
            (1,2),
            (37,3)
        })
        B.add_edges_from({
            (1,5),
            (1,2),
            (37,3)
        })

        expected = [
            {(0,1)},
            {(1,5)}
        ]
        actual = [set(g.edges()) for g in preprocessing.make_layers_disjoint(A, B)]

        assert actual == expected