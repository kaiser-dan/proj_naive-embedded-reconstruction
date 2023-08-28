import networkx as nx

from EMB.netsci import utils
# =========== TESTS ===========
class TestNetsciUtilsNodes:
    def test_all_nodes(self):
        A = nx.Graph()
        A.add_nodes_from([1,2,37])
        B = nx.Graph()
        B.add_nodes_from([37,64,1001])

        expected = {1,2,37,64,1001}
        actual = utils.all_nodes(A, B)

        assert actual == expected

    def test_common_nodes(self):
        A = nx.Graph()
        A.add_nodes_from([1,2,37])
        B = nx.Graph()
        B.add_nodes_from([37,64,1001])

        expected = {37}
        actual = utils.common_nodes(A, B)

        assert actual == expected

    def test_reindex_nodes(self):
        A = nx.Graph()
        A.add_nodes_from([1,2,37])

        expected = {1:0, 2:1, 37:2}
        actual = utils.reindex_nodes(A)

        assert actual == expected

class TestNetsciUtilsEdges:
    def test_all_edges(self):
        A = nx.Graph()
        A.add_edges_from([
            (1,2), (2,3), (3,37)
        ])
        B = nx.Graph()
        B.add_edges_from([
            (3,37), (37,6), (1,4)
        ])

        expected = {
            (1,2), (2,3), (3,37), (37,6), (1,4)
        }
        actual = utils.all_edges(A, B)

        assert actual == expected

    def test_common_edges(self):
        A = nx.Graph()
        A.add_edges_from([
            (1,2), (2,3), (3,37)
        ])
        B = nx.Graph()
        B.add_edges_from([
            (3,37), (37,6), (1,4)
        ])

        expected = {(3,37)}
        actual = utils.common_edges(A, B)

        assert actual == expected

    def test_find_edge(self):
        A = nx.Graph()
        A.add_edges_from([
            (1,2), (2,3), (3,37)
        ])
        B = nx.Graph()
        B.add_edges_from([
            (3,37), (37,6), (1,4)
        ])

        expected = [0]
        actual = utils.find_edge((1,2), A, B)

        assert actual == expected

    def test_find_edge_multiple(self):
        A = nx.Graph()
        A.add_edges_from([
            (1,2), (2,3), (3,37)
        ])
        B = nx.Graph()
        B.add_edges_from([
            (3,37), (37,6), (1,4)
        ])

        expected = [0,1]
        actual = utils.find_edge((3,37), A, B)

        assert actual == expected