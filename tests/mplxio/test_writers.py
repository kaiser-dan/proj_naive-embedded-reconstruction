from os.path import join as jn
from os.path import isfile

import networkx as nx

from emb.mplxio import writers
# =========== TESTS ===========
class TestWriters:
    G = nx.path_graph(10)
    H = nx.path_graph(20)
    duplex = {1: G, 37: H}
    fp = jn('tests', 'mplxio', 'output.mplx')
    def test_to_edgelist_core(self):
        writers.to_edgelist(self.duplex, self.fp)

        assert isfile(self.fp)

    def test_to_edgelist_layernames(self):
        writers.to_edgelist(self.duplex, self.fp)

        layers = set()
        fh = open(self.fp)
        for line in fh:
            layers.add(int(line.split()[0]))
        fh.close()

        assert layers == {1, 37}

    def test_to_edgelist_edges(self):
        writers.to_edgelist(self.duplex, self.fp)

        expected = {
            (1, i, i+1) for i in range(10-1)
        }
        expected.update({
            (37, i, i+1) for i in range(20-1)
        })

        actual = set()
        fh = open(self.fp)
        for line in fh:
            actual.add(
                (int(line.split()[0]),
                 int(line.split()[1]),
                 int(line.split()[2])))

        assert actual == expected