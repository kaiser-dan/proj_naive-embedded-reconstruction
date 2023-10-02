import pytest

import networkx as nx

@pytest.fixture
def simple_duplex():
    G = nx.erdos_renyi_graph(100, 0.01)
    H = nx.erdos_renyi_graph(100, 0.01)
    return {1: G, 2: H}