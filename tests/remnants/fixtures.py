import pytest
import networkx as nx

from embmplxrec.remnants import observer
# ========== Fixtures ==========
# --- Test Fixtures ---
@pytest.fixture
def simple_remnant():
    G = nx.path_graph(10)
    tmp = observer.partial_information([G], 0.5)

    R = tmp.layers[0]

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

    M = observer.partial_information([perimeter, interior], 0.5)

    return M

@pytest.fixture
def formed_mplx():
    alpha = nx.Graph()
    alpha.add_nodes_from(range(20))
    alpha.add_edges_from([(x, x+1) for x in range(1, 9)])

    beta = nx.Graph()
    beta.add_nodes_from(range(20))
    beta.add_edges_from([(x, x+1) for x in range(11, 19)])

    M = observer.partial_information([alpha, beta], 0.5)

    return M
