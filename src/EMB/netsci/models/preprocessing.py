"""Common multiplex preprocessing utility.
"""
# ============= SET-UP =================
__all__ = ['make_layers_disjoint', 'make_nodes_contiguous']

# --- Imports ---
import networkx as nx

from EMB.netsci import utils

# =================== FUNCTIONS ===================
def trim_inactive_nodes(graph):
    # Identify inactive nodes
    inactive_nodes = [
        node
        for node in graph
        if graph.degree(node) == 0
    ]

    # Remove inactive nodes
    graph.remove_nodes_from(inactive_nodes)

def get_all_nodes(*graphs):
    common_nodes = set()

    # Add each graph's nodes to set
    # ^ Note that set union automatically removes duplicates
    for graph in graphs:
        common_nodes.update(graph.nodes())

    return common_nodes

def get_shared_edges(*graphs):
    common_edges = set().union(*[graph.edges() for graph in graphs])

    # Restrict all edges to edges shared by each graph
    # ^ Note that set intersection determines what is shared _among all graphs_
    for graph in graphs:
        common_edges.intersection_update(graph.edges())

    return common_edges

def make_layers_disjoint(*graphs):
    # Create deepcopy of each graph object
    graphs = [graph.copy() for graph in graphs]

    # Remove shared edges
    common_edges = get_shared_edges(*graphs)
    for graph in graphs:
        graph.remove_edges_from(common_edges)

    # Remove any (possibly newly formed) inactive nodes
    for graph in graphs:
        trim_inactive_nodes(graph)

    # Ensure all nodes share the same node set
    common_nodes = get_all_nodes(*graphs)
    for graph in graphs:
        graph.add_nodes_from(common_nodes)

    return graphs

def make_nodes_contiguous(graph, current_id_to_new_id = dict()):
    if current_id_to_new_id != dict():
        current_ids_to_new_ids = utils.reindex_nodes(graph)

    nodes_reindexed = current_id_to_new_id.values()
    edges_reindexed = {
        (current_ids_to_new_ids[src], current_ids_to_new_ids[tgt])
        for src, tgt in graph.edges()
    }

    graph_reindexed = nx.Graph()
    graph_reindexed.add_nodes_from(nodes_reindexed)
    graph_reindexed.add_edges_from(edges_reindexed)

    return graph_reindexed
