from emb.utils import functions

# --- Nodes ---
def all_nodes(*graphs):
    all_nodes = set()

    # Add each graph's nodes to set
    # ^ Note that set union automatically removes duplicates
    for graph in graphs:
        all_nodes.update(graph.nodes())

    return all_nodes


def common_nodes(*graphs):
    common_nodes = set(graphs[0].nodes())

    for graph in graphs:
        common_nodes.intersection_update(graph.nodes())

    return common_nodes


def reindex_nodes(graph):
    new_id_to_current_id = dict(enumerate(sorted(graph.nodes())))
    current_id_to_new_id = functions.inverse_map(new_id_to_current_id)
    return current_id_to_new_id


# --- Edges ---
def all_edges(*graphs):
    edges = set()
    for graph in graphs:
        edges.update(graph.edges())

    return edges


def common_edges(*graphs):
    edges = set(graphs[0].edges())
    for graph in graphs:
        edges.intersection_update(graph.edges())

    return edges


def find_edge(edge, *graphs):
    graph_indices = []
    for idx, graph in enumerate(graphs):
        if graph.has_edge(*edge):
            graph_indices.append(idx)

    return graph_indices


def find_edges(edgeset, *graphs):
    graph_indices = []
    for edge in edgeset:
        graph_indices.append(find_edge(edge, *graphs))
