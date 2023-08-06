"""Project source code for common multiplex pre-processing utility.
"""
# ============= SET-UP =================
# No imports

# =================== FUNCTIONS ===================
def duplex_network (G, l1, l2, verbose=False):
    """Formats a duplex without inactive nodes or edge overlap.

    Parameters
    ----------
    G : dict
        Map from layer ids to corresponding graphs
    l1 : int
        Id of one of the layers in the resultant duplex
    l2 : int
        Id of the other layer in the resultant duplex
    verbose : bool, optional
        Indicator if verbose printing should be enabled, by default False

    Returns
    -------
    tuple
        Formatted layers of the resultant duplex
    """
    # >>> Book-keeping >>>
    # Create deepcopies of input networks
    G1 = G[l1].copy()
    G2 = G[l2].copy()
    # <<< Book-keeping <<<

    # >>> Duplex construction >>>
    # Remove edges common to both layers
    list_of_common_edges = []

    ## Identify common edges
    for edge in G[l1].edges():
        if G[l2].has_edge(edge[0], edge[1]):  # order safe for undirected networks
            list_of_common_edges.append([edge[0], edge[1]])

    ## Delete common edges from _both_ layers
    for e in list_of_common_edges:
        G1.remove_edge(e[0], e[1])
        G2.remove_edge(e[0], e[1])

    if verbose:
        print(f"Number of common edges removed: {len(list_of_common_edges)}")

    # Remove nodes with zero degree
    ## Identify nodes in one layer with no activity
    list_of_nodes = []
    for n in G1.nodes():
        if G1.degree(n)==0:
            list_of_nodes.append(n)

    ## Remove inactive nodes from one layer
    for n in list_of_nodes:
        G1.remove_node(n)
    if verbose:
        print(f"Number of inactive nodes removed from layer {l1}: {len(list_of_nodes)}")

    list_of_nodes = []
    ## Identify nodes in other layer with no activity
    for n in G2.nodes():
        if G2.degree(n)==0:
            list_of_nodes.append(n)
    ## Remove inactive nodes from one layer
    for n in list_of_nodes:
        G2.remove_node(n)
    if verbose:
        print(f"Number of inactive nodes removed from layer {l2}: {len(list_of_nodes)}")

    # Ensure node sets equivalent - Create union of nodes
    ## Identify nodes in either layer
    list_of_nodes = []
    for n in G1.nodes():
        list_of_nodes.append(n)
    for n in G2.nodes():
        list_of_nodes.append(n)
    if verbose:
        print(f"Size of active node set union from layers {l1} and {l2}: {len(list_of_nodes)}")
    ## Add nodes to both layers
    for n in list_of_nodes:
        G1.add_node(n)
        G2.add_node(n)
    # <<< Duplex construction <<<

    return G1, G2