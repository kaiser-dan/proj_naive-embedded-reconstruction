"""Project source code for synthetic benchmark multiplex generation.
"""
# ============= SET-UP =================
# --- Standard library ---
import os  # for calling shell scripts
import sys
import random  # for random shufflings
import subprocess

# --- Scientific computing ---
import numpy as np  # General computational tools

# --- Network science ---
import networkx as nx  # General network tools

# --- Project source code ---
# PATH adjustments and binaries
PATH_LFR = os.path.join(*["..", "..", "bin", "LFR", ""])
sys.path.append(PATH_LFR)


# =================== FUNCTIONS ===================
# --- Helpers ---
def generate_power_law(gamma, kmin, kmax):
    xmin = np.power(kmin, 1.0 - gamma)
    xmax = np.power(kmax, 1.0 - gamma)
    x = xmax - np.random.random()*(xmax - xmin)
    x = np.power(x, 1.0 / (1.0 - gamma))
    return int(x)

def control_correlation(degree, prob):
    tmp_degree = []
    for i in range(len(degree)):
        tmp_degree.append(degree[i])

    for i in range(len(tmp_degree)):
        if random.random() < prob:
            n = tmp_degree[i]
            j = random.randint(0, len(degree)-1)
            tmp_degree[i] = tmp_degree[j]
            tmp_degree[j] = n

    return tmp_degree


# --- Configuration model ---
def generate_multiplex_configuration (N, gamma, kmin, kmax, prob, sign):
    """Generate a two-layer multiplex with configuration model layers.

    Parameters
    ----------
    N : int
        Number of nodes
    gamma : float
        Degree sequence exponent. Positive as a parameter.
    kmin : float
        Minimum degree.
    kmax : float
        Maximum degree.
    prob : float
        Proportion of edges to shuffle from one layer to the second.
    sign : int
        Sort of degree sequence when creating second layer. Must be -1 or 1. -1 corresponds to anti-correlated, 1 to correlated.

    Returns
    -------
    dict
        Mapping of layer ids to layers. Default keys 1 and 2.
    """

    # >>> Book-keeping >>>
    degree = []  # layer one degree sequence
    tmp_degree = []  # layer two degree sequence
    G = {}  # dictionary of layers
    # <<< Book-keeping <<<

    # >>> Multiplex construction >>>
    # Sample layer one degree sequence
    for i in range(0, N):
        degree.append(generate_power_law (gamma, kmin, kmax))
    degree = sorted(degree)

    # Construct layer two degree sequence with specified (anti)correlation
    for i in range (0, len(degree)):
        if sign > 0:
            tmp_degree.append(degree[i])
        if sign < 0:
            tmp_degree.append(degree[len(degree)-i-1])
    tmp_degree = control_correlation (tmp_degree, prob)

    # Build layers from degree sequences
    G[1] = generate_configuration_model (degree)
    G[2] = generate_configuration_model (tmp_degree)
    # <<< Multiplex construction <<<

    return G

def generate_configuration_model (degree):
    """Generate a single instance of the configuration model with the specified degree sequence.

    Note that this function exclusively generates undirected and unweighted networks.

    Parameters
    ----------
    degree : list
        Desired degree sequence.

    Returns
    -------
    nx.Graph
        Resultant configuration model sample.
    """

    # >>> Book-keeping >>>
    G = nx.Graph()  # instantiate correctly sized graph
    for n in range(0, len(degree)):
        G.add_node(n)
    edges = []  # edges to add
    # <<< Book-keeping <<<

    # >>> Constructing graph >>>
    # Initialize edges
    for i in range(0, len(degree)):
        k = degree[i]
        for j in range(0, k):
            edges.append(i)

    # Identifying edges with configuration model
    for i in range(0, len(edges)):
        j = random.randint(0, len(edges)-1)
        tmp = edges[i]
        edges[i] = edges[j]
        edges[j] = tmp

    # Adding edges to graph
    for i in range(0, len(edges)-1, 2):
        n = edges[i]
        m = edges[i+1]
        if G.has_edge(n, m) == False and n != m:
            G.add_edge(n, m)
    # <<< Constructing graph <<<

    return G


# --- LFR model ---
# * NOTE: I swapped out `os.system` for the safer, more controllable `subprocess.check_call`
# * This also allows me to suppress that super annoying LFR generation logging to stdout
def generate_LFR_model(n,t1,t2,mu,avg_k,max_k, ROOT):
    # Call LFR generation, wait for completion, propogate bash exit codes
    subprocess.call(
        " ".join([
            f"{ROOT}/bin/LFR/benchmark.sh",
            "-N", f"{n}",
            "-k", f"{avg_k}",
            "-maxk", f"{max_k}",
            "-t1", f"{t1}",
            "-t2", f"{t2}",
            "-mu", f"{mu}"
        ]),
        # stdout=open(os.devnull, 'w'),
        # stderr=open(os.devnull, 'w'),
        shell=True
    )

    # Format resultant network as networkx Graph
    x=np.loadtxt("./network.dat")
    edges=[(int(x[i][0])-1,int(x[i][1])-1) for i in range(len(x))]
    g=nx.Graph(edges)

    # Format resultant node partition as dict
    x=np.loadtxt("./community.dat")
    coms={int(x[i][0])-1:int(x[i][1]) for i in range(len(x))}

    return g, coms

def generate_multiplex_LFR(N, tau1, tau2, mu, average_degree, max_degree, min_community, prob_relabel, ROOT="../../"):

    # >>> Book-keeping >>>
    groups = {}
    # <<< Book-keeping <<<

    # >>> Experimental data sampling >>>
    # Generate LFR network (one layer)
    H, comm = generate_LFR_model(n=N, t1=tau1, t2=tau2, mu=mu, avg_k=average_degree, max_k = max_degree, ROOT=ROOT)#, min_community=min_community)

    # Create list of communities
    for n in comm:
        c = comm[n]
        if c not in groups:
            groups[c] = []
        groups[c].append(n)


    # Create node -> community mapping
    sigma1 = {}
    for n in comm:
        sigma1[n] = comm[n]


    # Shuffle community labels
    new_labels = {}  # old_community_label -> new_community_label
    for C in groups:
        tmp = groups[C].copy()
        random.shuffle(tmp)
        for i in range(0, len(groups[C])):
            n = groups[C][i]
            m = tmp[i]
            new_labels[n] = m

    # Apply community label shuffling to node -> community mapping
    tmp_sigma2 = {}
    for n in sigma1:
        m = new_labels[n]
        tmp_sigma2[m] = sigma1[n]

    # Form duplex of LFR layers
    G = {}
    G[1] = H.copy()
    G[2] = nx.Graph()

    ## Ensure same node set in both layers
    for n in G[1]:
        G[2].add_node(n)

    ## Add edges to second layer _with label shuffling_
    for e in G[1].edges():
        n = new_labels[e[0]]
        m = new_labels[e[1]]
        G[2].add_edge(n, m)

    # Beak community correlation between layers
    list_nodes = list(G[2].nodes())
    new_labels = {}
    H = G[2].copy()
    for n in G[2]:
        new_labels[n] = n
    for n in new_labels:
        if random.random()<prob_relabel:
            m = random.choice(list_nodes)
            tmp = new_labels[n]
            new_labels[n] = new_labels[m]
            new_labels[m] = tmp

    G[2] = nx.Graph()
    for n in H:
        m = new_labels[n]
        G[2].add_node(m)
    for e in H.edges():
        n = new_labels[e[0]]
        m = new_labels[e[1]]
        G[2].add_edge(n, m)

    sigma2 = {}
    for n in tmp_sigma2:
        m = new_labels[n]
        sigma2[m] = tmp_sigma2[n]
    # <<< Experimental data sampling <<<

    return G, sigma1, sigma2, mu

def generate_multiplex_overlapping(num_nodes=100, proportion_overlap=0.5):
    """Generate a synthetic duplex with tunable number of overlapping active nodes.

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes in each layer (equivalent), by default 100
    proportion_overlap : float, optional
        Proportion of nodes to be active in both layers, by default 0.5

    Returns
    -------
    list
        Resultant layers of the synthetic duplex.
    """
    #create layer 1
    g1=nx.barabasi_albert_graph(num_nodes,1)
    g1.add_edges_from([(random.choice(list(g1.nodes())),random.choice(list(g1.nodes()))) for _ in range(10)])
    g1.remove_edges_from(nx.selfloop_edges(g1))

    #create layer 2
    g2=nx.barabasi_albert_graph(num_nodes,1)
    g2.add_edges_from([(random.choice(list(g2.nodes())),random.choice(list(g2.nodes()))) for _ in range(10)])
    g2.remove_edges_from(nx.selfloop_edges(g2))

    #relabel nodes
    shuffle=random.sample(g2.nodes(),num_nodes)
    mapping={i:shuffle[i] for i in g2.nodes()}
    g2=nx.relabel_nodes(g2,mapping)

    g1.add_nodes_from(list(range(num_nodes,2*num_nodes)))
    g2.add_nodes_from(list(range(num_nodes,2*num_nodes)))

    #overlapping nodes
    move=int((1-proportion_overlap)*num_nodes)
    mapping={i:(i+move)%(2*num_nodes) for i in g2.nodes()}
    g2=nx.relabel_nodes(g2,mapping)

    return [g1,g2]
