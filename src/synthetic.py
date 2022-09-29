# ============= SET-UP =================
# --- Standard library ---
import os  # for calling shell scripts
import sys
sys.path.append("../bin/LFR/")
import random  # for random shufflings
import subprocess

# --- Scientific ---
import numpy as np  # General computational tools
from sklearn import metrics, decomposition  # Measuring classifier performance

# --- Network science ---
import networkx as nx  # General network tools
from node2vec import Node2Vec as N2V  # Embedding tools

# --- Data handling and visualization ---
import pandas as pd  # Dataframe tools
from tabulate import tabulate  # Pretty printing for dataframes

# --- Miscellaneous ---
from tqdm.auto import tqdm  # Progress bar


# =================== FUNCTIONS ===================
# --- Drivers ---
# ~ Configuration model ~
def generate_multiplex_configuration (N, gamma, kmin, kmax, prob, sign):
    degree = []
    for i in range(0, N):
        degree.append(generate_power_law (gamma, kmin, kmax))
    degree = sorted(degree)
    tmp_degree = []
    for i in range (0, len(degree)):
        if sign > 0:
            tmp_degree.append(degree[i])
        if sign < 0:
            tmp_degree.append(degree[len(degree)-i-1])

    tmp_degree = control_correlation (tmp_degree, prob)

    G = {}
    G[1] = generate_configuration_model (degree)
    G[2] = generate_configuration_model (tmp_degree)

    return G

# ~ Community model ~
# * NOTE: I swapped out `os.system` for the safer, more controllable `subprocess.check_call`
# * This also allows me to suppress that super annoying LFR generation logging to stdout
def LFR(n,t1,t2,mu,avg_k,max_k):
    #function to generate LFR network as a networkx object and obtain community assignments

    # Call LFR generation, wait for completion, propogate bash exit codes
    subprocess.call(
        " ".join([
            "../../bin/LFR/benchmark.sh",
            "-N", f"{n}",
            "-k", f"{avg_k}",
            "-maxk", f"{max_k}",
            "-t1", f"{t1}",
            "-t2", f"{t2}",
            "-mu", f"{mu}"
        ]),
        stdout=open(os.devnull, 'w'),
        stderr=open(os.devnull, 'w'),
        shell=True
    )

    x=np.loadtxt('../../bin/LFR/network.dat')
    edges=[(int(x[i][0])-1,int(x[i][1])-1) for i in range(len(x))]
    g=nx.Graph(edges)

    x=np.loadtxt('../../bin/LFR/community.dat')
    coms={int(x[i][0])-1:int(x[i][1]) for i in range(len(x))}
    #nx.set_node_attributes(g,coms,name='community')

    return g, coms

# --- Experiment helpers ---
def read_file (filename):
    G = {}

    with open(filename) as file:
        for line in file:
            data = line.strip().split()
            l = int(data[0])
            n = int(data[1])
            m = int(data[2])
            if l not in G:
                G[l] = nx.Graph()
            G[l].add_edge(n ,m)

    return G

# ~ General experiment set up ~
def duplex_network (G, l1, l2):
    G1 = G[l1].copy()
    G2 = G[l2].copy()

    ##delete common edges
    list_of_common_edges = []

    for e in G[l1].edges():
        if G[l2].has_edge(e[0], e[1]):
            list_of_common_edges.append([e[0], e[1]])

    #print (len(list_of_common_edges))

    for e in list_of_common_edges:
        G1.remove_edge(e[0], e[1])
        G2.remove_edge(e[0], e[1])

    ##delete nodes with zero degree
    list_of_nodes = []
    for n in G1.nodes():
        if G1.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.remove_node(n)

    list_of_nodes = []
    for n in G2.nodes():
        if G2.degree(n)==0:
            list_of_nodes.append(n)
    for n in list_of_nodes:
        G2.remove_node(n)

    ##create union of nodes
    list_of_nodes = []
    for n in G1.nodes():
        list_of_nodes.append(n)
    for n in G2.nodes():
        list_of_nodes.append(n)
    for n in list_of_nodes:
        G1.add_node(n)
        G2.add_node(n)

    return G1, G2

def partial_information (G1, G2, frac):
    ##training/test sets
    Etest = {}
    Etrain = {}

    for e in G1.edges():
        if random.random() < frac:
            Etrain[e] = 1
        else:
            Etest[e] = 1

    for e in G2.edges():
        if random.random() < frac:
            Etrain[e] = 0
        else:
            Etest[e] = 0

    ##remnants
    rem_G1 = nx.Graph()
    rem_G2 = nx.Graph()
    for n in G1:
        rem_G1.add_node(n)
        rem_G2.add_node(n)
    for n in G2:
        rem_G1.add_node(n)
        rem_G2.add_node(n)


    # Add aggregate edges we don't know about
    for e in Etest:
        rem_G1.add_edge(e[0], e[1])
        rem_G2.add_edge(e[0], e[1])

    # Add to remnant alpha the things known to be in alpha
    # So remnant alpha is unknown + known(alpha)
    for e in Etrain:
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])


    return rem_G1, rem_G2, Etest

# ~ Individual layer generators ~
def generate_configuration_model (degree):
    edges = []
    for i in range(0, len(degree)):
        k = degree[i]
        for j in range(0, k):
            edges.append(i)

    ##random.shuffle(edges)
    for i in range(0, len(edges)):
        j = random.randint(0, len(edges)-1)
        tmp = edges[i]
        edges[i] = edges[j]
        edges[j] = tmp

    G = nx.Graph()
    for n in range(0, len(degree)):
        G.add_node(n)

    for i in range(0, len(edges)-1, 2):
        n = edges[i]
        m = edges[i+1]
        if G.has_edge(n, m) == False and n != m:
            G.add_edge(n, m)

    return G

def lfr_multiplex (N, tau1, tau2, mu, average_degree, max_degree, min_community, prob_relabel):
    H, comm = LFR(n=N, t1=tau1, t2=tau2, mu=mu, avg_k=average_degree, max_k = max_degree)
    groups = {}
    for n in comm:
        c = comm[n]
        if c not in groups:
            groups[c] = []
        groups[c].append(n)

    sigma1 = {}
    for n in comm:
        sigma1[n] = comm[n]

    new_labels = {}
    for C in groups:
        tmp = groups[C].copy()
        random.shuffle(tmp)
        for i in range(0, len(groups[C])):
            n = groups[C][i]
            m = tmp[i]
            new_labels[n] = m

    tmp_sigma2 = {}
    for n in sigma1:
        m = new_labels[n]
        tmp_sigma2[m] = sigma1[n]

    G = {}
    G[1] = H.copy()
    G[2] = nx.Graph()
    for n in G[1]:
        G[2].add_node(n)
    for e in G[1].edges():
        n = new_labels[e[0]]
        m = new_labels[e[1]]
        G[2].add_edge(n, m)


    ## break community correlation
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

    return G, sigma1, sigma2, mu


# --- Computation helpers ---
def generate_power_law (gamma, kmin, kmax):
    xmin = np.power(kmin, 1.0 - gamma)
    xmax = np.power(kmax, 1.0 - gamma)
    x = xmax - random.random()*(xmax - xmin)
    x = np.power(x, 1.0 / (1.0 -gamma))
    return int(x)

def control_correlation (degree, prob):
    tmp_degree = []
    for i in range(len(degree)):
        tmp_degree.append(degree[i])

    for i in range(len(tmp_degree)):
        if random.random()<prob:
            n = tmp_degree[i]
            j = random.randint(0, len(degree)-1)
            tmp_degree[i] = tmp_degree[j]
            tmp_degree[j] = n

    return tmp_degree
