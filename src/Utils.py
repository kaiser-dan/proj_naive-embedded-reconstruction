import networkx as nx
import random
from sklearn import metrics
from cdlib import algorithms
from cdlib import evaluation
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
import os
from tqdm import tqdm as tqdm
import pickle


def generate_power_law (gamma, kmin, kmax):
    xmin = np.power(kmin, 1.0 - gamma)
    xmax = np.power(kmax, 1.0 - gamma)
    x = xmax - random.random()*(xmax - xmin)
    x = np.power(x, 1.0 / (1.0 -gamma))
    return int(x)

######

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

#####

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

################

def generate_multiplex_configuration (N, gamma, kmin, kmax, prob, sign):
#     print ('# N = ', N)
#     print ('# gamma = ', gamma)
#     print ('# kmin = ', kmin)
#     print ('# kmax = ', kmax)
#     print ('# prob = ', prob)
#     print ('# sign = ' , sign)
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


#########

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


#     print ('# option = ', option)

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