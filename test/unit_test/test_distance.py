import sys
import pytest
import numpy as np
import networkx as nx

sys.path.append('../')
from distance import distance
from distance import score
from embed import N2V
from embed import LE
from utils import parameters

seed = 273
np.random.seed(seed = seed) # random seed for consistent results

# ============= Test objects =================
# --- Vectors ---
vector_a = np.array([1, 3])
vector_b = np.array([2, 4])

# --- Networks ---
n_node = 100
avg_k = 3
n_edge = int(n_node * avg_k / 2)
G = nx.gnm_random_graph(n = n_node, m = n_edge, seed = seed)

# --- Components, edges, etc ---
n_component = nx.number_connected_components(G)
components = [list(c) for c in sorted(nx.connected_components(G), key = len, reverse = True)]

node1 = np.random.choice(components[0])
node2 = np.random.choice(components[1])

edge_list = list(nx.edges(G))
test_edge_A = edge_list[np.random.randint(len(edge_list))] # existing edge
test_edge_B = (node1, node2) # non-existing edge

print('\n')
print('basic network info:')
print(f'n_node = {n_node}')
print(f'n_edge = {n_edge}')
print(f'n_components = {n_component}')
for i in range(n_component):
    print(f'component {i+1}: {components[i]}')

print(f'test_edge_A = {test_edge_A}: edge inside the gcc')
print(f'test_edge_B = {test_edge_B}: virtual edge between two different components')

# --- Representations ---
N2V_params, N2V_hparams = parameters.set_parameters_N2V()
LE_params, LE_hparams = parameters.set_parameters_LE()


representation = N2V.N2V(G, N2V_params, N2V_hparams) 
# representation = LE.LE(G, LE_params, LE_hparams)

# --- Tests ---
def test_euclidean_distance():
    va = vector_a
    vb = vector_b
    dist = distance.euclidean_distance(va, vb)

    assert round(dist, 2) == round(np.sqrt(2), 2)

def test_cosine_similarity():
    va = vector_a
    vb = vector_a
    cosim = distance.cosine_similarity(va, vb)
    
    assert round(cosim, 2) == round(1.0, 2)

def test_component_penalized_embedded_edge_distance1():
    '''
    penalty: distance penalty between different components

    dist1: distance between two nodes which have an edge
    between them

    dist2: distance between two nodes belonging to
    two different components respectively
    '''
    penalty = 2**8
    dist1 = distance.component_penalized_embedded_edge_distance(test_edge_A, 
                                                                G, 
                                                                representation, 
                                                                penalty,)

    dist2 = distance.component_penalized_embedded_edge_distance(test_edge_B, 
                                                                G, 
                                                                representation, 
                                                                penalty,)
    assert dist1 < penalty  

def test_component_penalized_embedded_edge_distance2():
    '''
    penalty: distance penalty between different components

    dist1: distance between two nodes which have an edge
    between them

    dist2: distance between two nodes belonging to
    two different components respectively
    '''
    penalty = 2**8
    dist1 = distance.component_penalized_embedded_edge_distance(test_edge_A, 
                                                                G, 
                                                                representation, 
                                                                penalty,)

    dist2 = distance.component_penalized_embedded_edge_distance(test_edge_B, 
                                                                G, 
                                                                representation, 
                                                                penalty,)
    assert penalty < dist2

def test_component_penalized_embedded_edge_distance3():
    '''
    penalty: distance penalty between different components

    dist1: distance between two nodes which have an edge
    between them

    dist2: distance between two nodes belonging to
    two different components respectively
    '''
    penalty = 2**8
    dist1 = distance.component_penalized_embedded_edge_distance(test_edge_A, 
                                                                G, 
                                                                representation, 
                                                                penalty,)

    dist2 = distance.component_penalized_embedded_edge_distance(test_edge_B, 
                                                                G, 
                                                                representation, 
                                                                penalty,)
    assert dist1 < dist2




