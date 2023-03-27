
# Stdlib
import sys
import os
import random

# Scientific computing
import numpy as np

# Network science
import networkx as nx

# Data handling and vis
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from node2vec import Node2Vec

# Project source
sys.path.append("../../src/")
from synthetic import *
from utils import *
import copy



# ============= FUNCTIONS =================
def N2V(graph, parameters, hyperparameters):
    """Embed `graph` using node2vec.
    Parameters
    ----------
    graph : nx.Graph
        Graph to embed. Node and edge attributes are ignored.
    parameters : dict
        Keyword arguments for node2vec walk generation.
    hyperparameters : dict
        Keyword arguments for word2vec fitting on node2vec-generated walks.
    Returns
    -------
    dict
        Map of node ids to embedded vectors.
    """
    # Sample random walks
    embedding_model = Node2Vec(graph, **{k: v for k, v in parameters.items() if k in ["dimensions","walk_length","num_walks","workers","quiet"]})

    # Embed walks with word2vec and retrieve model
    embedding_model = embedding_model.fit()#**hyperparameters)
    embedding_model = embedding_model.wv

    # Retrieve resultant vectors
    vectors = embedding_model.vectors

    # Retrieve word2vec internal hash of node ids to vector indices
    node_labels = embedding_model.index_to_key

    # Map node ids into corresponding vector
    # This accounts for graphs with non-consecutive node ids
    embedding = {
        int(node_label): vectors[node_index]
        for node_index, node_label in enumerate(node_labels)
    }

    return embedding



def set_parameters_N2V(
    dimensions=128,
    walk_length=30,
    num_walks=100,
    workers=8,
    quiet=True,
    window=10,
    min_count=1,
    batch_words=4,
    penalty="l2",
    theta_min=0.05,
    theta_max=0.5,
    theta_num=10,
    repeat=5
        ):
    parameters = {
        # >>> Node2Vec embedding <<<
        "dimensions": dimensions,  # euclidean dimension to embedd
        "walk_length": walk_length,  # number of nodes in each walk
        "num_walks": num_walks,  # number of walks per node
        "workers": workers,  # for cpu parallel work
        "quiet": quiet,  # verbose printing
        # >>> Simulations <<<
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,
    }

    hyperparameters = {
        # >>> Node2Vec embedding <<<
        "window": window,  # maximum distance between the current and predicted word within a sentence.
        "min_count": min_count,  # ignores all words with total frequency lower than this
        "batch_words": batch_words,  # [unsure]

        # >>> Logistic regression <<<
        "penalty": penalty,  # L2 regularization

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters, hyperparameters


'''
This function takes a graph, parameters and hyperparameters as input and returns a mapping of nodes to their embedded vectors.
The function performs the following steps:

1. Initializes two empty data structures: 
vectors_per_component and vectors
2. Retrieves each component as a graph and stores them in a list sorted by size in descending order
3. Embeds each component by themselves using the N2V function
4. Scales the results of each component by the average norm of the largest component
5. Amalgamates the results by mapping each node to its embedded vector
6. Finally, the function returns the mapping of nodes to their embedded vectors.

Thank ChatGPT for this! :D 
'''

def N2V_normalized_per_component(graph, parameters, hyperparameters):
    vectors_per_component = []  # list of vector embeddings, canonical ordering
    vectors = {}  # amalgamated mapping of nodes to their embedded vectors (by component)
    # Retrieve each component as a graph
    component_subgraphs = sorted([ graph.subgraph(component).copy() for component in nx.connected_components(graph)], key=len,reverse=True)
    # Embed each component by themselves
    for component_subgraph in component_subgraphs:
        vectors_per_component.append(N2V(component_subgraph, parameters, hyperparameters))
    #scale results
    average_norm_gcc=np.mean([np.linalg.norm(i) for i in vectors_per_component[0].values()])
    for n,component_subgraph in enumerate(vectors_per_component[1:]):
        average_norm_multiplier=average_norm_gcc/np.mean([np.linalg.norm(i) for i in component_subgraph.values()])
        for node in component_subgraph:
            component_subgraph[node]=average_norm_multiplier*(component_subgraph[node])
    # Amalgamate results
    for component_vectors in vectors_per_component:
        for node, vector in component_vectors.items():
            vectors[node] = vector
    return vectors