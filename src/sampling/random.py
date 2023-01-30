"""Project source code for random observed sub-multiplex simulation.
"""
# ============= SET-UP =================
# --- Standard library ---
import random

# --- Scientific computing ---
import numpy as np

# --- Network science ---
import networkx as nx

# --- Project code ---
from utils.remnants import _build_remnants

# ============= FUNCTIONS =================
def partial_information(G1, G2, frac):
    # Training/test sets
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

    # Remnants
    rem_G1, rem_G2, Etest = _build_remnants(G1, G2, Etrain, Etest)

    return rem_G1, rem_G2, Etest