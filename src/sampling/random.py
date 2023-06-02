"""Project source code for random observed sub-multiplex simulation.
"""
# ============= SET-UP =================
# --- Standard library ---
import random

# --- Scientific computing ---

# --- Network science ---

# --- Project code ---
from sampling.remnants import build_remnants


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
    rem_G1, rem_G2 = build_remnants(G1, G2, Etrain, Etest, theta=frac)

    return rem_G1, rem_G2, Etest, Etrain
