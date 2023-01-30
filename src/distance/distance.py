"""Project source code for calculating distances between vectors.
"""
# ============= SET-UP =================
# --- Scientific ---
import numpy as np


# ============= FUNCTIONS =================
# --- Drivers ---
def embedded_edge_distance(edge, X, Y, distance_=euclidean_distance):
    src, tgt = edge

    x = X[src] - X[tgt]
    y = Y[src] - Y[tgt]

    return distance_(x, y)

# --- Computations ---
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def cosine_similarity(x, y):
    dot_ = np.dot(x, y)

    # Applying Cauchy-Schwartz inequality
    cosine_ = dot_ / (np.norm(x) * np.norm(y))

    return np.arccos(cosine_)

def poincare_disk_distance(x, y):
    raise NotImplementedError("Hyperbolic distance not yet implemented!")