"""Project source code for calculating distances between vectors.


"""
# ============= SET-UP =================
# --- Scientific ---
import numpy as np


# ============= FUNCTIONS =================
def embedded_edge_distance(edge, X, Y):
    src, tgt = edge

    x = X[src] - X[tgt]
    y = Y[src] - Y[tgt]

    return vector_distance(x, y)

def vector_distance(x, y):
    return np.linalg.norm(x - y)

def poincare_disk_distance(x, y):
    return