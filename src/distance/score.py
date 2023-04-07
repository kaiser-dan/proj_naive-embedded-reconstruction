"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Scientific computing ---
import numpy as np

# --- Network science ---
from networkx import node_connected_component as component

# --- Project source code ---
from distance.distance import *

# ============= FUNCTIONS =================
# --- Likelihood models ---
# Basic convex models
def identity_(x): return x
def inverse_(x): return 1/x
def negexp_(x): return np.exp(-x)


# Sigmoid models
def logistic_(x): return 1 / (1 + negexp_(x))
def tanh_(x): return np.tanh(x)
def arctan_(x): return np.arctan(x)


# --- Drivers ---
# Probabilities
def likelihood(
    target_distance, *other_distances,
        likelihood_model = inverse_):
    # Calculate likelihood for specified layer (given distance)
    likelihood_of_target = likelihood_model(target_distance)

    # Normalize likelihood by likelihood over all layers
    likelihoods_of_others = [
        likelihood_model(dist) for dist in other_distances
    ]
    normalization_term = likelihood_of_target + sum(likelihoods_of_others)

    return likelihood_of_target / normalization_term

# --- Helpers ---
def scale_probability(p):
    return 2*p - 1

def format_distance_ratios(X):
    # Apply logarithmic transform to regularize division space
    X = np.log(X)

    # Remove NaNs for sklearn model
    X = np.nan_to_num(X, nan=-1e-32)

    # Shape features for sklearn model
    X = X.reshape(-1, 1)

    return X