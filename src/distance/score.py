"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Scientific ---
import numpy as np


# ============= FUNCTIONS =================
# --- Likelihood models ---
# Basic convex models
def inverse_(x): return 1/x
def negexp_(x): return np.exp(-x)


# Sigmoid models
def logistic_(x): return 1 / (1 + negexp_(x))
def tanh_(x): return np.tanh(x)
def arctan_(x): return np.arctan(x)


# --- Drivers ---
def likelihood(
    target_distance, *other_distances,
        likelihood_model: function = inverse_):
    # Calculate likelihood for specified layer (given distance)
    likelihood_of_target = likelihood_model(target_distance)

    # Normalize likelihood by likelihood over all layers
    likelihoods_of_others = [
        likelihood_model(dist) for dist in other_distances
    ]
    normalization_term = likelihood_of_target + sum(likelihoods_of_others)

    return likelihood_of_target / normalization_term
