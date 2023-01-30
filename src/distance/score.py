"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Scientific ---
import numpy as np


# ============= FUNCTIONS =================
# --- Likelihood models ---
# Basic convex models
inverse_ = lambda x: 1/x
negexp_ = lambda x: np.exp(-x)

# Sigmoid models
logistic_ = lambda x: 1 / (1 + negexp_(x))
tanh_ = lambda x: np.tanh(x)
arctan_ = lambda x: np.arctan(x)

# --- Drivers ---
def likelihood(target_distance, *other_distances, likelihood_model:function = inverse_):
    likelihood_of_target = likelihood_model(target_distance)
    likelihoods_of_others = [likelihood_model(dist) for dist in other_distances]
    normalization_term = likelihood_of_target + sum(likelihoods_of_others)

    return likelihood_of_target / normalization_term