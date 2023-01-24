"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Scientific ---
import numpy as np


# ============= FUNCTIONS =================
inverse = lambda x: 1/x
negexp = lambda x: np.exp(-x)
logistic = lambda x: 1 / (1 + negexp(x))
tanh = lambda x: np.tanh(x)
arctan = lambda x: np.arctan(x)

def likelihood(target, *other_distances, metric:function = inverse):
    likelihood_of_target = metric(target)
    normalization_term = likelihood_of_target + sum([metric(dist) for dist in other_distances])

    return likelihood_of_target / normalization_term