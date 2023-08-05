"""Project source code for calculating likelihoods from vector distances.
"""
# ============= SET-UP =================
# --- Project source code ---
from distance import _likelihoods

# ============= FUNCTIONS =================
def edge_likelihood(
        target_value, *other_values,
        normalize=True,
        likelihood_model = _likelihoods.inverse):
    # Calculate likelihood for specified layer (given distance)
    likelihood_of_target = likelihood_model(target_value)

    # Normalize likelihood by likelihood over all layers
    normalization_term = 0
    if normalize:
        normalization_term += likelihood_of_target
        for value in other_values:
            normalization_term += likelihood_model(value)
    else:
        normalization_term += 1

    return likelihood_of_target / normalization_term

# --- Helpers ---
def scale_probability(p):
    return 2*p - 1