"""Project source code for controlling implicit correlations in synthetic multiplexes.
"""
# ============= SET-UP =================
# --- Standard library ---
import random

# ============= FUNCTIONS =================
def control_correlation(degree, prob):
    tmp_degree = []
    for i in range(len(degree)):
        tmp_degree.append(degree[i])

    for i in range(len(tmp_degree)):
        if random.random() < prob:
            n = tmp_degree[i]
            j = random.randint(0, len(degree)-1)
            tmp_degree[i] = tmp_degree[j]
            tmp_degree[j] = n

    return tmp_degree
