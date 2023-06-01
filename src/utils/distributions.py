"""Project source code for generating scale-free degree distributions and controlling the correlation of two of these distributions.
"""
# ============= SET-UP =================
# --- Standard library ---
import random

# --- Scientific computation ---
import numpy as np

# ============= FUNCTIONS =================
def generate_power_law(gamma, kmin, kmax):
    xmin = np.power(kmin, 1.0 - gamma)
    xmax = np.power(kmax, 1.0 - gamma)
    x = xmax - np.random.random()*(xmax - xmin)
    x = np.power(x, 1.0 / (1.0 - gamma))
    return int(x)

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