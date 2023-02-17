"""Project source code for power law distribution sampling methods.
"""
# ============= SET-UP =================
# --- Scientific computation ---
import numpy as np


# ============= FUNCTIONS =================
def generate_power_law(gamma, kmin, kmax):
    xmin = np.power(kmin, 1.0 - gamma)
    xmax = np.power(kmax, 1.0 - gamma)
    x = xmax - np.random.random()*(xmax - xmin)
    x = np.power(x, 1.0 / (1.0 - gamma))
    return int(x)