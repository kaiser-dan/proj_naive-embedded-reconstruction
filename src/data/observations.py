"""Project source code for caching commonly used multiplex observational data.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import pickle
from dataclasses import dataclass

# --- Scientific computing ---
import numpy as np

# --- Network science ---
import networkx as nx

# --- Project source ---
# PATH adjustments
ROOT = "../../"
sys.path.append(f"{ROOT}/src/")

from sampling.random import partial_information
from embed.N2V import N2V

# ========== CLASSES ==========
@dataclass
class PreprocessedData:
    """Class for storing relevant multiplex observational data for reconstruction experiments."""
    system: str
    layers: tuple[int, int]
    theta: float
    remnants: tuple[nx.Graph, nx.Graph]
    embeddings: tuple[dict, dict]
    observed_edges: dict[tuple[int, int], int]
    unobserved_edges: dict[tuple[int, int], int]



# =================== FUNCTIONS ===================
def get_preprocessed_data(
        system, layers,
        theta, repetition,
        ROOT="../../data/input/preprocessed/"):
    filename = \
        f"{ROOT}/cache_system={system}_layers={layers[0]}-{layers[1]}_theta={theta:.2f}_rep={repetition}.pkl"

    with open(filename, "rb") as _fh:
        preprocessed_data = pickle.load(_fh)

    return preprocessed_data

def calculate_preprocessed_data(
        G, H,
        system, layers,
        theta, repetition,
        embedding_parameters, embedding_hyperparameters,
        ROOT="../../data/input/preprocessed/"):
    filename = \
        f"{ROOT}/cache_system={system}_layers={layers[0]}-{layers[1]}_theta={theta:.2f}_rep={repetition}.pkl"

    # Calculate remnants
    R_G, R_H, unobserved_edges, observed_edges = partial_information(G, H, theta)

    # Embed remnants
    E_G = N2V(R_G, embedding_parameters, embedding_hyperparameters)
    E_H = N2V(R_H, embedding_parameters, embedding_hyperparameters)

    # Format class
    preprocessed_data = PreprocessedData(
        system, layers, theta,
        (R_G, R_H), (E_G, E_H),
        observed_edges, unobserved_edges
    )

    # Save to disk
    with open(filename, "wb") as _fh:
        pickle.dump(preprocessed_data, _fh, pickle.HIGHEST_PROTOCOL)
