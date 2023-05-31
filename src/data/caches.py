"""Project source code for caching commonly used multiplex observational data.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import os
import pickle
from dataclasses import dataclass

# --- Scientific computing ---
import numpy as np

# --- Network science ---
import networkx as nx

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

from sampling.random import partial_information
from embed.N2V import N2V
from embed.LE import LE
from embed.embedding import Embedding

# ========== CLASSES ==========
@dataclass
class CachedEmbeddings:
    """Class for storing relevant multiplex observational data for reconstruction experiments."""
    system: str
    layers: tuple[int, int]
    theta: float
    remnants: tuple[nx.Graph, nx.Graph]
    embeddings: tuple[Embedding, Embedding]
    observed_edges: dict[tuple[int, int], int]
    unobserved_edges: dict[tuple[int, int], int]

# =================== FUNCTIONS ===================
# --- Precomputing and File I/O ---
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
        per_component=False,
        EMBEDDING="N2V",
        ROOT="../../data/input/preprocessed/"):
    filename = \
        f"{ROOT}/cache_system={system}_layers={layers[0]}-{layers[1]}_embedding={EMBEDDING}_theta={theta:.2f}_rep={repetition}.pkl"

    if os.path.isfile(filename):
        print(f"File already exists, skipping cache (if you want to force re-run, move or delete {filename} first)")
        return

    # Calculate remnants
    R_G, R_H, unobserved_edges, observed_edges = partial_information(G, H, theta)

    # Embed remnants
    if EMBEDDING == "N2V":
        E_G = N2V(R_G, embedding_parameters, embedding_hyperparameters, per_component=per_component)
        E_H = N2V(R_H, embedding_parameters, embedding_hyperparameters, per_component=per_component)
    elif EMBEDDING == "LE":
        nodelist = sorted(R_G.nodes())
        E_G = LE(R_G, embedding_parameters, embedding_hyperparameters, nodelist=nodelist, per_component=per_component)
        E_H = LE(R_H, embedding_parameters, embedding_hyperparameters, nodelist=nodelist, per_component=per_component)

    # Format class
    preprocessed_data = PreprocessedData(
        system, layers, theta,
        (R_G, R_H), (E_G, E_H),
        observed_edges, unobserved_edges
    )

    # Save to disk
    with open(filename, "wb") as _fh:
        pickle.dump(preprocessed_data, _fh, pickle.HIGHEST_PROTOCOL)


# --- Helpers ---
def _get_center_of_mass(vectors):
    return np.mean(vectors, axis=0)


def _align_centers(U, V):
    ubar = _get_center_of_mass(U)
    vbar = _get_center_of_mass(V)

    delta = ubar - vbar

    Vprime = [v + delta for v in V]

    return Vprime


def _center_to_origin(V):
    vbar = _get_center_of_mass(V)

    Vprime = [v - vbar for v in V]

    return Vprime