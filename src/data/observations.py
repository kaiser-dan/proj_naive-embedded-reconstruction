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
ROOT = "../../"
sys.path.append(f"{ROOT}/src/")

from sampling.random import partial_information
from embed.N2V import N2V
from embed.LE import LE

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

    def renormalize(self):
        # Retrieve components
        R_G_components = [
            self.remnants[0].subgraph(component).copy()
            for component in nx.connected_components(self.remnants[0])
        ]
        R_H_components = [
            self.remnants[1].subgraph(component).copy()
            for component in nx.connected_components(self.remnants[1])
        ]
        R_G_components = sorted(R_G_components, key=len, reverse=True)
        R_H_components = sorted(R_H_components, key=len, reverse=True)

        # Identify GCC and it's mean norm
        R_G_GCC = R_G_components[0]
        R_H_GCC = R_H_components[0]

        R_G_GCC_norm = np.mean([
            np.linalg.norm(self.embeddings[0][node])
            for node in R_G_GCC.nodes()
        ])
        R_H_GCC_norm = np.mean([
            np.linalg.norm(self.embeddings[1][node])
            for node in R_H_GCC.nodes()
        ])


        # Renormalize components by mean GCC norm
        for component in R_G_components:
        # for component in R_G_components[1:]:
            # component_norm = np.mean([
            component_norm = np.sum([
                np.linalg.norm(self.embeddings[0][node])
                for node in component.nodes()
            ])
            if np.equal(component_norm, 0):
                component_norm += 1e-12

            renormalization_factor = 1 / component_norm
            # renormalization_factor = R_G_GCC_norm / component_norm
            for node in component.nodes():
                self.embeddings[0][node] = renormalization_factor * self.embeddings[0][node]

        for component in R_H_components:
        # for component in R_H_components[1:]:
            # component_norm = np.mean([
            component_norm = np.sum([
                np.linalg.norm(self.embeddings[1][node])
                for node in component.nodes()
            ])
            if np.equal(component_norm, 0):
                component_norm += 1e-12

            renormalization_factor = 1 / component_norm
            # renormalization_factor = R_H_GCC_norm / component_norm
            for node in component.nodes():
                self.embeddings[1][node] = renormalization_factor * self.embeddings[1][node]

        return self.embeddings

    def align_centers(self):
        # Retrieve components
        R_G_components = [
            list(component)
            for component in nx.connected_components(self.remnants[0])
        ]
        R_H_components = [
            list(component)
            for component in nx.connected_components(self.remnants[1])
        ]
        R_G_components = sorted(R_G_components, key=len, reverse=True)
        R_H_components = sorted(R_H_components, key=len, reverse=True)

        # Get vectors for each component
        R_G_components_vectors = [
            [self.embeddings[0][node] for node in component]
            for component in R_G_components
        ]
        R_H_components_vectors = [
            [self.embeddings[1][node] for node in component]
            for component in R_H_components
        ]

        # Align each vector set to GCC center
        R_G_shifted_components_vectors = []
        for V in R_G_components_vectors:
            R_G_shifted_components_vectors.append(_center_to_origin(V))
        R_H_shifted_components_vectors = []
        for V in R_H_components_vectors:
            R_H_shifted_components_vectors.append(_center_to_origin(V))

        # Replace old vector with shifted vector
        for component_id, component_vectors in enumerate(R_G_shifted_components_vectors):
            if len(component_vectors) == 1:
                # print("AHHH")
                # print(component_vectors)
                R_G_shifted_components_vectors[component_id] = \
                    _align_centers(
                        R_G_shifted_components_vectors[0],
                        component_vectors
                    )
                component_vectors = R_G_shifted_components_vectors[component_id]
                # print()

            for node_index, shifted_vector in enumerate(component_vectors):
                self.embeddings[0][R_G_components[component_id][node_index]] = shifted_vector
        for component_id, component_vectors in enumerate(R_H_shifted_components_vectors):
            if len(component_vectors) == 1:
                R_H_shifted_components_vectors[component_id] = \
                    _align_centers(
                        R_H_shifted_components_vectors[0],
                        component_vectors
                    )
                component_vectors = R_H_shifted_components_vectors[component_id]
            for node_index, shifted_vector in enumerate(component_vectors):
                self.embeddings[1][R_H_components[component_id][node_index]] = shifted_vector

        return self.embeddings


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
        E_G = N2V(R_G, embedding_parameters, embedding_hyperparameters)
        E_H = N2V(R_H, embedding_parameters, embedding_hyperparameters)
    elif EMBEDDING == "LE":
        nodelist = sorted(R_G.nodes())
        E_G = LE(R_G, embedding_parameters, embedding_hyperparameters, nodelist=nodelist)
        E_H = LE(R_H, embedding_parameters, embedding_hyperparameters, nodelist=nodelist)

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