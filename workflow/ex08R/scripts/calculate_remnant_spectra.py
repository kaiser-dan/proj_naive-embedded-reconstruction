# ============= SET-UP =================
# --- Standard library ---
import pickle

# --- Scientific ---
import numpy as np  # General computational tools
from scipy.sparse.linalg import eigsh

# --- Network science ---
import networkx as nx

# --- Data handling and visualization ---


# =================== FUNCTIONS ===================

# ============== MAIN ===============
def main(remnants, num_values):
    # Book-keeping
    _nodes = sorted(remnants[0].nodes())  # * Force networkx indexing
    _nodes_reindexing = {node: idx for idx, node in enumerate(_nodes)}

    # Calculate normalized Laplacian
    L_normalized = tuple((nx.normalized_laplacian_matrix(G, nodelist=_nodes) for G in remnants))

    # Calculate eigenspectra
    spectra_ = [eigsh(L, k=num_values, which="SM") for L in L_normalized]

    # * Remove first eigenvalue, correlated with degree
    # * Set aside trivial eigenvalues
    nontrivial_eigenvalue_column_indices = [
        [
            idx for idx in range(1, num_values)
            if not np.isclose(spectra_G[0][idx], 0)
        ]
        for spectra_G in spectra_
    ]

    spectra = [
        # Eigenvectors
        np.array([
            row[nontrivial_eigenvalue_column_indices[j]]
            for row in spectra_[j][1]
        ])
        for j in range(len(spectra_))
    ]

    return spectra, _nodes_reindexing

if __name__ == "__main__":
    # Load observation
    with open(snakemake.input.observation, "rb") as _fh:
        remnants = pickle.load(_fh)["remnant_duplex"]

    # Run observation procedure
    spectra = main(remnants, snakemake.params.num_values)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(spectra, _fh, pickle.HIGHEST_PROTOCOL)
