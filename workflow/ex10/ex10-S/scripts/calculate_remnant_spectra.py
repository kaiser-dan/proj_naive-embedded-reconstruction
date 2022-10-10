# ============= SET-UP =================
# --- Standard library ---
import pickle
import warnings

# --- Scientific ---
import numpy as np  # General computational tools
from scipy.sparse.linalg import eigsh

# --- Network science ---
import networkx as nx

# --- Miscellaneous ---
warnings.filterwarnings("ignore")


# =================== FUNCTIONS ===================

# ============== MAIN ===============
def main(remnants, hyperparams):
    # Book-keeping
    ## Indexing objects
    _r = len(remnants)
    _nodes = sorted(remnants[0].nodes())  # * Force networkx indexing
    _nodes_reindexing = {node: idx for idx, node in enumerate(_nodes)}  # Allow for non-contiguous node indices

    ## Hyperparams
    dimension = np.array([hyperparams["dimension"]]*_r)
    maxiter = len(_nodes)*hyperparams["maxiter_multiplier"]
    if hyperparams["tol_exp"] >= 0:
        tol = 0
    else:
        tol = 10**hyperparams["tol_exp"]

    # Calculate normalized Laplacian
    L_normalized = tuple((
        nx.normalized_laplacian_matrix(G, nodelist=_nodes)
        for G in remnants
    ))

    # Account for first eigenvalue correlated with degrees
    dimension += np.array([1]*_r)
    # Account for algebraic multiplicity of trivial eigenvalues equal to number of connected components
    num_components = np.array([
        nx.number_connected_components(R)
        for R in remnants
    ])
    dimension += num_components

    # Calculate eigenspectra
    spectra = [
        eigsh(
            L_normalized[idx], k=dimension[idx],
            which="SM", maxiter=maxiter, tol=tol,
        )
        for idx in range(_r)
    ]

    # * Ensure algebraic multiplcity of trivial eigenvalue matches num_components
    w = [spectra_[0] for spectra_ in spectra]
    for idx, w_ in enumerate(w):
        trivial_ = sum([np.isclose(val, 0) for val in w_])
        components_ = num_components[idx]
        if trivial_ != components_:
            raise ValueError(
                f"""Number of components and algebraic multiplicity
                of trivial eigenvalue do not match in remnant layer {idx}
                Found {components_} components, {trivial_} near-0 eigenvalues
                {w_}
                """
                )

    # Retrieve eigenvectors and first non-trivial dimension-many components
    eigenvectors = [spectra_[1] for spectra_ in spectra]
    eigenvectors = [
        np.array([
            vector[-hyperparams["dimension"]:]
            for vector in eigenvectors_
        ])
        for eigenvectors_ in eigenvectors
    ]

    return eigenvectors, _nodes_reindexing

if __name__ == "__main__":
    # Load observation
    with open(snakemake.input.observation, "rb") as _fh:
        remnants = pickle.load(_fh)["remnant_duplex"]

    # Run observation procedure
    hyperparams = dict(snakemake.params)
    (eigenvectors, _nodes_reindexing) = main(remnants, hyperparams)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump((eigenvectors, _nodes_reindexing), _fh, pickle.HIGHEST_PROTOCOL)
