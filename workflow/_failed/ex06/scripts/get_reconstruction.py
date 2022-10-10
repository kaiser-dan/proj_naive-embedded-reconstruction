# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output

# --- Scientific ---
import numpy as np
from scipy.sparse.linalg import eigsh

# --- Network science ---
import networkx as nx

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
# --- Drivers ---
def reconstruct_system(remnants, testset, metric, num_eigenvalues):
    # Book-keeping
    edge_gt_pairs = testset.items()

    # Retrieve eigenspectra
    eigenvalues, eigenvectors = helper_get_eigenspectra(remnants, num_eigenvalues)

    # Apply reconstruction
    reconstruction_wide = [
        classify_edge(egp, eigenvectors, metric, num_eigenvalues)
        for egp in edge_gt_pairs
    ]

    # Tidy data
    scores_ = [r[0] for r in reconstruction_wide]
    classes_ = [r[1] for r in reconstruction_wide]
    ground_truths_ = [r[2] for r in reconstruction_wide]
    df = pd.DataFrame(
        {
            "Score": scores_,
            "Classification": classes_,
            "Ground_Truth": ground_truths_,
            "Metric": np.repeat(metric, len(edge_gt_pairs))
        },
        index=[egp_[0] for egp_ in edge_gt_pairs]
    )

    return df


def classify_edge(edge_gt_pair, eigenvectors, metric="inverse", num_eigenvalues=50):
    """
    Retrieve, for one edge, the (score, class, gt) under the remnant embeddings.
    """
    # Book-keeping
    edge, ground_truth_ = edge_gt_pair

    # Apply classification functions
    score_ = helper_binary_score_from_spectra(edge, eigenvectors, metric, num_eigenvalues)
    class_ = helper_binary_classify_edge_from_score(score_)

    return (score_, class_, ground_truth_)


# --- Helpers ---
def helper_get_eigenspectra(remnants, num_eigenvalues):
    """
    Calculate (eigenvalues, eigenvectors) for remnant graphs - include first [num_eigenvalues]
    """
    # Book-keeping
    remnant_G, remnant_H = remnants
    nodes_ = sorted(remnant_G.nodes())

    # Calculate laplacians
    _normalized_laplacians = (
        _get_norm_lap(remnant_G, nodes_), _get_norm_lap(remnant_H, nodes_)
    )

    # Calculate eigenspectra
    _eigenspectra = (
        eigsh(_normalized_laplacians[0], k=num_eigenvalues, which="SM"),
        eigsh(_normalized_laplacians[1], k=num_eigenvalues, which="SM")
    )

    # Group returns
    eigenvalues = [spectra[0] for spectra in _eigenspectra]
    eigenvectors = [spectra[1] for spectra in _eigenspectra]

    return eigenvalues, eigenvectors


def helper_binary_score_from_spectra(edge, eigenvectors, metric, num_eigenvalues):
    # Book-keeping
    i, j = edge
    x, y = eigenvectors

    # * NOTE: The first eigencomponent is proportional to the node degree and is excluded in clustering
    d1 = np.linalg.norm(x[i, 1:num_eigenvalues] - x[j, 1:num_eigenvalues])
    d2 = np.linalg.norm(y[i, 1:num_eigenvalues] - y[j, 1:num_eigenvalues])

    # Add small amount to avoid ZeroDivisionErrors
    d1 += 1e-60
    d2 += 1e-60

    if metric == "inverse":
        d1 = 1 / d1
        d2 = 1 / d2
    elif metric == "negexp":
        d1 = np.exp(-d1)
        d2 = np.exp(-d2)
    elif metric == "logistic":
        d1 = 1 / (1 + np.exp(-d1))
        d2 = 1 / (1 + np.exp(-d2))

    return d1 / (d1 + d2)


def helper_binary_classify_edge_from_score(t_G):
    """
    Classify edge based on random draw weighted by score towards layer α
    """
    # If within tolerance of 0.5, randomly classify
    if not np.isclose(t_G, 0.5):
        # If score non-central, assign to G weighted by score
        class_ = int(np.random.rand() <= t_G)
    else:
        class_ = np.random.randint(2)

    return class_


# --- Anonymous helpers ---
def _get_norm_lap(G, nodelist):
    return nx.normalized_laplacian_matrix(G, nodelist=nodelist)


# ============== MAIN ===============
def main(remnants, test_set, metric="inverse", num_eigenvalues=50):
    # Reconstruct system
    reconstruction = reconstruct_system(remnants, test_set, metric, num_eigenvalues)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(reconstruction, _fh, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Load pickled system from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        system_ = pickle.load(_fh)

    # Run observation procedure
    main(system_["remnant_duplex"], system_["test_set"], snakemake.wildcards["metric"], snakemake.params["num_eigenvalues"])
