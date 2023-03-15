# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output

# --- Scientific ---
import numpy as np  # General computational tools

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
def per_embedding_distance(edge, vectors):
    # Book-keeping
    (i, j) = edge
    G, H = vectors
    x = np.array(G[i]) - np.array(G[j])
    y = np.array(H[i]) - np.array(H[j])

    # Calculate distance for each embedding
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    return (norm_x, norm_y)


# ============== MAIN ===============
def main(vectors, test_set):
    # Calculate distances system
    distances = {
        edge: per_embedding_distance(edge, vectors)
        for edge in test_set
    }

    return distances


if __name__ == "__main__":
    # Load system observation
    with open(snakemake.input["observation"], "rb") as _fh:
        test_set = pickle.load(_fh)["test_set"]

    # Load embeddings
    with open(snakemake.input["vectors"], "rb") as _fh:
        vectors = pickle.load(_fh)

    # Run observation procedure
    distances = main(vectors, test_set)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(distances, _fh, pickle.HIGHEST_PROTOCOL)
