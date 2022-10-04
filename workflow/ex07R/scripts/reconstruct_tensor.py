# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output

# --- Scientific ---
import numpy as np  # General computational tools

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
def select_metric(metric):
    if metric == "inverse":
        metric_ = lambda x, y: (1/x) / ((1/x) + (1/y))
    elif metric == "negexp":
        metric_ = lambda x, y: np.exp(-x) / (np.exp(-x) + np.exp(-y))

    return metric_


def classify_from_score(score):
    # If within tolerance of 0.5, randomly classify
    if not np.isclose(score, 0.5):
        # If score non-central, assign to G weighted by score
        class_ = int(np.random.rand() <= score)
    else:
        class_ = np.random.randint(2)

    return class_


# ============== MAIN ===============
def main(distances, observation, metric):
    # Book-keeping
    m = len(distances)
    edges = list(distances.keys())
    originations = [observation["test_set"][edge] for edge in edges]
    metric_ = select_metric(metric)

    scores = [None]*m
    classifications = [None]*m

    # Reconstruct system
    for idx, (_, distances_) in enumerate(distances.items()):
        # Prevent degeneracy
        distances_ = [d + 1e-16 for d in distances_]

        # Calculate score and draw classification
        score = metric_(*distances_)
        classification = classify_from_score(score)
        scores[idx] = score
        classifications[idx] = classification

    reconstruction = {
        "edge": edges,
        "score": scores,
        "classification": classifications,
        "origination": originations
    }

    return reconstruction


if __name__ == "__main__":
    # Load distances
    with open(snakemake.input.distances, "rb") as _fh:
        distances = pickle.load(_fh)

    # Load observation
    with open(snakemake.input.observation, "rb") as _fh:
        observation = pickle.load(_fh)

    # Run observation procedure
    reconstruction = main(distances, observation, snakemake.params.metric)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(reconstruction, _fh, pickle.HIGHEST_PROTOCOL)
