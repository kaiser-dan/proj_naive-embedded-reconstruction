# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output
import snakemake

# --- Scientific ---
import numpy as np  # General computational tools

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
# --- Drivers ---
def reconstruct_system(R_G_vectors, R_H_vectors, testset, metric):
    # Book-keeping
    matrices = (R_G_vectors, R_H_vectors)
    edge_gt_pairs = testset.items()
    m_ = len(edge_gt_pairs)

    # Apply reconstruction
    reconstruction_wide = [
        classify_edge(egp, matrices, metric)
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
            "Metric": np.repeat(metric, m_)
        },
        index=[egp_[0] for egp_ in edge_gt_pairs]
    )

    return df

def classify_edge(edge_gt_pair, matrices, metric="inverse"):
    """
    Retrieve, for one edge, the (score, class, gt) under the remnant embeddings.
    """
    # Book-keeping
    (i,j), ground_truth_ = edge_gt_pair
    G,H = matrices

    # Calulate difference vector for edge
    x = np.array(G[i]) - np.array(G[j])
    y = np.array(H[i]) - np.array(H[j])

    # Apply classification functions
    scores = helper_binary_score_from_vectors(x,y, metric)
    score_ = scores[0]
    class_ = helper_binary_classify_edge_from_score(score_)

    return (score_, class_, ground_truth_)


# --- Helpers ---
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

def helper_binary_score_from_vectors(x,y, metric="inverse"):
    """
    Calculate α likelihood for two vectors
    """
    if metric == "inverse":
        _metric = _metric_inverse
    elif metric == "negexp":
        _metric = _metric_negexp
    elif metric == "logistic":
        _metric = _metric_logistic

    scores = _metric(x,y)

    return scores


# --- Anonymous helpers ---
def _check_degeneracy(distances):
    _compare = lambda d_: d_ == 0  # Element-wise comparison
    return any(np.vectorize(_compare)(distances))  # Apply comparison to vector

def _metric_inverse(x,y):
    # Apply norm
    d = [np.linalg.norm(x), np.linalg.norm(y)]

    # Check degeneracy
    if _check_degeneracy(d):
        return np.array([1/len(d)]*len(d))

    # Apply multiplicative inverse
    metric_d = np.array([1/d_  for d_ in d])

    # Convert to score
    normalizing_Z = np.sum(metric_d)
    scores = metric_d / normalizing_Z

    return scores

def _metric_negexp(x,y):
    # Apply norm
    d = [np.linalg.norm(x), np.linalg.norm(y)]

    # Check degeneracy
    if _check_degeneracy(d):
        return np.array([1/len(d)]*len(d))

    # Apply multiplicative inverse
    metric_d = np.array([np.exp(-d_)  for d_ in d])

    # Convert to score
    normalizing_Z = np.sum(metric_d)
    scores = metric_d / normalizing_Z

    return scores

def _metric_logistic(x,y):
    # Anonymous functions to shorten script lines lol
    _exp = lambda x, y: np.exp(np.dot(np.transpose(x), y))
    _frac = lambda exp_: exp_ / (exp_ + 1)
    _logistic = lambda x, y: _frac(_exp(x,y))

    # Apply logistic distances
    metric_d = np.array(
        map(_logistic, x,y)
    )

    # * Do not need to check degeneracy here, strictly positive
    # * We are not operating on norms

    # Convert to score
    normalizing_Z = np.sum(metric_d)
    scores = metric_d / normalizing_Z

    return scores


# ============== MAIN ===============
def main(distances, observation, metric):
    # Book-keeping
    m = len(distances)
    edges = list(distances.keys())
    originations = [observation["test_set"][edge] for edge in edges]

    scores = [None]*m
    classifications = [None]*m

    # Reconstruct system
    for idx, edge in enumerate(edges):
        score = SCORE
        classification = CLASSIFICATION
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
    with open(snakemake.input["distances"], "rb") as _fh:
        distances = pickle.load(_fh)

    # Load observation
    with open(snakemake.input["observation"], "rb") as _fh:
        observation = pickle.load(_fh)

    # Run observation procedure
    reconstruction = main(distances, observation, snakemake.params["metric"])

        # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(reconstruction, _fh, pickle.HIGHEST_PROTOCOL)