# ================= SET-UP =======================
# --- Standard library ---
import sys  # System pathing
from datetime import datetime  # Timestamping data
import time  # naive profiling
import pickle  # serialized data

# --- Scientific ---
import numpy as np  # General computational tools
from sklearn import metrics  # Measuring classifier performance

# --- Network science ---
import networkx as nx  # General network tools
from node2vec import Node2Vec as N2V  # Embedding tools

# --- Project source code ---
sys.path.append("../src/")
from Utils import *  # Custom synthetic benchmarks

# --- Other import aliases ---
today = datetime.today
accuracy = metrics.accuracy_score
auroc = metrics.roc_auc_score


# ================ FUNCTIONS ================
# --- Computations ---
def form_system(params):
    # Process parameters
    N = params["N"]
    tau1, tau2 = params["tau1"], params["tau2"]
    mu, min_community = params["mu"], params["min_community"]
    average_degree, max_degree = params["avg_k"], int(np.sqrt(N))
    prob_relabel = params["prob"]
    pfi = params["pfi"]

    # Form "raw" duplex
    D, _sigma1, _sigma2, _mu_temp = lfr_multiplex(N, tau1, tau2, mu, average_degree, max_degree, min_community, prob_relabel)

    # Split into layers
    G, H = duplex_network(D, 1, 2)

    # Observe partial information
    R_G, R_H, testset = partial_information(G, H, pfi)

    # Restrict to largest connected component (if specified)
    if params["largest_component"]:
        R_G_ = nx.Graph()
        R_H_ = nx.Graph()
        R_G_.add_nodes_from(R_G.nodes())
        R_H_.add_nodes_from(R_H.nodes())

        maxcc_R_G = max(nx.connected_components(R_G), key=len)
        maxcc_R_H = max(nx.connected_components(R_H), key=len)

        edges_R_G_ = set(R_G.subgraph(maxcc_R_G).edges())
        edges_R_H_ = set(R_H.subgraph(maxcc_R_H).edges())
        R_G_.add_edges_from(edges_R_G_)
        R_H_.add_edges_from(edges_R_H_)

        testset = {
            edge: gt_
            for edge, gt_ in testset.items()
            if edge in edges_R_H_ | edges_R_H_
        }

    return G, H, R_G, R_H, testset

def embed_system(R_G, R_H, params):
    # Process parameters
    dimensions, walk_length, num_walks = params["dimensions"], params["walk_length"], params["num_walks"]
    workers, window, min_count, batch_words = params["workers"], params["window"], params["min_count"], params["batch_words"]

    # Generate walks
    R_G_emb = N2V(R_G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True).fit(window=window, min_count=min_count, batch_words=batch_words)
    R_H_emb = N2V(R_H, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True).fit(window=window, min_count=min_count, batch_words=batch_words)

    # Retrieve embedded models
    G_ = R_G_emb.wv
    H_ = R_H_emb.wv

    return G_, H_

def reconstruct_system(R_G_vectors, R_H_vectors, testset, params):
    # Book-keeping
    ## Note: G is class 1, H is class 0
    cls = []  # hard classifications
    scores = []  # soft classifications
    gt = []  # true classification

    # Helper function for logistic likelihood
    _dot = lambda x, y: np.exp(-1*np.dot(np.transpose(x), y))

    # Classifying test set
    for edge, gt_ in testset.items():
        i, j = edge

        # Track true classification for performance calculations
        gt.append(gt_)

        # Retrieve embedded vectors for edge's incident nodes
        v_G_i = R_G_vectors[i]
        v_G_j = R_G_vectors[j]
        v_H_i = R_H_vectors[i]
        v_H_j = R_H_vectors[j]

        # Compute distance in each remnant embedding
        if params["distance"] == "logistic":
            d_G = 1 / (1 + _dot(v_G_i, v_G_j))
            d_H = 1 / (1 + _dot(v_H_i, v_H_j))
        else:
            d_G = np.linalg.norm(v_G_i - v_G_j)
            d_H = np.linalg.norm(v_H_i - v_H_j)

            if params["distance"] == "inverse":
                d_G = 1 / d_G
                d_H = 1 / d_H
            elif params["distance"] == "negexp":
                d_G = np.exp(-d_G)
                d_H = np.exp(-d_H)

        # Normalize likelihoods into probabilities
        t_G = d_G / (d_G + d_H)
        t_H = 1 - t_G

        scores.append(t_G)

        # Make hard classification based on relative probability
        cls_ = np.random.randint(2)
        if t_G != t_H:
            if np.random.rand() <= t_G:
                cls_ = 1
            else:
                cls_ = 0
        cls.append(cls_)

    return cls, scores, gt

# --- Helpers ---
def measure_performance(cls, scores, gt):
    acc = accuracy(gt, cls)
    auc = auroc(gt, scores)

    return acc, auc

# ================= MAIN ==============
def main(record):
    # Book-keeping
    _start_time = time.perf_counter()

    # Prepare experiment
    G, H, R_G, R_H, testset = form_system(record)  # sample duplex model
    R_G_embedding, R_H_embedding = embed_system(R_G, R_H, record)  # apply node2vec embedding
    record["_time_embedd"] = time.perf_counter() - _start_time

    # ! Reindex to avoid (pseudo) random node reindexing
    R_G_vectors = {
        int(R_G_embedding.index_to_key[i]) : R_G_embedding.vectors[i]
        for i in range(G.number_of_nodes())
    }
    R_H_vectors = {
        int(R_H_embedding.index_to_key[i]) : R_H_embedding.vectors[i]
        for i in range(H.number_of_nodes())
    }

    # Reconstruct system from embeddings
    classifications, scores, ground_truth = reconstruct_system(R_G_vectors, R_H_vectors, testset, record)

    # Measure performance
    auroc_ = auroc(ground_truth, scores)
    accuracy_ = accuracy(ground_truth, classifications)

    # Update accuracy column
    record["AUROC"] = auroc_
    record["Accuracy"] = accuracy_

    # Return record
    record["_time_all"] = time.perf_counter() - _start_time
    return record, R_G_embedding, R_H_embedding


if __name__ == "__main__":
    # Process workflow parameters
    # Network parameters
    N = int(snakemake.wildcards["N"])
    tau1 = float(snakemake.wildcards["tau1"])
    tau2 = float(snakemake.wildcards["tau2"])
    avg_k = int(snakemake.wildcards["avg_k"])
    min_community = int(snakemake.wildcards["min_community"])
    mu = float(snakemake.wildcards["mu"])
    prob = float(snakemake.wildcards["prob"])
    pfi = float(snakemake.wildcards["pfi"])
    # Experiment parameters
    distance = str(snakemake.wildcards["distance"])
    rep = int(snakemake.wildcards["repetition"])
    largest_component = bool(snakemake.params["largest_component"])

    # Initializing experiment record
    record = {
        "N": N,
        "tau1": tau1, "tau2": tau2,
        "avg_k": avg_k,
        "min_community": min_community, "mu": mu,
        "prob": prob,
        "pfi": pfi,
        "distance": distance, "rep": rep, "largest_component": largest_component
    }
    record.update(snakemake.params["config_embedding"])

    # Apply workflow
    record, R_G_embedding, R_H_embedding = main(record)

    # * NOTE: Currently fixing kmax as sqrt(N)!
    record["kmax"] = np.sqrt(record["N"])

    # Save embedding to disk (if applicable)
    if snakemake.params["save_embeddings"]:
        R_G_embedding.save_word2vec_format(snakemake.params["template_vector"].replace("raw", "embedding_original"))
        R_H_embedding.save_word2vec_format(snakemake.params["template_vector"].replace("raw", "embedding_shuffled"))

    # Save record to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(record, _fh, pickle.HIGHEST_PROTOCOL)
