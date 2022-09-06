# =============== SET-UP =================
# --- Standard library ---
import time
import pickle

# --- Scientific ---
import numpy as np
from sklearn import metrics

# --- Network science ---
from node2vec import Node2Vec as N2V
from Utils import *

# ================ FUNCTIONS ================
def form_system(params):
    # Process parameters
    N, gamma, kmin, prob, sign = params["N"], params["gamma"], params["kmin"], params["prob"], params["sign"]
    pfi = params["pfi"]
    kmax = np.sqrt(N)

    # Form "raw" duplex
    D = generate_multiplex_configuration(N, gamma, kmin, kmax, prob, sign)

    # Split into layers
    G, H = duplex_network(D, 1, 2)

    # Observe partial information
    R_G, R_H, agg = partial_information(G, H, pfi)

    return G, H, R_G, R_H, agg

def embed_system(R_G, R_H, params):
    # Process parameters
    dimensions, walk_length, num_walks = params["dimensions"], params["walk_length"], params["num_walks"]
    workers = params["workers"]
    window, min_count, batch_words = params["window"], params["min_count"], params["batch_words"]

    # Generate walks
    R_G_emb = N2V(R_G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    R_H_emb = N2V(R_H, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    R_G_emb = R_G_emb.fit(window=window, min_count=min_count, batch_words=batch_words)
    R_H_emb = R_H_emb.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Retrieve embedded vectors
    G_ = R_G_emb.wv
    H_ = R_H_emb.wv

    return G_, H_

def check_accuracy(G, H, agg, G_, H_):
    cls = []
    gt = []
    for src, tgt in agg:
        v_G_src = G_[src, :]
        v_G_tgt = G_[tgt, :]
        v_H_src = H_[src, :]
        v_H_tgt = H_[tgt, :]

        d_G = np.linalg.norm(v_G_src - v_G_tgt)
        d_H = np.linalg.norm(v_H_src - v_H_tgt)

        if d_G < d_H:
            cls.append(0)
        elif d_H > d_G:
            cls.append(1)
        else:
            cls.append(np.random.randint(0, 2))

        if (src, tgt) in G.edges():
            gt.append(0)
        else:
            gt.append(1)

    acc = metrics.accuracy_score(gt, cls)
    return acc



# ================= MAIN ==============
def main(record):
    # Book-keeping
    _start_time = time.perf_counter()

    # Prepare experiment
    G, H, G_remnant, H_remnant, agg = form_system(record)  # sample duplex model
    G_embedded_remnant, H_embedded_remnant = embed_system(G_remnant, H_remnant, record)  # apply node2vec embedding
    record["_time_embedd"] = time.perf_counter() - _start_time

    # Measure performance
    v_G_remnant = G_embedded_remnant.vectors  # vectors as (N,d) matrices
    v_H_remnant = H_embedded_remnant.vectors
    accuracy = check_accuracy(G, H, agg, v_G_remnant, v_H_remnant)

    # Update accuracy column
    record["Accuracy"] = accuracy

    # Return record
    record["_time_all"] = time.perf_counter() - _start_time
    return record, G_embedded_remnant, H_embedded_remnant


if __name__ == "__main__":
    # Process workflow parameters
    record = {
        "N": int(snakemake.wildcards["N"]),
        "gamma": float(snakemake.wildcards["gamma"]),
        "kmin": int(snakemake.wildcards["kmin"]),
        "prob": float(snakemake.wildcards["prob"]),
        "sign": int(snakemake.wildcards["sign"]),
        "pfi": float(snakemake.wildcards["pfi"]),
        "rep": int(snakemake.wildcards["repetition"])
    }
    record.update(snakemake.params["config_embedding"])

    # Apply workflow
    record, G_embedded_remnant, H_embedded_remnant = main(record)

    # * NOTE: Currently fixing kmax as sqrt(N)!
    record["kmax"] = np.sqrt(record["N"])

    # Save embedding to disk (if applicable)
    if snakemake.params["save_embeddings"]:
        G_embedded_remnant.save_word2vec_format(snakemake.output[0].replace("raw", "embedding_original").replace("pkl", "mat"))
        H_embedded_remnant.save_word2vec_format(snakemake.output[0].replace("raw", "embedding_shuffled").replace("pkl", "mat"))

    # Save record to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(record, _fh, pickle.HIGHEST_PROTOCOL)
