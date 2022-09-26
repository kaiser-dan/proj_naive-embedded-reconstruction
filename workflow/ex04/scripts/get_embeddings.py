# ============= SET-UP =================
# --- Standard library ---
import os  # for calling shell scripts
import sys
import pickle

# --- Scientific ---
import numpy as np  # General computational tools

# --- Network science ---
import networkx as nx
from node2vec import Node2Vec as N2V  # Embedding tools

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
# --- Drivers ---
def embed_system(remnant_G, remnant_H, hyperparams, embedding_method = "node2vec"):
    N = remnant_G.number_of_nodes()

    # Apply embedding to remnant graphs
    if embedding_method == "node2vec":
        R_G_embedded_model, R_H_embedded_model = helper_node2vec(remnant_G, remnant_H, hyperparams)

        # * NOTE: Need to apply indexing function!
        R_G_vectors = helper_index_vectors_from_model(R_G_embedded_model, range(N))
        R_H_vectors = helper_index_vectors_from_model(R_H_embedded_model, range(N))
    else:
        raise NotImplementedError("Only Node2Vec prepared rn!")

    # Retrieve vectors from embedding

    return R_G_vectors, R_H_vectors


# --- Helpers ---
def helper_node2vec(remnant_G, remnant_H, hyperparams):
    # Process hyperparameters
    # Embedding parameters
    dimensions, walk_length, num_walks = hyperparams["dimensions"], hyperparams["walk_length"], hyperparams["num_walks"]

    # Computing parameters
    workers, window, min_count, batch_words = hyperparams["workers"], hyperparams["window"], hyperparams["min_count"], hyperparams["batch_words"]

    # Generate walks
    R_G_embedding = N2V(remnant_G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    R_H_embedding = N2V(remnant_H, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)

    # Fit word2vec models
    R_G_embedding = R_G_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
    R_H_embedding = R_H_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)

    # Retrieve embedded models
    R_G_embedded_model = R_G_embedding.wv
    R_H_embedded_model = R_H_embedding.wv

    return R_G_embedded_model, R_H_embedded_model

def helper_index_vectors_from_model(embedding_model, index_range):
    index_ = embedding_model.index_to_key
    vectors_ = embedding_model.vectors

    vectors = {int(index_[idx]): vectors_[idx] for idx in index_range}

    return vectors
# ============== MAIN ===============
def main(system_, parameters):
    # Book-keeping
    remnant_G, remnant_H = system_["remnant_duplex"]

    # Embed vectors
    remnant_vectors = embed_system(remnant_G, remnant_H, parameters, embedding_method = "node2vec")

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(remnant_vectors, _fh, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Load pickled duplex from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        system_ = pickle.load(_fh)

    # Run observation procedure
    main(system_, snakemake.params["hyperparams"])
