# ============= SET-UP =================
# --- Standard library ---
import os  # for calling shell scripts
import sys
import pickle

# --- Scientific ---
import numpy as np  # General computational tools

# --- Network science ---
import networkx as nx
import mercator

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
# --- Drivers ---
def embed_system(remnant_G, remnant_H, hyperparams, embedding_method = "node2vec"):
    N = remnant_G.number_of_nodes()

    # Apply embedding to remnant graphs
    if embedding_method == "node2vec":
        R_G_embedded_model, R_H_embedded_model = helper_mercator(remnant_G, remnant_H, hyperparams)

        # * NOTE: Need to apply indexing function!
        R_G_vectors = helper_index_vectors_from_model(R_G_embedded_model, range(N))
        R_H_vectors = helper_index_vectors_from_model(R_H_embedded_model, range(N))
    else:
        raise NotImplementedError("Only Node2Vec prepared rn!")

    # Retrieve vectors from embedding

    return R_G_vectors, R_H_vectors


# --- Helpers ---
def helper_mercator(remnant_G, remnant_H, hyperparams):
    # Save temporary edge lists
    _fh_R_G = "G.edgelist"
    _fh_R_H = "H.edgelist"
    nx.write_edgelist(R_G, _fh_R_G)
    nx.write_edgelist(R_H, _fh_R_H)

    # Embed on edgelists

    # * Check if file already exists - won't automatically override because it is stupid!
    if os.path.exists(_fh_E_G):
        os.remove(_fh_E_G)
    if os.path.exists(_fh_E_H):
        os.remove(_fh_E_H)

    mercator.embed(_fh_R_G, output_name=_fh_E_G)
    mercator.embed(_fh_R_G, output_name=_fh_E_H)

    _fh_E_G = _fh_E_G + ".inf_coord"
    _fh_E_H = _fh_E_H + ".inf_coord"

    # Load vectors from file
    E_G = np.loadtxt(_fh_E_G, usecols=(2,3))
    E_H = np.loadtxt(_fh_E_H, usecols=(2,3))

    return E_G, E_H

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
