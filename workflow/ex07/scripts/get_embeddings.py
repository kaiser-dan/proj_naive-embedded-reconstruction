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
        R_G_embedded_model, R_H_embedded_model, mapping = helper_node2vec(remnant_G, remnant_H, hyperparams, per_component=True)

        # * NOTE: Need to apply indexing function!
        R_G_vectors = helper_index_vectors_from_model(R_G_embedded_model, range(N), {key: val[0] for key, val in mapping.items()}, per_component=True)
        R_H_vectors = helper_index_vectors_from_model(R_H_embedded_model, range(N), {key: val[1] for key, val in mapping.items()}, per_component=True)
    else:
        raise NotImplementedError("Only Node2Vec prepared rn!")

    # Retrieve vectors from embedding

    return R_G_vectors, R_H_vectors


# --- Helpers ---
def helper_node2vec(remnant_G, remnant_H, hyperparams, per_component=True):
    # Process hyperparameters
    # Embedding parameters
    dimensions, walk_length, num_walks = hyperparams["dimensions"], hyperparams["walk_length"], hyperparams["num_walks"]

    # Computing parameters
    workers, window, min_count, batch_words = hyperparams["workers"], hyperparams["window"], hyperparams["min_count"], hyperparams["batch_words"]


    # Begin embedding (possibly by component)
    if per_component:
        mapping = {node_: [0,0] for node_ in remnant_G.number_of_nodes()}
        G_ = []
        H_ = []
        for c, component in enumerate(nx.connected_components(remnant_G)):
            for node_ in component:
                mapping[node_][0] = c
            component = remnant_G.subgraph(c).copy()

            R_G_embedding = N2V(component, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
            R_G_embedding = R_G_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
            R_G_embedded_model = R_G_embedding.wv
            G_.append(R_G_embedded_model)
        for component in nx.connected_components(remnant_H):
            for node_ in component:
                mapping[node_][1] = c
            component = remnant_H.subgraph(c).copy()

            R_H_embedding = N2V(component, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
            R_H_embedding = R_H_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
            R_H_embedded_model = R_H_embedding.wv
            H_.append(R_H_embedded_model)

        return G_, H_, mapping

    else:
        # Generate walks
        R_G_embedding = N2V(remnant_G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
        R_H_embedding = N2V(remnant_H, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)

        # Fit word2vec models
        R_G_embedding = R_G_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
        R_H_embedding = R_H_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)

        # Retrieve embedded models
        R_G_embedded_model = R_G_embedding.wv
        R_H_embedded_model = R_H_embedding.wv

        return R_G_embedded_model, R_H_embedded_model, None

def helper_index_vectors_from_model(embedding_model, node_index_range, mapping, per_component=True):
    if per_component:
        vectors = {}

        for node_ in node_index_range:
            component_model = embedding_model[mapping[node_]]
            index_ = component_model.index_to_key
            vectors_ = component_model.vectors

            vectors.update(
                {int(index_[idx]): vectors_[idx] for idx in node_index_range}
            )

    else:
        index_ = embedding_model.index_to_key
        vectors_ = embedding_model.vectors

        vectors = {int(index_[idx]): vectors_[idx] for idx in node_index_range}

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
