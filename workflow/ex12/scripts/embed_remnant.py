# ============= SET-UP =================
# --- Standard library ---
import os  # for calling shell scripts
import sys
import pickle

# --- Scientific ---
import numpy as np  # General computational tools
from scipy.sparse.linalg import eigsh

# --- Network science ---
import networkx as nx
from node2vec import Node2Vec as N2V  # Embedding tools

# --- Data handling and visualization ---
import pandas as pd


# =================== FUNCTIONS ===================
# --- Drivers ---
def embed_system(remnant_G, remnant_H, hyperparams, embedding_method = "node2vec", per_component=False):
    N = remnant_G.number_of_nodes()

    # Apply embedding to remnant graphs
    if embedding_method == "node2vec":
        R_G_embedded_model, R_H_embedded_model = helper_node2vec(remnant_G, remnant_H, hyperparams, per_component=per_component)

        # * NOTE: Need to apply indexing function!
        R_G_vectors = helper_index_vectors_from_model(R_G_embedded_model, range(N), per_component=per_component)
        R_H_vectors = helper_index_vectors_from_model(R_H_embedded_model, range(N), per_component=per_component)
    elif embedding_method == "LE":
        remnants = (remnant_G, remnant_H)
        R_G_vectors, R_H_vectors = LE(remnants, hyperparams)
    else:
        raise NotImplementedError("Invalid embedding method!")

    # Retrieve vectors from embedding
    vectors = (R_G_vectors, R_H_vectors)

    return vectors

# --- Helpers ---
def LE(remnants, hyperparams):
    # Book-keeping
    ## Indexing objects for remnants
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
            ncv=6*dimension[idx]+1
        )
        for idx in range(_r)
    ]

    # * Ensure algebraic multiplcity of trivial eigenvalue matches num_components
    w = [spectra_[0] for spectra_ in spectra]
    for idx, w_ in enumerate(w):
        trivial_ = sum([np.isclose(val, 0) for val in w_])
        components_ = num_components[idx]
        if trivial_ != components_:
            print(
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

    out_dict_remnants_1={_nodes_reindexing[node]:eigenvectors[0][node] for node in _nodes}
    out_dict_remnants_2={_nodes_reindexing[node]:eigenvectors[1][node] for node in _nodes}

    return out_dict_remnants_1, out_dict_remnants_2


def helper_node2vec(remnant_G, remnant_H, hyperparams, per_component=True):
    # Process hyperparameters
    # Embedding parameters
    dimensions, walk_length, num_walks = hyperparams["dimensions"], hyperparams["walk_length"], hyperparams["num_walks"]

    # Computing parameters
    workers, window, min_count, batch_words = hyperparams["workers"], hyperparams["window"], hyperparams["min_count"], hyperparams["batch_words"]


    # Begin embedding (possibly by component)
    if per_component:
        G_ = []
        H_ = []
        for c, component in enumerate(nx.connected_components(remnant_G)):
            component = remnant_G.subgraph(component).copy()

            R_G_embedding = N2V(component, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
            R_G_embedding = R_G_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
            R_G_embedded_model = R_G_embedding.wv
            G_.append(R_G_embedded_model)

        for component in nx.connected_components(remnant_H):
            component = remnant_H.subgraph(component).copy()

            R_H_embedding = N2V(component, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
            R_H_embedding = R_H_embedding.fit(window=window, min_count=min_count, batch_words=batch_words)
            R_H_embedded_model = R_H_embedding.wv
            H_.append(R_H_embedded_model)

        return G_, H_

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

        return R_G_embedded_model, R_H_embedded_model

def helper_index_vectors_from_model(embedding_model, node_index_range, per_component=True):
    if per_component:
        vectors = {node_: 0 for node_ in node_index_range}

        for component in embedding_model:
            index_ = component.index_to_key
            vectors_ = component.vectors

            vectors.update({
                int(index_[idx]): vectors_[idx]
                for idx in range(len(index_))
            })

    else:
        index_ = embedding_model.index_to_key
        vectors_ = embedding_model.vectors

        vectors = {int(index_[idx]): vectors_[idx] for idx in node_index_range}

    return vectors
# ============== MAIN ===============
def main(system_, parameters):
    # Book-keeping
    remnant_G, remnant_H = system_["remnant_duplex"]
    observed_G, observed_H = system_["observed_duplex"]  # used in convex embedding

    # Embed vectors
    remnant_vectors = embed_system(remnant_G, remnant_H, parameters, embedding_method = parameters["method"])  # embed R
    observed_vectors = embed_system(observed_G, observed_H, parameters, embedding_method = parameters["method"])  # embed theta

    return remnant_vectors, observed_vectors

if __name__ == "__main__":
    # Load pickled duplex from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        system_ = pickle.load(_fh)

    # Run observation procedure
    embedded_vectors = main(system_, snakemake.params["hyperparams"])

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(embedded_vectors, _fh, pickle.HIGHEST_PROTOCOL)
