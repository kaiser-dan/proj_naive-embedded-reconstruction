"""Script to train and evaluate reconstruction from cached embeddings.
"""
# ================= SET-UP =======================
# >>> Standard library <<<
# --- System ---
import os
import sys

# --- Scientific computing ---
from multiprocessing import Pool

# --- Data handling ---
import pickle
import pandas as pd

# --- Logging and configuration ---
import logging

# >>> Project source <<<
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

# --- Data ---
from src.data.caches import CachedEmbeddings

# --- Embeddings ---
from src.embed import helpers

# --- Classifiers ---
from src.classifiers import logreg as lr
from src.classifiers import features

# >>> Globals <<<
# Logistic regression hyperparameters
LOGREG = {
    "penalty": None,
    "solver": "newton-cg",
}

DIR_EDGELISTS = os.path.join(ROOT, "data", "input", "SYSLFR", "edgelists", "")
DIR_CACHES = os.path.join(ROOT, "data", "input", "SYSLFR", "caches", "")
DIR_DFS = os.path.join(ROOT, "data", "output", "dataframes", "")

# >>> Miscellaneous <<<
logging.basicConfig(level=logging.INFO)

# ================ FUNCTIONS ======================
# >>> Data handling <<<
def get_data(cache_fp):
    # Get caches
    with open(cache_fp, 'rb') as _fh:
        cache = pickle.load(_fh)

    # Get original edgelists
    edgelists_fp = f"{DIR_EDGELISTS}/edgelists{cache_fp.split('edgelists')[1]}"
    with open(edgelists_fp, 'rb') as _fh:
        edgelists = pickle.load(_fh)

    # Fix edge labels
    ## Training edges
    fixed_observed_edges = {}
    for edge in cache.observed_edges:
        if edgelists[0].has_edge(*edge):
            fixed_observed_edges[edge] = 1
        else:
            fixed_observed_edges[edge] = 0
    cache.observed_edges = fixed_observed_edges

    ## Testing edges
    fixed_unobserved_edges = {}
    for edge in cache.unobserved_edges:
        if edgelists[0].has_edge(*edge):
            fixed_unobserved_edges[edge] = 1
        else:
            fixed_unobserved_edges[edge] = 0
    cache.unobserved_edges = fixed_unobserved_edges

    return cache

# >>> Computers <<<
def get_training_features(cache, observed_edges):
    Y = features.get_labels(observed_edges)

    # Degree feature
    src_G, tgt_G = features.get_degrees(cache.remnants[0].remnant, observed_edges)
    src_H, tgt_H = features.get_degrees(cache.remnants[1].remnant, observed_edges)
    degree_products_G = src_G * tgt_G
    degree_products_H = src_H * tgt_H
    X_degrees = features.as_configuration(degree_products_G, degree_products_H)

    # Distances feature
    distances_G = features.get_distances(cache.embeddings[0].vectors, observed_edges)
    distances_H = features.get_distances(cache.embeddings[1].vectors, observed_edges)
    X_distances = features.as_configuration(distances_G, distances_H)

    # Feature matrix
    X = features.format_feature_matrix((X_degrees, X_distances))

    return X, Y

def get_testing_features(cache, unobserved_edges):
    # Get testing labels
    Y = features.get_labels(unobserved_edges)

    # Degree feature
    src_G, tgt_G = features.get_degrees(cache.remnants[0].remnant, unobserved_edges)
    src_H, tgt_H = features.get_degrees(cache.remnants[1].remnant, unobserved_edges)
    degree_products_G = src_G * tgt_G
    degree_products_H = src_H * tgt_H
    X_degrees = features.as_configuration(degree_products_G, degree_products_H)

    # Distances feature
    distances_G = features.get_distances(cache.embeddings[0].vectors, unobserved_edges)
    distances_H = features.get_distances(cache.embeddings[1].vectors, unobserved_edges)
    X_distances = features.as_configuration(distances_G, distances_H)

    # Feature matrix
    X = features.format_feature_matrix((X_degrees, X_distances))

    return X, Y

def evaluate(cache_fp):
    # --- Book-keeping ---
    record = {
        "accuracy": None,
        "auroc": None,
        "pr": None
    }
    
    cache = get_data(cache_fp)
    
    logging.debug(f"Starting {cache_fp}")

    # Set up features
    # cache.embeddings[0].normalize(helpers.get_components(cache.remnants[0].remnant))
    # cache.embeddings[1].normalize(helpers.get_components(cache.remnants[1].remnant))
    X_train, Y_train = get_training_features(cache, cache.observed_edges)
    X_test, Y_test = get_testing_features(cache, cache.unobserved_edges)

    # Train model
    model = lr.LogReg("LogReg", ("deg", "emb"), dict(), X_train, Y_train, LOGREG)

    # Evaluate reconstruction(s)
    acc = model.testing_performance(X_test, Y_test, "ACC")
    auroc = model.testing_performance(X_test, Y_test, "AUROC")
    pr = model.testing_performance(X_test, Y_test, "PR")

    # Form output record
    record.update({
        "accuracy": acc,
        "auroc": auroc,
        "pr": pr,
        "method": cache_fp.split("method")[1].split("_")[0][1:],
        "theta": cache_fp.split("theta")[1].split("_")[0][1:],
        "mu": cache_fp.split("mu")[1].split("_")[0][1:],
    })

    # Return
    return record

# >>> Drivers <<<
def analysis(
        query="",
        processes=4,
        chunksize=12):
    # --- Book-keeping ---
    # Restrict CachedEmbeddings roster based on query
    dir_ = [
        f"{DIR_CACHES}/{fp}" 
        for fp in os.listdir(DIR_CACHES)
        if query in fp and "N-10000" in fp
    ]
    
    logging.info(f"Reconstructing {len(dir_)} CachedEmbeddings")
    
    # --- Computations ---
    with Pool(processes=processes) as p:
        records = p.map(evaluate, dir_, chunksize=chunksize)
    
    logging.info("Finished reconstructions; saving to dataframe")
    
    # Save to disk
    df = pd.DataFrame.from_records(records)
    df.to_csv(f"{DIR_DFS}/dataframe_EMB_exSYSLFR_query-{query}.csv", index=False)

    
    
# ========== MAIN ===========
def main():
    # --- Dispatch ---
    args = sys.argv  # gather CL args
    
    # No args passed, run defaults
    if len(args) == 1:
        analysis()
        quit()  # quit so we don't trigger next conditionals
    
    # Some args passed
    analysis_args = []
    ## Query
    if len(args) >= 2:
        logging.info(f"Found query argument '{sys.argv[1]}'")
        analysis_args.append(sys.argv[1])
    
    ## Query + processes
    if len(args) >= 3:
        logging.info(f"Found processes argument '{sys.argv[2]}'")
        analysis_args.append(int(sys.argv[2]))
        
    ## Query + processes + chunksize
    if len(args) == 4:
        logging.info(f"Found chunksize argument '{sys.argv[3]}'")
        analysis_args.append(int(sys.argv[3]))
        
    ## Run process
    analysis(*analysis_args)
    
    
    
if __name__ == "__main__":
    main()
