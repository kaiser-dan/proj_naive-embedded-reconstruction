"""CLI to calculate training and test feature matrices (and labels) and train model.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import pickle
import argparse

# --- Source code ---
from embmplxrec import classifiers
from embmplxrec import embeddings
from embmplxrec import features
import embmplxrec.utils

# --- Globals ---
## Parameters
PARAMS_LOGREG = {
    "penalty": None,
    "solver": "newton-cholesky",
}

## Filepaths & templates
# * Relative to project root!
DIR_EDGELISTS = os.path.join("data", "input", "edgelists", "")
DIR_REMNANTS = os.path.join("data", "input", "remnants", "")
DIR_EMBEDDINGS = os.path.join("data", "interim", "caches", "")

DIR_MODELS = os.path.join("data", "interim", "debug_models", "")
FILEPATH_TEMPLATE = "model_normalized-{normalize}_{basename}"


## Logging
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/train_model_{embmplxrec.utils.get_today(time=True)}.log",
    console_level=30)

# ================ FUNCTIONS ======================
# --- Command-line Interface ---
def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Calculate features matrices and train reconstruction model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument(
        "filepath_remnants",
        type=str,
        help="Filepath of pickled remnant duplex.")
    parser.add_argument(
        "filepath_embeddings",
        type=str,
        help="Filepath of pickled remnant embeddings.")

    # Optional arguments, flag
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embedding vectors before calculating distance-based features.")

    return parser

def verify_args(args):
    # - File input -
    for fp in [args.filepath_remnants, args.filepath_embeddings]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"{fp} not found!")

    return

def gather_args():
    # Initialize argparse
    parser = setup_argument_parser()

    # Gather command-line arguments
    args = parser.parse_args()

    # Verify arguments are acceptable
    verify_args(args)

    return args

# --- Feature calculations ---
def get_degree_feature(graphs, edges):
    # Initialize product of degrees
    degree_products = []

    # For every edge, find product of incident node degrees
    # Repeat for every layer in remnant multiplex
    for graph in graphs:
        degree_products.append(features.degrees.get_degrees(graph, edges))

    # Formulate degree products as k_i^ak_j^a / (k_i^ak_j^a + k_i^bk_j^b)
    feature_degrees = features.formatters.as_configuration(
        *degree_products,
        transform=lambda x: x)

    return feature_degrees

def get_distance_feature(vectors, edges):
    # Initialize vector distances
    vector_distances = []

    for vectors_ in vectors:
        vector_distances.append(features.distances.get_distances(vectors_, edges))

    # Formulate degree products as 1/d_e^a / (1/d_e^a + 1/d_e^b)
    # * Note: 1/d applied automatically as preprocessing step
    # * in `as_configuration`
    # ? Move this ^ (comment) functionality?
    feature_distances = features.formatters.as_configuration(*vector_distances)

    return feature_distances


def calculate_feature_matrix(graphs, edges, vectors):
    # Calculate each feature
    feature_degrees = get_degree_feature(graphs, edges)
    feature_distances = get_distance_feature(vectors, edges)

    # Format as a unified feature matrix
    # ^ sklearn models require single numpy array for features
    X = features.formatters.format_feature_matrix(feature_degrees, feature_distances)

    return X

def get_edge_sets(remnant_multiplex):
    return remnant_multiplex.observed, remnant_multiplex.unobserved

# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Load data
    with open(args.filepath_remnants, 'rb') as _fh:
        remnant_multiplex = pickle.load(_fh)
    with open(args.filepath_embeddings, 'rb') as _fh:
        remnant_embeddings = pickle.load(_fh)

    # Normalize vectors
    if args.normalize:
        logger.info("Normalizing embedded vectors")
        remnant_embeddings[0].normalize(embeddings.helpers.get_components(remnant_multiplex.layers[0].graph))
        remnant_embeddings[1].normalize(embeddings.helpers.get_components(remnant_multiplex.layers[1].graph))
    else:
        logger.warning("*NOT* normalizing embedded vectors!")

    # Retrieve edges -> layers mappings
    training_edges, testing_edges = get_edge_sets(remnant_multiplex)

    # Alias data used to calculate features
    graphs = [
        layer.graph
        for layer in remnant_multiplex.layers
    ]
    print("AHHH", graphs)
    vectors = [
        emb.vectors
        for emb in remnant_embeddings
    ]

    # Calculate training features
    X_train = calculate_feature_matrix(graphs, training_edges.keys(), vectors)
    Y_train = list(training_edges.values())

    # Calculate testing features
    X_test = calculate_feature_matrix(graphs, testing_edges.keys(), vectors)
    Y_test = list(testing_edges.values())

    # Train reconstruction model
    model = classifiers.models.LogReg(
        "LogReg", ("deg", "emb"), dict(),
        X_train, Y_train,
        PARAMS_LOGREG)

    # Save model and testing data to disk
    filepath_output = os.path.join(
            DIR_MODELS,
            FILEPATH_TEMPLATE.format(
                normalize=args.normalize,
                basename=os.path.basename(args.filepath_embeddings))
    )
    with open(filepath_output, 'wb') as _fh:
        pickle.dump((model, X_test, Y_test), _fh, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

