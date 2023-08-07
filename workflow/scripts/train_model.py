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

DIR_MODELS = os.path.join("data", "interim", "models", "")
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
    # parser.add_argument(
    #     "filepath_edgelists",
    #     type=str,
    #     help="Filepath of pickled ground truth duplex.")
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
# - Training -
def get_training_features(remnant_multiplex, remnant_embeddings):
    # Calculate model features
    ## Degree product
    degree_products = [
        features.degrees.get_degrees(
            remnant_layer.graph,
            remnant_multiplex.observed)
        for remnant_layer in remnant_multiplex.layers
    ]
    feature_degrees = features.formatters.as_configuration(
        *degree_products,
        transform=lambda x: x)

    ## Vector distances
    vector_distances = [
        features.distances.get_distances(
            embedding.vectors,
            remnant_multiplex.observed)
        for embedding in remnant_embeddings
    ]
    feature_distances = features.formatters.as_configuration(*vector_distances)

    # Format feature matrix
    X_train = features.formatters.format_feature_matrix(feature_degrees, feature_distances)

    return X_train

# TODO: Add to source code, generalize beyond two layers
def get_training_labels(remnant_multiplex):
    # Retrieve training labels
    Y_train = []
    for edge in remnant_multiplex.observed:
        layer = 0
        if edge in remnant_multiplex.layers[-1].observed:
            layer = 1
        Y_train.append(layer)

    return Y_train

# - Testing -
def get_testing_features(remnant_multiplex, remnant_embeddings):
    # Calculate model features
    ## Degree product
    degree_products = [
        features.degrees.get_degrees(
            remnant_layer.graph,
            remnant_multiplex.unobserved)
        for remnant_layer in remnant_multiplex.layers
    ]
    feature_degrees = features.formatters.as_configuration(
        *degree_products,
        transform=lambda x: x)

    ## Vector distances
    vector_distances = [
        features.distances.get_distances(
            embedding.vectors,
            remnant_multiplex.unobserved)
        for embedding in remnant_embeddings
    ]
    feature_distances = features.formatters.as_configuration(*vector_distances)

    # Format feature matrix
    X_test = features.formatters.format_feature_matrix(feature_degrees, feature_distances)

    return X_test

# TODO: Add to source code, generalize beyond two layers
def get_testing_labels(remnant_multiplex, filepath_remnants):
    # Get ground truth
    edgelists_fp = f"{DIR_EDGELISTS}/multiplex-{filepath_remnants.split('multiplex-')[1]}"
    with open(edgelists_fp, 'rb') as _fh:
        edgelists = pickle.load(_fh)

    # Retrieve testing labels
    Y_test = []
    for edge in remnant_multiplex.unobserved:
        layer = 0
        if edgelists.layers[-1].graph.has_edge(*edge):
            layer = 1
        Y_test.append(layer)

    return Y_test


# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Load data
    with open(args.filepath_remnants, 'rb') as _fh:
        remnant_multiplex = pickle.load(_fh)
    with open(args.filepath_embeddings, 'rb') as _fh:
        remnant_embeddings = pickle.load(_fh)

    # & >>> Debug >>>
    logger.debug(f"Examining first remnant layer...")
    layer_ = remnant_multiplex.layers[0]
    embedding_ = remnant_embeddings[0]
    logger.debug(f"Number of nodes: {layer_.graph.number_of_nodes()}")
    logger.debug(f"Minimum node index: {min(layer_.graph.nodes())}")
    logger.debug(f"Minimum vector index: {min(embedding_.vectors.keys())}")
    logger.debug(f"0 in nodes? {0 in layer_.graph}")
    logger.debug(f"0 in vectors? {0 in embedding_.vectors}")

    logger.debug(f"Examining second remnant layer...")
    layer_ = remnant_multiplex.layers[1]
    embedding_ = remnant_embeddings[1]
    logger.debug(f"Number of nodes: {layer_.graph.number_of_nodes()}")
    logger.debug(f"Minimum node index: {min(layer_.graph.nodes())}")
    logger.debug(f"Minimum vector index: {min(embedding_.vectors.keys())}")
    logger.debug(f"0 in nodes? {0 in layer_.graph}")
    logger.debug(f"0 in vectors? {0 in embedding_.vectors}")

    logger.debug(f"Type of vectors: {type(list(embedding_.vectors.keys())[0])}")
    # & <<< Debug <<<

    # Normalize vectors
    if args.normalize:
        logger.info("Normalizing embedded vectors")
        remnant_embeddings[0].normalize(embeddings.helpers.get_components(remnant_multiplex.layers[0].graph))
        remnant_embeddings[1].normalize(embeddings.helpers.get_components(remnant_multiplex.layers[1].graph))
    else:
        logger.warning("*NOT* normalizing embedded vectors!")


    # Calculate training features
    X_train = get_training_features(remnant_multiplex, remnant_embeddings)
    Y_train = get_training_labels(remnant_multiplex)

    # Calculate testing features
    X_test = get_testing_features(remnant_multiplex, remnant_embeddings)
    Y_test = get_testing_labels(remnant_multiplex, os.path.basename(args.filepath_remnants))

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

