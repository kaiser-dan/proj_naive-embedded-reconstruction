import sys
import os

import numpy as np
import networkx as nx

from emb import mplxio
from emb import netsci
from emb import embeddings
from emb import classifiers
from emb import utils

from emb.remnants.observer import AGGREGATE_LABEL

from emb import logger as LOGGER

def _parse_args(args):
    """Assumes input {input_rmnt} {input_emb} {output}.

    where:
    - {input_gt} is the filepath for the original nonoverlapping multiplex.
    - {input_rmnt} is filepath for remnant multiplex.
    - {input_emb} is filepath for embedded remnant multiplex.
    - {output} is filepath for output model filepath.
    """
    # Ensure input file exists
    if not os.path.exists(args[0]):
        raise FileNotFoundError(args[0])
    if not os.path.exists(args[1]):
        raise FileNotFoundError(args[1])
    if not os.path.exists(args[2]):
        raise FileNotFoundError(args[2])

    # Ensure directory path of output file exists
    if not os.path.exists(os.path.dirname(args[3])):
        os.system(f"mkdir {os.path.dirname(args[3])}")

    return args


def main(filepath_input_gt, filepath_input_rmnt, filepath_input_emb, filepath_output):
    # Bring inputs into memory
    # ? Apply disjoint processing here?
    gt_multiplex = mplxio.from_edgelist(filepath_input_gt)  # label -> nx.Graph
    remnant_multiplex = mplxio.from_edgelist(filepath_input_rmnt)  # label -> nx.Graph
    vectors = mplxio.safe_load(filepath_input_emb)  # label -> [node -> array]

    # Retrieve edgesets (input datum)
    testing_edges = remnant_multiplex[AGGREGATE_LABEL].edges()
    training_edges = netsci.all_edges(*remnant_multiplex.values()) - set(testing_edges)

    # Retrieve edge's layers (class labels)
    y_train = classifiers.features.get_edge_to_layer(
        training_edges, gt_multiplex.values()
    )
    y_test = classifiers.features.get_edge_to_layer(
        testing_edges, gt_multiplex.values()
    )

    # Calculate feature sets
    # Normalize embeddings
    remnant_multiplex = utils.cutkey(remnant_multiplex, AGGREGATE_LABEL)
    components = {
        label: sorted(nx.connected_components(graph))
        for label, graph in remnant_multiplex.items()
    }
    vectors = utils.cutkey(vectors, AGGREGATE_LABEL)

    for layer in remnant_multiplex.keys():
        LOGGER.debug(f"--- Layer: {layer} ---")
        LOGGER.debug(f"|Components| = {[len(comp) for comp in components[layer]]}")
        LOGGER.debug(f"|Vectors| = {len(vectors[layer])}")

    for label, vectorset in vectors.items():
        LOGGER.info(f"Normalizing layer {label}...")

        node2id = dict()  # ? is this necessary?

        if "LE" in filepath_input_emb:
            LOGGER.debug('LE method detected - checking for non-compatible prior formatting...')
            if vectorset is not dict:
                LOGGER.debug('Incompatible format detected! Attempting conversion...')
                vectors[layer] = dict(enumerate(vectorset))
                node2id = {
                    node: new_node
                    for new_node, node in enumerate(remnant_multiplex[label].nodes())
                }
                # LOGGER.debug(f"New vectors are: {vectors}")

        vectors[label] = embeddings.normalize_vectors(
            vectors[layer], components[label], node2id=node2id
        )

    # Distances
    training_distances = classifiers.get_distances_feature(
        vectors.values(), training_edges, training=True
    )
    testing_distances = classifiers.get_distances_feature(
        vectors.values(), testing_edges, training=False
    )

    # Degrees
    training_degrees = classifiers.get_degrees_feature(
        remnant_multiplex.values(), training_edges, training=True
    )
    testing_degrees = classifiers.get_degrees_feature(
        remnant_multiplex.values(), testing_edges, training=False
    )

    # Formatting as 2xM matrices
    X_train = np.array([training_distances, training_degrees]).transpose()
    X_test = np.array([testing_distances, testing_degrees]).transpose()

    # Train model
    model = classifiers.train_model(X_train, list(y_train.values()))

    # Evaluate model
    accuracy, auroc, pr = classifiers.evaluate_model(
        model, X_test, list(y_test.values())
    )

    # Retrieve identifying information from filename
    getval = lambda part: part.split("-")[1]
    parts = filepath_input_emb.split("_")

    # Indeitifiers shared by LFR or real
    for part in parts:
        if "embedding" in part:
            embedding = getval(part)
        elif "theta" in part:
            theta = getval(part)
        elif "multiplex" in part:
            if "clean" in part:
                system = part.split("-")[2]
            else:
                system = part.split("-")[1]

    # LFR identifiers
    if "LFR" in filepath_input_emb:
        for part in parts:
            if "N" in part:
                N = getval(part)
            if "mu" in part:
                mu = getval(part)
            elif "T1" in part:
                t1 = getval(part)
            elif "T2" in part:
                t2 = getval(part)
            elif "prob" in part:
                prob = getval(part).split(".")[0]

        record_str = ""
        record_str += f"{system},{N},{theta},"
        record_str += f"{embedding},{mu},{t1},{t2},{prob},"
        record_str += f"{accuracy},{auroc},{pr},"
        record_str += f"{model.intercept_[0]},"
        record_str += f"{model.coef_[0][0]},{model.coef_[0][1]}"
        print(record_str)

    # Real identifiers
    else:
        for part in parts:
            if "l1" in part:
                l1 = getval(part)
            elif "l2" in part:
                l2 = getval(part).split(".")[0]

        record_str = ""
        record_str += f"{system},{l1}-{l2},{theta},"
        record_str += f"{embedding},"
        record_str += f"{accuracy},{auroc},{pr},"
        record_str += f"{model.intercept_[0]},"
        record_str += f"{model.coef_[0][0]},{model.coef_[0][1]}"
        print(record_str)


if __name__ == "__main__":
    # Check and parse args
    args = _parse_args(sys.argv[1:])

    main(*args)
