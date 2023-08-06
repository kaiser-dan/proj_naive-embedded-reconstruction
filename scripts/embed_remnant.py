"""CLI to embed a remnant multiplex.
"""
# ================= SET-UP =======================
# --- Standard library ---
import os
import pickle
import argparse

# --- Source code ---
from embmplxrec.data import io
from embmplxrec import embeddings
import embmplxrec.utils

# --- Globals ---
## Filepaths & templates
ROOT = os.path.join("..", "")
DIR_CACHES = os.path.join(ROOT, "data", "interim", "caches", "")
FILEPATH_TEMPLATE = "embedding_method-{method}_dim-{dim}_embrep-{rep}_{basename}"

## Logging
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    filename=f".logs/embed_remnant_{embmplxrec.utils.get_today(time=True)}.log")

# ================ FUNCTIONS ======================
# --- Command-line Interface ---
def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Embed remnant duplex and save to disk.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Positional arguments
    parser.add_argument(
        "filepath",
        type=str,
        help="Filepath of pickled remnant duplex.")
    parser.add_argument(
        "embedding",
        choices=["N2V", "LE", "ISOMAP"],
        help="Embedding method.")
    parser.add_argument(
        "dimensions",
        dest="dim",
        type=int,
        help="Embedding dimension.")

    # Optional arguments, flag
    parser.add_argument(
        "-w", "--walklength", dest="walk_length",
        type=int, default=30,
        help="N2V Walk Length.")
    parser.add_argument(
        "-c", "--cores", dest="num_cores",
        type=int, default=1,
        help="Number of workers (N2V only)."
    )
    parser.add_argument(
        "--repeat", dest="reps",
        type=int, default=1,
        help="Number of embeddings to gather on given remnant duplex.")

    return parser

def verify_args(args):
    # - File input -
    if not os.path.isfile(args.filepath):
        raise FileNotFoundError(f"{args.filepath} not found!")

    # - Numeric parameters -
    _fp_after_N = args.filepath.split("N-")[1]
    N = int(_fp_after_N.split("_")[0])
    if args.dimensions <= 0 or args.dimensions >= N:
        raise ValueError("The number of dimensions must be positive and not exceed N")
    if args.reps <= 0:
        raise ValueError("Number of repetitions must be a positive integer!")
    if args.walk_length <= 0 or args.walk_length >= N:
        raise ValueError("Walk length must be positive and not exceed N!")

    return

def gather_args():
    # Initialize argparse
    parser = setup_argument_parser()

    # Gather command-line arguments
    args = vars(parser.parse_args())

    # Verify arguments are acceptable
    verify_args(args)

    return args


# ================ MAIN ======================
def main():
    # Parse CL arguments
    args = gather_args()

    # Separate directory and basename of input RemnantMultipex
    _, basename = os.path.split(args.filepath)

    # Attempt remnant observation for each repetitions
    for rep in range(1, args.reps+1):
        # Construct filepath
        filepath_ = FILEPATH_TEMPLATE.format(
            method=args.method,
            dim=args.dim,
            rep=rep,
            basename=basename)
        filepath = os.path.join(DIR_CACHES, filepath_)

        # Check if file already exists
        if os.path.isfile(filepath):
            logger.info(f"File '{filepath_}' already exists! Skipping remnant embedding.")
            continue

        # Load RemnantMultiplex
        with open(args.filepath, 'rb') as _fh:
            remnant_multiplex = pickle.load(_fh)

        # Embed remnants
        ## Declare dispatch
        match args.embedding:
            case "N2V":
                embedding_function = embeddings.N2V.N2V
            case "LE":
                embedding_function = embeddings.LE.LE
            case "ISOMAP":
                embedding_function = embeddings.Isomap.Isomap
            # TODO: Fix HOPE embedding
            case "HOPE":
                raise NotImplementedError("HOPE is currently not implemented!")

        ## Set kwargs for N2V/Gensim
        parameters = {"dimensions": args.dim}
        if args.embedding == "N2V":
            parameters.update({
                "walk_length": args.walk_length,
                "workers": args.num_cores,
            })


        ## Apply graph embedding method
        embedded_remnants = []
        for remnant_layer in remnant_multiplex.layers:
            embedding = embedding_function(
                remnant_layer.graph,
                parameters,
                dict())  # TODO: Remove hyperparameter as positional argument

            embedded_remnants.append(embedding)

        # Save to disk
        io.safe_save(embedded_remnants, filepath)


if __name__ == "__main__":
    main()
