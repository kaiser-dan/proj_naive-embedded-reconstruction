import sys
import os

from emb import mplxio
from emb import embeddings


def _parse_args(args):
    """Assumes input {input} {embedding} {output}.

    where:
    - {input} is filepath for remnant multiplex edgelist
    - {embedding} is a graph embedding method.
    - {output} is filepath for output remnant embedding vectors. Must contain "{embedding}" for filepath completion.
    """
    # Ensure input file exists
    if not os.path.exists(args[0]):
        raise FileNotFoundError(args[0])

    # Ensure embedding is supported
    if args[1] not in embeddings.ACCEPTED_EMBEDDINGS:
        raise NotImplementedError(
            f"'{args[1]}' not an accepted embedding. Choices: {embeddings.ACCEPTED_EMBEDDINGS}"
        )

    # Ensure directory path of output file exists
    if not os.path.exists(os.path.dirname(args[2])):
        os.system(f"mkdir {os.path.dirname(args[2])}")

    return args


def main(filepath_input, embedding, filepath_output):
    # Bring remnant multiplex into memory
    remnant_multiplex = mplxio.from_edgelist(filepath_input)

    # Embed with selected embedding method
    match embedding:
        case "N2V":
            vectors = embeddings.embed_multiplex_N2V(remnant_multiplex, dimensions=128)
        case "LE":
            vectors = embeddings.embed_multiplex_LE(remnant_multiplex, k=128)
        case "Isomap":
            vectors = embeddings.embed_multiplex_Isomap(
                remnant_multiplex, dimensions=128
            )
        case "HOPE":
            vectors = embeddings.embed_multiplex_HOPE(remnant_multiplex, dimensions=128)
        case _:
            raise NotImplementedError(embedding)

    mplxio.safe_save(vectors, filepath_output)

    return


if __name__ == "__main__":
    # Check and parse args
    args = _parse_args(sys.argv[1:])

    main(*args)
