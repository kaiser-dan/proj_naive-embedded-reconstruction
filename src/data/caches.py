"""Project source code for caching commonly used multiplex observational data.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import os
import pickle
from dataclasses import dataclass

# --- Network science ---
import networkx as nx

# --- Project source ---
# PATH adjustments
ROOT = os.path.join(*["..", "..", ""])
SRC = os.path.join(*[ROOT, "src", ""])
sys.path.append(ROOT)
sys.path.append(SRC)

## Embeddings
from embed.embedding import Embedding
from embed import LE, N2V, Isomap

## Remnants
from sampling.remnants import Remnant

## Utils
from utils import parameters as params

# ========== CLASSES ==========
@dataclass
class CachedEmbeddings:
    """Class for storing relevant multiplex observational data for reconstruction experiments.

    Data
    ----
    name : str
        Name of underlying system.
    layers : tuple[int, int]
        Indices of layers of the original system considered in this induced duplex.
    theta : float
        Relative size of training set.
    remnants : tuple[Remnant, Remnant]
        Remnant layers.
    embeddings : list[Embedding, Embedding]
        Embeddings of remnant layers.
    observed_edges : dict[tuple[int, int], int]
        Mapping of training set edges into their true layers, as given by indices of `layers`.
    unobserved_edges : dict[tuple[int, int], int]
        Mapping of test set edges into their true layers, as given by indices of `layers`.

    Methods
    -------
    save(filepath: str)
        Saves object to the given filepath.
    """
    name: str
    layers: tuple[int, int]
    theta: float
    remnants: tuple[Remnant, Remnant]
    embeddings: list[Embedding, Embedding]  # has to be mutable to post-process
    observed_edges: dict[tuple[int, int], int]
    unobserved_edges: dict[tuple[int, int], int]

    # --- Public methods ---
    def save(self, filepath: str):
        save_cached_embedding(self, filepath)


# =================== FUNCTIONS ===================
# --- File I/O ---
def save_cached_embedding(cache: CachedEmbeddings, filepath: str):
    try:
        fh = open(filepath, "wb")
        pickle.dump(cache, fh, pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        sys.stderr.write(f"{err}\n Error serializing CachedEmbedding instance!")
    finally:
        fh.close()

    return

def load_cached_embedding(filepath: str):
    if not os.path.isfile(filepath):
        raise FileNotFoundError

    try:
        fh = open(filepath, "wb")
        cache = pickle.load(fh)
    except Exception as err:
        sys.stderr.write(f"{err}\n Error loading CachedEmbedding instance!")
    finally:
        fh.close()

    return cache


# --- Pseudo-constructors ---
def build_cachedremnants(
        name: str,
        layers: tuple[int, int],
        remnants: tuple[Remnant, Remnant],
        embedder: str = "LE",
        per_component = False,
        **kwargs):
    # Dispatch correct embedding method
    if embedder == "LE":  # TODO: Simplify with regex on "X-PC"
        embedding_function = LE.LE
        parameters, hyperparameters, experiment_setup = params.set_parameters_LE(**kwargs)
    elif embedder == "N2V":
        embedding_function = N2V.N2V
        parameters, hyperparameters, experiment_setup = params.set_parameters_N2V(**kwargs, quiet=False)
    elif embedder == "ISOMAP":
        embedding_function = Isomap.Isomap
        parameters = {"dimension": 128}
        # TODO: Tidy hyperparameter handling
        hyperparameters = {"embedding": dict()}  # to avoid downstream keyerrors
    else:
        raise NotImplementedError(f"Embedder {embedder} not a recognized/implemented graph embedding!")

    # Embed remnants (list of Embedding objects)
    embeddings = [
        embedding_function(
            remnant.remnant,
            parameters=parameters,
            hyperparameters=hyperparameters["embedding"],
            per_component=per_component
        )
        for remnant in remnants
    ]

    # Aggregate training and test sets
    observed_edges = remnants[0].known_edges
    unobserved_edges = remnants[0].unknown_edges
    observed_edges.update(remnants[1].known_edges)
    unobserved_edges.update(remnants[1].unknown_edges)

    # Create CachedEmbedding objects
    cache = CachedEmbeddings(
        name=name,
        layers=layers,
        theta=remnants[0].theta,
        remnants=remnants,
        embeddings=embeddings,
        observed_edges=observed_edges,
        unobserved_edges=unobserved_edges
    )

    return cache

