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

from embed.embedding import Embedding
from sampling.remnants import Remnant

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
    embeddings : tuple[Embedding, Embedding]
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
    embeddings: tuple[Embedding, Embedding]
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