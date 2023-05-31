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

# ========== CLASSES ==========
@dataclass
class CachedEmbeddings:
    """Class for storing relevant multiplex observational data for reconstruction experiments."""
    system: str
    layers: tuple[int, int]
    theta: float
    remnants: tuple[nx.Graph, nx.Graph]
    embeddings: tuple[Embedding, Embedding]
    observed_edges: dict[tuple[int, int], int]
    unobserved_edges: dict[tuple[int, int], int]


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