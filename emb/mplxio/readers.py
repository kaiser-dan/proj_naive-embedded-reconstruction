"""Common multiplex input utility.
"""
# ============= SET-UP =================
__all__ = ['from_edgelist', 'safe_load']

# --- Imports ---
import os
import pickle

import networkx as nx

from loguru import logger as LOGGER

# --- Globals ---
EDGELIST_EXTENSIONS = {
    '.edges', '.edgelist',
    '.multiplex', '.mplx',
    '.rmnt', '.drmnt', '.wrmnt', '.dwrmnt'}

# =================== FUNCTIONS ===================
def from_edgelist(filepath: str, delimiter=None):
    """Read multiplex from edgelist.

    Assumes edge-colored multidimensional graph topology in plain-text file.
    Each line is assumed to be a new edge datum of the format [layer] [src] [tgt] [...],
    where any additional information (e.g. edge weights or labels) are ignored.
    Currently only supports undirected edges.

    Parameters
    ----------
    file_handle : string or FileObject
        Relative file path to edge-list.
    delimiter: string or None, optional
        Layer/node delimiter in edgelist file, by default None (default `split` whitespace).


    Returns
    -------
    dict
        Mapping of layer labels to nx.Graph objects for that layer.
    """
    # Preprocessing checks
    if not _check_exists(filepath):
        raise FileNotFoundError(filepath)
    if os.path.splitext(filepath)[1] not in EDGELIST_EXTENSIONS:
        LOGGER.warning("File does not have recognized extension. See `EDGELIST_EXTENSIONS` for list of recognized edgelist file extensions. Attempting to proceed...")

    # Initialize return struct
    multiplex = {}  # initialize `layer_idx -> nx.Graph` mapping

    # Open file input stream
    iostream = open(filepath, 'r')

    # Process edges into respective layers
    # ^ Done sequentially to avoid excessive memory usage and data races
    for line in iostream:
        data = _process_line(line, delimiter)  # remove whitespace
        layer_idx = int(data[0])
        src_node = int(data[1])
        tgt_node = int(data[2])

        # Add edge to layer graph object
        if layer_idx not in multiplex:
            multiplex[layer_idx] = nx.Graph()  # instantiate layer graph if non-existent
        multiplex[layer_idx].add_edge(src_node, tgt_node)

    # Close input stream
    iostream.close()

    return multiplex

# TODO: Tidy and test!
def safe_load(filepath: str):
    with open(filepath, 'rb') as _fh:
        data = pickle.load(_fh)

    return data


# --- Helpers ---
def _check_exists(filepath):
    return os.path.exists(filepath)

def _process_line(line, delimiter=None):
    return line.strip().split(delimiter)