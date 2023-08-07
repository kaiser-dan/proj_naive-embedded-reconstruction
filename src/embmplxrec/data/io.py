"""Project source code for common multiplex I/O utility.
"""
# ============= SET-UP =================
# --- Standard library ---
import os
import pickle
from typing import Any

# --- Network science ---
import networkx as nx


# =================== FUNCTIONS ===================
# --- Input ---
def read_file(file_handle, delimiter=None):
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
    # >>> Book-keeping >>>
    multiplex = {}  # initialize `layer_idx -> nx.Graph` mapping
    def process_line(line): return line.strip().split()
    if delimiter is not None:
        def process_line(line): return line.strip().split(delimiter)
    # <<< Book-keeping <<<

    # >>> Reading multiplex from disk >>>
    # Open file I/O handler
    with open(file_handle) as raw_edge_data:
        # Process each line individually
        # ^ Done sequentially to avoid excessive memory usage
        for line in raw_edge_data:
            data = process_line(line)  # remove whitespace
            layer_idx = int(data[0])
            src_node = int(data[1])
            tgt_node = int(data[2])

            # Add edge to layer graph object
            if layer_idx not in multiplex:
                multiplex[layer_idx] = nx.Graph()  # instantiate layer graph if non-existent
            multiplex[layer_idx].add_edge(src_node, tgt_node)
    # <<< Reading multiplex from disk <<<

    return multiplex

# --- Output ---
def safe_save(object: Any, filepath: str):
    # Ensure filename is unique
    if os.path.isfile(filepath):
        filepath = unique_filename_extension(filepath)

    # Save as pickled object
    with open(filepath, 'wb') as _fh:
        pickle.dump(object, _fh, pickle.HIGHEST_PROTOCOL)

def unique_filename_extension(filepath: str, delimiter='_') -> str:
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]  # remove period so f-string is clearer

    # Append smallest necessary identifier to form unique file
    copy_number = 1
    unique_filepath = "{}" + delimiter + "{}.{}"
    while os.path.isfile(unique_filepath.format(basename, copy_number, extension)):
        copy_number += 1
    filepath = unique_filepath.format(basename, copy_number, extension)

    return filepath