"""Common multiplex output utility.
"""
# ============= SET-UP =================
__all__ = ["safe_save", "to_edgelist"]

# --- Imports ---
import os
import pickle
from typing import Any

from . import LOGGER


# =================== FUNCTIONS ===================
def safe_save(object: Any, filepath: str, separator: str = '_'):
    """
    Pickles an object and saves to disk, ensuring no existing files are overwritten.

    Automatically modifies specified filepath to remove name conflicts with existing files.
    Appends the smallest positive integer needed to filename (before extension) to specify a unique file by name.

    Parameters
    ----------
    object : Any
        Pickle-able Python object.
    filepath : str
        Desired output file.
    separator : str, optional
        Character preceding unique identifier, by default _
    """
    # Ensure filename is unique
    if os.path.isfile(filepath):
        filepath = _unique_filename_extension(filepath, separator)

    # Save as pickled object
    with open(filepath, 'wb') as _fh:
        pickle.dump(object, _fh, pickle.HIGHEST_PROTOCOL)

    return

def to_edgelist(multiplex: dict, filehandle: str):
    """
    Saves a multiplex to plain-text edgelist file.

    Assumes standard format `[layer] [src] [tgt]`. Directional edges are supported, however, currently all edge attributes are ignored.

    Parameters
    ----------
    multiplex : dict
        Dictionary with layer ids as keys and networkx graph objects as values.
    filehandle : str
        Desired output file.
    """
    # Open file output stream
    iostream = open(filehandle, 'w')

    # Write each layer to file
    for layer, graph in multiplex.items():
        for edge in graph.edges():
            src, tgt = edge
            line = f"{layer} {src} {tgt}\n"
            iostream.write(line)

    # Close iostream
    iostream.close()

    return


# --- Helpers ---
def _unique_filename_extension(filepath, separator = '_'):
    # Build filename pattern with possible unique identifier
    unique_filepath = "{}" + separator + "{}.{}"
    basename, extension = os.path.splitext(filepath)
    extension = extension[1:]  # remove period so f-string is clearer

    # Search positive integers for smallest value which will
    # yield a unique file by name.
    copy_number = 1
    while os.path.isfile(unique_filepath.format(basename, copy_number, extension)):
        copy_number += 1

    # Append smallest necessary identifier to form unique file
    filepath = unique_filepath.format(basename, copy_number, extension)

    return filepath