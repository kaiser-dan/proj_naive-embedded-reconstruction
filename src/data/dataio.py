"""Project source code for common multiplex I/O utility.
"""
# ============= SET-UP =================
# --- Standard library ---
from datetime import datetime  # date metadata

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

def get_input_filehandle(
        SYSTEM,
        ROOT="../../",
        DIR="data/input/raw/",
        PREFACE="duplex",
        POSTFIX=".edgelist"):
    return f"{ROOT}{DIR}{PREFACE}_system={SYSTEM}{POSTFIX}"


# --- Output ---
def save_df(dataframe, output_filehandle):
    dataframe.to_csv(output_filehandle)
    return

def get_output_filehandle(
        PROJECT_ID, RESEARCHERS="DK",
        CURRENT_VERSION="v1.0", DATE=None,
        ROOT="../../",
        DIR="results/dataframes/",
        PREFACE="dataframe",
        POSTFIX=".csv"):
    # >>> Formatting metadata >>>
    # Formatting standard date
    if DATE is None:
        DATE = datetime.today().strftime("%Y%m%d")

    # Experiment tag
    TAG = f"{PROJECT_ID}{CURRENT_VERSION}_{RESEARCHERS}_{DATE}"

    # Fill in output filehandle
    output_filehandle = f"{ROOT}{DIR}{PREFACE}_{TAG}{POSTFIX}"
    # <<< Formatting metadata <<<

    return output_filehandle, TAG
