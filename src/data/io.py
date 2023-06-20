"""Project source code for common multiplex I/O utility.
"""
# ============= SET-UP =================
# --- Standard library ---
import os
import sys
import pickle

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
def save_multiplex(M: dict[int, nx.Graph], filepath: str):
    try:
        fh = open(filepath, "wb")
        pickle.dump(M, fh, pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        sys.stderr.write(f"{err}\n Error saving multiplex!")
    finally:
        fh.close()



# --- Helper ---
def process_filename(filename: str):
    tags = dict()
    tags["keywords"] = set()

    basename = os.path.splitext(os.path.basename(filename))[0]
    split_ = basename.split("_")

    for tag in split_:
        if "-" in tag:
            name, value = tag.split("-")[:2]  # In case there are more than 1 -
            tags[name] = value
        else:
            tags["keywords"].add(tag)

    return tags


if __name__ == "__main__":
    s = "data/output/reconstructions/performance_model_penalty-None_method-N2V_percomponent-False_dim-32_embrep-0_remnants_theta-0.1_strategy-RANDOM_remrep-0_edgelists_name-LFR_N-250_T1-2.1_T2-1.0_kavg-10.0-15_mu-0.1_prob-1.0_rep-1.dat"
    print(process_filename(s))