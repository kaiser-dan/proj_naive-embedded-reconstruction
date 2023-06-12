"""Depreciated project source code.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Project source ---
sys.path.append("./")
from distance.distance import embedded_edge_distance, component_penalized_embedded_edge_distance


# ========== FUNCTIONS ==========
def get_distances(vectors, edges):
    # >>> Book-keeping >>>
    G, H = vectors  # alias input layer embedded node vectors
    # <<< Book-keeping <<<

    # >>> Distance calculations >>>
    G_distances = [embedded_edge_distance(edge, G) for edge in edges]
    H_distances = [embedded_edge_distance(edge, H) for edge in edges]
    # <<< Distance calculations

    return G_distances, H_distances

def get_biased_distances(vectors, edges, components):
    # >>> Book-keeping >>>
    G, H = vectors  # alias input layer embedded node vectors
    G_components, H_components = components  # alias component mapping of remnants
    # <<< Book-keeping <<<

    # >>> Distance calculations >>>
    G_distances = [component_penalized_embedded_edge_distance(edge, G, G_components) for edge in edges]
    H_distances = [component_penalized_embedded_edge_distance(edge, H, H_components) for edge in edges]
    # <<< Distance calculations

    return G_distances, H_distances

def prepare_feature_matrix_confdeg_dist(configuration_degrees, distances):
    # >>> Book-keeping >>>
    N = len(configuration_degrees)
    NUM_FEATURES = 3

    G_distances, H_distances = distances

    feature_matrix = np.empty((N, NUM_FEATURES))
    # <<< Book-keeping <<<

    # >>> Format feature matrix >>>
    feature_matrix[:, 0] = configuration_degrees
    feature_matrix[:, 1] = G_distances
    feature_matrix[:, 2] = H_distances
    # <<< Format feature matrix <<<

    return feature_matrix


def embedded_edge_distance_ratio(
    edge, vectors_numerator, vectors_denominator,
        distance_=euclidean_distance):
    # Calculate edge distance in each remnant
    dist_numerator = embedded_edge_distance(edge, vectors_numerator, distance_)
    dist_denominator = embedded_edge_distance(edge, vectors_denominator, distance_)

    # Calculate distance ratio
    try:
        ratio = dist_numerator / dist_denominator
    except ZeroDivisionError:
        ratio = np.finfo(np.float64).max
    finally:
        return ratio


def component_penalized_embedded_edge_distance_ratio(
    edge,
    graph_numerator, graph_denominator,
    vectors_numerator, vectors_denominator,
        penalty=2**8, distance_=euclidean_distance):
    # >>> Book-keeping >>>
    src, tgt = edge  # identify nodes incident to edge
    # <<< Book-keeping <<<

    # >>> Score (feature) calculation >>>
    # Calculate edge distance in each remnant
    dist_numerator = component_penalized_embedded_edge_distance(edge, graph_numerator, vectors_numerator, penalty, distance_)
    dist_denominator = component_penalized_embedded_edge_distance(edge, graph_denominator, vectors_denominator, penalty, distance_)

    # Account for isolated component bias
    component_numerator_src = component(graph_numerator, src)
    component_numerator_tgt = component(graph_numerator, tgt)
    component_denominator_src = component(graph_denominator, src)
    component_denominator_tgt = component(graph_denominator, tgt)
    if (component_numerator_src == component_numerator_tgt) and (len(component_numerator_src) <= 15):
        dist_numerator += penalty
    if (component_denominator_src == component_denominator_tgt) and (len(component_denominator_src) <= 15):
        dist_denominator += penalty

    # Calculate distance ratio
    try:
        ratio = dist_numerator / dist_denominator
    except ZeroDivisionError:
        ratio = np.finfo(np.float64).max
    # <<< Score (feature) calculation <<<
    finally:
        return ratio


def format_distance_ratios(X):
    # Apply logarithmic transform to regularize division space
    X = np.log(X)

    # Remove NaNs for sklearn model
    X = np.nan_to_num(X, nan=-1e-32)

    # Shape features for sklearn model
    X = X.reshape(-1, 1)

    return X

def _build_remnants(G1, G2, Etrain, Etest):
    # Remnants
    rem_G1 = nx.Graph()
    rem_G2 = nx.Graph()
    for n in G1:
        rem_G1.add_node(n)
        rem_G2.add_node(n)
    for n in G2:
        rem_G1.add_node(n)
        rem_G2.add_node(n)

    # Add aggregate edges we don't know about
    for e in Etest:
        rem_G1.add_edge(e[0], e[1])
        rem_G2.add_edge(e[0], e[1])

    # Add to remnant alpha the things known to be in alpha
    # So remnant alpha is unknown + known(alpha)
    for e in Etrain:
        if Etrain[e] == 1:
            rem_G1.add_edge(e[0], e[1])
        if Etrain[e] == 0:
            rem_G2.add_edge(e[0], e[1])

    return rem_G1, rem_G2, Etest

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

def format_feature_matrix_statsmodels(
        feature_set, M_train, M_test,
        observed_edges, unobserved_edges,
        feature_distances_train=None,
        feature_distances_test=None,
        feature_degrees_train=None,
        feature_degrees_test=None,
):
    if feature_set == {"imb"}:
        feature_matrix_train = np.array([0]*M_train).reshape(-1,1)
        feature_matrix_test = np.array([0]*M_test).reshape(-1,1)
    elif feature_set == {"emb_c"} or feature_set == {"emb_r"} or feature_set == {"emb"}:
        feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
    elif feature_set == {"deg"}:
        feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif feature_set == {"imb", "emb_c"} or feature_set == {"imb", "emb_r"} or feature_set == {"imb", "emb"}:
        feature_matrix_train = np.array(feature_distances_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_distances_test).reshape(-1,1)
    elif feature_set == {"imb", "deg"}:
        feature_matrix_train = np.array(feature_degrees_train).reshape(-1,1)
        feature_matrix_test = np.array(feature_degrees_test).reshape(-1,1)
    elif feature_set == {"emb_c", "deg"} or feature_set == {"emb_c", "deg", "imb"} or \
            feature_set == {"emb_r", "deg"} or feature_set == {"emb_r", "deg", "imb"} or \
            feature_set == {"emb", "deg"} or feature_set == {"emb", "deg", "imb"}:
        feature_matrix_train = np.empty((M_train, 2))
        feature_matrix_train[:, 0] = feature_distances_train
        feature_matrix_train[:, 1] = feature_degrees_train

        feature_matrix_test = np.empty((M_test, 2))
        feature_matrix_test[:, 0] = feature_distances_test
        feature_matrix_test[:, 1] = feature_degrees_test

    labels_train, labels_test = get_labels(
        observed_edges, unobserved_edges
    )

    # ^ Convert feature matrix to dataframe
    if "emb" in feature_set and "deg" in feature_set:
        df = pd.DataFrame({
            "label": labels_train,
            "distance": feature_matrix_train[:, 0],
            "degree": feature_matrix_train[:, 1]
        })
        df_test = pd.DataFrame({
            "label": labels_test,
            "distance": feature_matrix_test[:, 0],
            "degree": feature_matrix_test[:, 1]
        })
        y, X = dmatrices("label ~ distance + degree", data=df, return_type="dataframe")
        _, X_test = dmatrices("label ~ distance + degree", data=df_test, return_type="dataframe")
    elif "emb" in feature_set and "deg" not in feature_set:
        df = pd.DataFrame({
            "label": labels_train,
            "distance": feature_matrix_train[:, 0],
        })
        df_test = pd.DataFrame({
            "label": labels_test,
            "distance": feature_matrix_test[:, 0],
        })
        y, X = dmatrices("label ~ distance", data=df, return_type="dataframe")
        _, X_test = dmatrices("label ~ distance", data=df_test, return_type="dataframe")
    elif "emb" not in feature_set and "deg" in feature_set:
        df = pd.DataFrame({
            "label": labels_train,
            "degree": feature_matrix_train[:, 0],
        })
        df_test = pd.DataFrame({
            "label": labels_test,
            "degree": feature_matrix_test[:, 0]
        })
        y, X = dmatrices("label ~ degree", data=df, return_type="dataframe")
        _, X_test = dmatrices("label ~ degree", data=df_test, return_type="dataframe")

    return y, X, X_test

def get_distance_ratios_feature(distances_G, distances_H, zde_penalty=1e-12):
    # >>> Book-keeping >>>
    M = len(distances_G)  # get number of observations in dataset
    distance_ratios = []  # initialize feature set
    # <<< Book-keeping <<<

    # >>> Calculate distance ratios >>>
    for idx in range(M):
        s_G = 1 / (distances_G[idx] + zde_penalty)
        s_H = 1 / (distances_H[idx] + zde_penalty)
        probability = s_G / s_H

        distance_ratios.append(probability)
    # <<< Calculate distance ratios <<<

    return distance_ratios