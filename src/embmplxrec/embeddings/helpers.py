"""Helpers for dealing with graph embeddings and their outputs.
"""
# ============= SET-UP =================
# --- Scientific computing ---
import numpy as np

# --- Network science ---
import networkx as nx

# ============= FUNCTIONS =================
# --- Dealing with node labels ---
def reindex_nodes(graph):
    """Reindex graph nodes to contiguous range [0, N-1].

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    dict
        node label -> new node label
    """
    reindexed_nodes = {
        index: new_index
        for new_index, index in enumerate(sorted(graph.nodes()))
    }
    return reindexed_nodes


def get_contiguous_vectors(model):
    """Get vectors from graph embbedding accounting for potential non-contiguous node ids.

    Parameters
    ----------
    model : word2vec model

    Returns
    -------
    dict
        node label -> embedded vector
    """
    # Retrieve 'raw' vectors
    vectors = model.vectors

    # Retrieve word2vec internal hash of node ids to vector indices
    node_labels = model.index_to_key

    # Map node ids into corresponding vector
    # This accounts for graphs with non-consecutive node ids
    embedding = {
        int(node_label): vectors[node_index]
        for node_index, node_label in enumerate(node_labels)
    }

    return embedding

# --- Converting data structs ---
def dict_to_matrix(D):
    # * Assumes contiguous keys from 0
    num_rows = len(D)
    num_cols = len(D[0])

    M = np.empty((num_rows, num_cols))

    for row_idx, row in D.items():
        M[row_idx] = row

    return M

def matrix_to_dict(M):
    D = {
        row_idx: row
        for row_idx, row in enumerate(M)
    }
    return D

# --- Dealing with components ---
def get_components(graph):
    return [
        graph.subgraph(component).copy()
        for component in nx.connected_components(graph)
    ]

# --- Dealing with parameters ---
def set_parameters_N2V(
        # N2V
        dimensions=128,
        walk_length=30,
        num_walks=100,
        workers=8,
        quiet=True,
        window=10,
        min_count=1,
        batch_words=4,
        # LogReg
        penalty="l2",
        fit_intercept=True,
        solver="lbfgs",
        class_weight=None,
        # Other
        theta_min=0.05,
        theta_max=0.5,
        theta_num=10,
        repeat=5):
    """Prepare parameters for N2V-based reconstruction experiments.

    Parameters
    ----------
    dimensions : int, optional
        _description_, by default 128
    walk_length : int, optional
        _description_, by default 30
    num_walks : int, optional
        _description_, by default 100
    workers : int, optional
        _description_, by default 8
    quiet : bool, optional
        _description_, by default True
    window : int, optional
        _description_, by default 10
    min_count : int, optional
        _description_, by default 1
    batch_words : int, optional
        _description_, by default 4
    penalty: str, default='l2'
        Specify the norm of the penalty:
        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.
        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.
    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function, by default True
    solver: str, default='lbfgs'
        Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
            - For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
            - For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
            - ‘liblinear’ is limited to one-versus-rest schemes.
            - ‘newton-cholesky’ is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories. Note that it is limited to binary classification and the one-versus-rest reduction for multiclass classification. Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.
    class_weight: ! FILL IN LATER
    theta_min : float, optional
        Minimum relative size of training set (inclusive), by default 0.05
    theta_max : float, optional
        Maximum relative size of training set (inclusive), by default 0.5
    theta_num : int, optional
        Number of theta values to sample, by default 10
    repeat : int, optional
        Number of repetitions to do for each parameter vertex, by default 5

    Returns
    -------
    parameters : dict
        Parameters for N2V embedding
    hyperparameters : dict
        Hyperparameters for N2V embedding
    experiment_setup : dict
        Specifications for experimental workflow.
    """
    # >>> Node2Vec embedding parameters <<<
    parameters = {
        "dimensions": dimensions,  # euclidean dimension to embedd
        "walk_length": walk_length,  # number of nodes in each walk
        "num_walks": num_walks,  # number of walks per node
        "workers": workers,  # for cpu parallel work
        "quiet": quiet,  # verbose printing
    }

    hyperparameters = {
        "embedding": {
            # >>> Node2Vec embedding hyperparameters <<<
            "window": window,  # maximum distance between the current and predicted word within a sentence.
            "min_count": min_count,  # ignores all words with total frequency lower than this
            "batch_words": batch_words,  # [unsure]
        },
        "classifier": {
            # >>> Logistic regression <<<
            "penalty": penalty,  # L2 regularization
            "fit_intercept": fit_intercept,   # whether to fit an intercept,
            "solver": solver,
            "class_weight": class_weight
        }
    }

    # >>> Simulations <<<
    experiment_setup = {
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters, hyperparameters, experiment_setup

def set_parameters_LE(
        # LE
        dimensions=128,
        per_component=False,
        which="SM",
        maxiter=100000,
        tol=-4,
        # LogReg
        penalty=None,
        fit_intercept=True,
        # Other
        theta_min=0.05,
        theta_max=0.95,
        theta_num=11,
        repeat=5):
    """Prepare parameters for LE-based reconstruction experiments.

    Parameters
    ----------
    dimensions : int, optional
        _description_, by default 128
    which: str, optional,
        _description_, by default "SM"
    maxiter : int, optional
        _description_, by default 1000
    tol : int, optional
        _description_, by default -8
    penalty: dict, default='None'
        Specify the norm of the penalty:
        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.
        .. warning::
           Some penalties may not work with some solvers. See the parameter
           `solver` below, to know the compatibility between the penalty and
           solver.
    fit_intercept : bool, optional
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function, by default True
    theta_min : float, optional
        Minimum relative size of training set (inclusive), by default 0.05
    theta_max : float, optional
        Maximum relative size of training set (inclusive), by default 0.5
    theta_num : int, optional
        Number of theta values to sample, by default 10
    repeat : int, optional
        Number of repetitions to do for each parameter vertex, by default 5


    Returns
    -------
    parameters : dict
        Parameters for LE embedding
    hyperparameters : dict
        Hyperparameters for LE embedding
    experiment_setup : dict
        Specifications for experimental workflow.
    """
    # >>> LE embedding parameters <<<
    parameters = {
        "k": dimensions,  # needs k for scipy.sparse.linalg.eigsh
    }

    hyperparameters = {
        # >>> LE embedding hyperparameters <<<
        "embedding": {
            "which": which,
            "maxiter": maxiter,
            "tol": tol,
            "ncv": 6,  # ! Don't touch
        },
        # >>> Logistic regression <<<
        "classifier": {
            "penalty": penalty,  # L2 regularization
            "fit_intercept": fit_intercept,   # whether to fit an intercept
            "random_seed": 37,  # random seed
        }
    }


    # >>> Simulations <<<
    experiment_setup = {
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters, hyperparameters, experiment_setup


# --- Helpers ---
def build_theta_range(experiment_setup):
    return np.linspace(
        experiment_setup["theta_min"],
        experiment_setup["theta_max"],
        num=experiment_setup["theta_num"],
        endpoint=True
    )
