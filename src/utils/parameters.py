"""Project source code for controlling implicit correlations in synthetic multiplexes.

Throughout, `parameters` are independent variables controlling embedding behavior.
They have a theoretical motivation for accepting different values.
In contrast, `hyperparameters` are independent variables for embedding or regression
that have less convincing theoretical reasons to be altered (with respect to the original analysis).
"""
# ============= SET-UP =================
from numpy import linspace

# ============= FUNCTIONS =================
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
    penalty: dict, default='l2'
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
            "fit_intercept": fit_intercept,   # whether to fit an intercept
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
        maxiter=100,
        tol=-8,
        # LogReg
        penalty="l2",
        fit_intercept=True,
        # Other
        theta_min=0.05,
        theta_max=0.5,
        theta_num=10,
        repeat=5):
    """Prepare parameters for LE-based reconstruction experiments.

    Parameters
    ----------
    dimensions : int, optional
        _description_, by default 128
    maxiter : int, optional
        _description_, by default 100
    tol : int, optional
        _description_, by default -8
    penalty: dict, default='l2'
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
        "embedding": {
            # >>> LE embedding hyperparameters <<<
            "maxiter": maxiter,
            "tol": tol,
            "NCV": 6,
        },
        "classifier": {
            # >>> Logistic regression <<<
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
    return linspace(
        experiment_setup["theta_min"],
        experiment_setup["theta_max"],
        experiment_setup["theta_num"]
    )
