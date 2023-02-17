"""Project source code for controlling implicit correlations in synthetic multiplexes.
"""
# ============= SET-UP =================
import numpy as np

# ============= FUNCTIONS =================
def set_parameters_N2V(
    dimensions=128,
    walk_length=30,
    num_walks=100,
    workers=8,
    quiet=True,
    window=10,
    min_count=1,
    batch_words=4,
    penalty="l2",
    theta_min=0.05,
    theta_max=0.5,
    theta_num=10,
    repeat=5
        ):
    parameters = {
        # >>> Node2Vec embedding <<<
        "dimensions": dimensions,  # euclidean dimension to embedd
        "walk_length": walk_length,  # number of nodes in each walk
        "num_walks": num_walks,  # number of walks per node
        "workers": workers,  # for cpu parallel work
        "quiet": quiet,  # verbose printing
        # >>> Simulations <<<
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,
    }

    hyperparameters = {
        # >>> Node2Vec embedding <<<
        "window": window,  # maximum distance between the current and predicted word within a sentence.
        "min_count": min_count,  # ignores all words with total frequency lower than this
        "batch_words": batch_words,  # [unsure]

        # >>> Logistic regression <<<
        "penalty": penalty,  # L2 regularization

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters, hyperparameters

def set_parameters_LE(
    dimensions=128,
    maxiter=100,
    tol=-8,
    penalty="l2",
    theta_min=0.05,
    theta_max=0.5,
    theta_num=10,
    repeat=5
        ):
    parameters = {
        # >>> LE embedding <<<
        "k": dimensions,  # needs k for scipy.sparse.linalg.eigsh

        # >>> Simulations <<<
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_num": theta_num,
    }

    hyperparameters = {
        # >>> LE embedding <<<
        "maxiter": maxiter,
        "tol": tol,
        "NCV": 6,

        # >>> Logistic regression <<<
        "penalty": penalty,  # L2 regularization

        # >>> Other <<<
        "repeat": repeat  # number of simulations
    }

    return parameters, hyperparameters


def build_theta_range(min, max, num):
    return np.linspace(min, max, num)
