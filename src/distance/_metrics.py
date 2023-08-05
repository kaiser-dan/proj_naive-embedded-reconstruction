import sys
import os

from numpy import Inf, absolute, dot
from numpy.linalg import norm

sys.path.append(os.path.join("..", "utils", ""))
import utils

logger = utils.get_module_logger(name=__name__, file_level=0, console_level=30)

# >>> Helpers >>>
def _handle_mismatched_dims(x, y):
    logger.warning("Attempting to add vectors of different sizes; casting as Infinity instead (for disconnected components)")
    if len(x) != len(y):
        return Inf
    else:
        raise ValueError("Value error encountered _not_ related to dimension mismatch!")
# <<< Helpers <<<

# >>> Euclidean metrics >>>
def euclidean_distance(x, y):
    try:  # attempt norm of difference vector
        distance = norm(x-y)
    except ValueError:
        distance = _handle_mismatched_dims(x, y)
    return distance

def cosine_similarity(x, y):
    try:  # attempt similarity of vectors
        similarity = dot(x, y) / (norm(x) * norm(y))
    except ValueError:
        similarity = _handle_mismatched_dims(x, y)
    return similarity

def L_norm(x, y, p=2):
    # Calculate difference vector
    try:
        difference = x - y
    except ValueError:
        difference = _handle_mismatched_dims(x, y)
    finally:
        abs_difference = absolute(difference)

    # Calculate L_p norm of difference vector
    # & Efficient calculations available for p in {1, 2, infinity}
    if p == 1:
        return sum(abs_difference)
    elif p == 2:
        return euclidean_distance(x, y)
    elif p == Inf:
        return max(abs_difference)
    else:
        return sum(abs_difference**p)**(1./p)
# <<< Euclidean metrics <<<

# >>> Hyperbolic metrics >>>
def poincare_disk_metric(x, y): raise NotImplementedError("Hyperbolic distance not yet implemented!")

# <<< Hyperbolic metrics <<<