"""Mathematical functions used to evaluate likelihoods and metric distances."""
# ========== SET-UP ==========
# --- Standard library ---
import sys

# --- Scientific computing ---
import numpy as np

# --- Source code ---
import embmplxrec.utils
logger = embmplxrec.utils.get_module_logger(
    name=__name__,
    file_level=10,
    console_level=30)

# --- Globals ---
TOLERANCE = 1e-10  # absolute tolerance on comparisons to 0

# ========== FUNCTIONS ==========
# --- Likelihoods ---
# Convex models
def identity(x): return x
def inverse(x):
    if not np.isclose(x, 0, atol=TOLERANCE):
        return 1/x
    else:
        logger.warning("Quantity too close to zero, returning np.inf instead.")
        return np.inf

# TODO: Add floating-point comparison safety for small floats
# TODO: See python docs on sys.float_info.epsilon versus sys.float_info.min
def bounded_inverse(x, tolerance=sys.float_info.epsilon):
    # Ensure x has float dtype
    x = x.astype(float)

    # Add machine tolerance to avoid division by zero
    x += tolerance

    # Calculate the multiplicative inverse (with tolerance added)
    return 1 / x

def negexp(x): return np.exp(-x)

# Sigmoid models
def logistic(x): return 1 / (1 + negexp(x))
def tanh(x): return tanh(x)
def arctan(x): return arctan(x)

# --- Vector-space metrics ---
# Euclidean metrics
def euclidean_distance(x, y):
    try:  # attempt norm of difference vector
        distance = np.linalg.norm(x-y)
    except ValueError:
        distance = _handle_mismatched_dims(x, y)
    return distance

def cosine_similarity(x, y):
    try:  # attempt similarity of vectors
        similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
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
        abs_difference = np.absolute(difference)

    # Calculate L_p norm of difference vector
    # & Efficient calculations available for p in {1, 2, infinity}
    if p == 1:
        return sum(abs_difference)
    elif p == 2:
        return euclidean_distance(x, y)
    elif p == np.inf:
        return max(abs_difference)
    else:
        return sum(abs_difference**p)**(1./p)

# Hyperbolic metrics
def poincare_disk_metric(x, y): raise NotImplementedError("Hyperbolic distance not yet implemented!")


# --- Helpers ---
def _handle_mismatched_dims(x, y):
    logger.warning("Attempting to add vectors of different sizes; casting as np.inf instead (for disconnected components)")
    if len(x) != len(y):
        return np.inf
    else:
        raise ValueError("Value error encountered _not_ related to dimension mismatch!")

def scale_probability(p):
    if not (0 <= p and p <= 1):
        logger.error(f"Given probability ({p}) is not in valid domain [0,1]!")
    else:
        return 2*p - 1

def get_labels(edges):
    return list(edges.values())