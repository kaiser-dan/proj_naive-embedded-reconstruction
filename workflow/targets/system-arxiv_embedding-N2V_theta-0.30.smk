import numpy as np

_EMBEDDINGS = ["N2V"]
_THETAS = ["0.30"]
_REPS = np.arange(1,5)
_LAYERS = ["6-7"]

TARGETS_ = expand(
        DIR_MODELS+"model-imbalanced_embed-N2V_remnants_theta-{theta}_rep-{rep}_clean-multiplex-arxiv_{layerpair}.model",
        theta=_THETAS,
        rep=_REPS,
        layerpair=_LAYERS
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_
