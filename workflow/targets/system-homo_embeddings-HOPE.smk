import numpy as np

_EMBEDDINGS = ["HOPE"]
_THETAS = ['0.30']
_REPS = np.arange(1,5)
_LAYERS = ["1-2", "1-3", "2-3"]

TARGETS_ = expand(
        DIR_MODELS+"model_embed-HOPE_remnants_theta-{theta}_rep-{rep}_clean-multiplex-homo_{layerpair}.model",
        theta=_THETAS,
        rep=_REPS,
        layerpair=_LAYERS
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_