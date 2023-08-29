import numpy as np

_EMBEDDINGS = ["N2V"]
_THETAS = [
    f"{theta:.2f}"
    for theta in np.linspace(0.05, 0.95, 19)
]
_REPS = np.arange(1,5)
_LAYERS = ["2-6", "2-7", "6-7"]

TARGETS_ = expand(
        DIR_MODELS+"model_embed-HOPE_remnants_theta-{theta}_rep-{rep}_clean-multiplex-arxiv_{layerpair}.model",
        theta=_THETAS,
        rep=_REPS,
        layerpair=_LAYERS
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_