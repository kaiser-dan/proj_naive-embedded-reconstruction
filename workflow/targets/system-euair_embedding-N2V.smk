import numpy as np

_EMBEDDINGS = ["N2V"]
_THETAS = [
    f"{theta:.2f}"
    for theta in np.linspace(0.05, 0.95, 19)
]
_REPS = np.arange(1,5)
_LAYERS = ["1-2", "1-3", "2-3"]

TARGETS_ = expand(
        DIR_MODELS+"model_embed-N2V_remnants_theta-{theta}_rep-{rep}_clean-multiplex-euair_{layerpair}.model",
        theta=_THETAS,
        rep=_REPS,
        layerpair=_LAYERS
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_