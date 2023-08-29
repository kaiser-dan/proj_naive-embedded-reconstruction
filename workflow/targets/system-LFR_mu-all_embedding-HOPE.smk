import numpy as np

_EMBEDDINGS = ["HOPE"]
_THETAS = [f"{theta:.2f}" for theta in np.linspace(0.05, 0.95, 19)]
_REPS = np.arange(1,5)
_MPLXS = [
    x.split("multiplex-")[1]
    for x in basenames(files_in(DIR_EDGELISTS))
    if "LFR" in x
]

TARGETS_ = expand(
        DIR_MODELS+"model_embed-{embedding}_remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.model",
        embedding=_EMBEDDINGS,
        theta=_THETAS,
        rep=_REPS,
        mplx=_MPLXS
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_