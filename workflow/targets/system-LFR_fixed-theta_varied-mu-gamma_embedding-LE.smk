import numpy as np

_EMBEDDINGS = ["LE"]
_THETAS = ["0.30"]
_REPS = np.arange(1,5)
_MPLXS = [
    x.split("multiplex-")[1]
    for x in basenames(files_in(DIR_EDGELISTS))
    if "LFR" in x \
      and "N-100_" not in x
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
