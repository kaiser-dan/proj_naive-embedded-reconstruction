TARGETS_ = expand(
        DIR_MODELS+"model_embed-{embedding}_remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.model",
        embedding=["HOPE"],
        theta=[
            f"{x:.2f}"
            for x in [
                0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            ],
        rep=[1,2,3,4,5], mplx=[x.split("multiplex-")[1] for x in basenames(files_in(DIR_EDGELISTS)) if "LFR" in x]
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_