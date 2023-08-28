TARGETS_ = expand(
        DIR_MODELS+"model_embed-HOPE_remnants_theta-{theta}_rep-{rep}_clean-multiplex-celegans_{layerpair}.model",
        theta=[
            f"{x:.2f}"
            for x in [
                0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            ],
        rep=[1,2,3,4,5], layerpair = ["l1-1_l2-2", "l1-1_l2-3", "l1-2_l2-3"]
    )

if 'TARGETS' in globals():
    TARGETS += TARGETS_
else:
    TARGETS = TARGETS_