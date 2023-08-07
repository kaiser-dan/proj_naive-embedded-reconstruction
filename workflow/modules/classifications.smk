# ================================
# Model training/evaluation rules
# ================================
# Model training rule - Trains reconstruction model and gathers test features
rule train_models:
    input:
        remnants = DIR_REMNANTS + "remnants_strategy-RANDOM_theta-{theta}_remrep-{remrep}_multiplex-{filepath}",
        embeddings = DIR_EMBEDDINGS + "embeddings_method-{method}_dim-{dim}_embrep-{embrep}_remnants_strategy-RANDOM_theta-{theta}_remrep-{remrep}_multiplex-{filepath}"
    output:
        DIR_MODELS + "model_normalized-{normalize}_embeddings_method-{method}_dim-{dim}_embrep-{embrep}_remnants_strategy-RANDOM_theta-{theta}_remrep-{remrep}_multiplex-{filepath}"
    shell:
        "python " + SCRIPTS + "train_model.py {input.remnants} {input.embeddings} --normalize"

# rule evaluate_models: