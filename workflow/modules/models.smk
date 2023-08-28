ruleorder: train_models_real > train_models_LFR

rule train_models_LFR:
    input:
        gt=DIR_EDGELISTS+"multiplex-{mplx}.mplx",
        rmnt=DIR_REMNANTS+"remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.rmnt",
        emb=DIR_EMBEDDINGS+"embed-{embedding}_remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.vecs"
    output:
        DIR_MODELS+"model_embed-{embedding}_remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.model"
    log:
        DIR_LOGS+"model_{embedding}_{theta}-{rep}_{mplx}.log"
    shell:
        "python workflow/scripts/reconstruct_multiplex.py {input.gt} {input.rmnt} {input.emb} {output} > {output}"

rule train_models_real:
    input:
        gt=DIR_EDGELISTS+"clean-multiplex-{system}_l1-{l1}_l2-{l2}.mplx",
        rmnt=DIR_REMNANTS+"remnants_theta-{theta}_rep-{rep}_clean-multiplex-{system}_l1-{l1}_l2-{l2}.rmnt",
        emb=DIR_EMBEDDINGS+"embed-{embedding}_remnants_theta-{theta}_rep-{rep}_clean-multiplex-{system}_l1-{l1}_l2-{l2}.vecs"
    output:
        DIR_MODELS+"model_embed-{embedding}_remnants_theta-{theta}_rep-{rep}_clean-multiplex-{system}_l1-{l1}_l2-{l2}.model"
    log:
        DIR_LOGS+"model_{embedding}_{theta}-{rep}_{system}_{l1}-{l2}.log"
    shell:
        "python workflow/scripts/reconstruct_multiplex.py {input.gt} {input.rmnt} {input.emb} {output} > {output}"
