ruleorder: embeddings_real > embeddings_LFR

rule embeddings_LFR:
    input:
        DIR_REMNANTS+"remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.rmnt"
    output:
        DIR_EMBEDDINGS+"embed-{embedding}_remnants_theta-{theta}_rep-{rep}_multiplex-{mplx}.vecs"
    log:
        DIR_LOGS+"embedding_{embedding}_{theta}-{rep}_{mplx}.log"
    shell:
        "python workflow/scripts/embed_remnants.py {input} {wildcards.embedding} {output}"

rule embeddings_real:
    input:
        DIR_REMNANTS+"remnants_theta-{theta}_rep-{rep}_clean-multiplex-{system}_l1-{l1}_l2-{l2}.rmnt"
    output:
        DIR_EMBEDDINGS+"embed-{embedding}_remnants_theta-{theta}_rep-{rep}_clean-multiplex-{system}_l1-{l1}_l2-{l2}.vecs"
    log:
        DIR_LOGS+"embedding_{embedding}_{theta}-{rep}_{system}_{l1}-{l2}.log"
    shell:
        "python workflow/scripts/embed_remnants.py {input} {wildcards.embedding} {output}"