# =====================
# Remnant embeddings
# =====================
# Main embedding rule - Embeds a specified remnant multiplex
rule embed_remnants:
    input:
        DIR_REMNANTS + "remnants_{filepath}"
    output:
        DIR_EMBEDDINGS + "embeddings_method-{method}_dim-{dim}_embrep-{rep}_remnants_{filepath}"
    shell:
        "python " + SCRIPTS + "embed_remnant.py {input} {wildcards.method} {wildcards.dim} -r {wildcards.rep}"