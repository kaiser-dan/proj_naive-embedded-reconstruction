# =============================
# Synthetic multiplex sampling
# =============================
# Main synthesis rule - Generates a multiplex from an LFR model
rule generate_synthetics_LFR:
    output:
        DIR_EDGELISTS + "multiplex-LFR_N-{N}_T1-{t1}_T2-{t2}_kavg-{kavg}_kmax-{kmax}_mu-{mu}_prob-{prob}.pkl"
    shell:
        "python " + SCRIPTS + "generate_synthetic-LFR.py -N {wildcards.N} -u {wildcards.mu} -d {wildcards.t1} -c {wildcards.t2} -k {wildcards.kavg} -m {wildcards.kmax} -p {wildcards.prob}"

# rule generate_synthetics_PSO: