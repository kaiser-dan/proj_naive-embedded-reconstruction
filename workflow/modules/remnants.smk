# =====================
# Remnant observations
# =====================
# Main observation rule - Observes experimental training data from mulitplex
rule observe_remnants_random:
    input:
        DIR_EDGELISTS + "multiplex-{filepath}"
    output:
        DIR_REMNANTS + "remnants_strategy-{strategy}_theta-{theta}_remrep-{rep}_multiplex-{filepath}"
    shell:
        "python " + SCRIPTS + "observe_remnant.py {input} {wildcards.theta} --strategy {wildcards.strategy} -r {wildcards.rep}"


# rule observe_remnants_snowball:

# rule observe_remnants_random_cumulative:

# rule observe_remnants_snowball_cumulative:
# ../../data/input/remnants/remnants_strategy-RANDOM_theta-0.9_remrep-1_multiplex-LFR_N-100_T1-2.1_T2-1.0_kavg-6_kmax-10_mu-0.1_prob-1.0.pkl
# ../../data/input/remnants/remnants_strategy-RANDOM_theta-0.9_remrep-1_multiplex-LFR_N-100_T1-2.1_T2-1.0_kavg-6_kmax-10_mu-0.1_prob-1.0.pkl