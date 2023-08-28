# ! BROKEN
# import numpy as np

# rule observe_remnants_real:
#     input:
#         DIR_EDGELISTS+"multiplex-{system}_l1-{l1}_l2-{l2}.mplx",
#     output:
#         expand(
#             DIR_REMNANTS+"remnants_theta-{theta}_multiplex-{{system}}_l1-{{l1}}_l2-{{l2}}.rmnt",
#             theta=[f"{x:.2f}" for x in np.linspace(0.00, 0.95, num=20, endpoint=True, dtype=float)],
#             # system=wildcards.system,
#             # l1=wildcards.l1,
#             # l2=wildcards.l2
#         )
#     # log:
#     #     DIR_LOGS+"remnant_{theta}_{system}_{l1}-{l2}.log"
#     shell:
#         "python workflow/scripts/observe_remnant.py {input}"
