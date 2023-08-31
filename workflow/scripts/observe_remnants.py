import sys
import os

import numpy as np

from EMB import mplxio
from EMB import remnants

THETAS = np.linspace(0.00, 0.95, num=20, endpoint=True, dtype=float)

def _parse_args(args):
    """Assumes input {input} {output}.

    where:
    - {input} is filepath for multiplex edgelist
    - {output} is filepath for output remnant multiplex edgelist. Must contain "{theta}" for filepath completion.
    """
    # Ensure input file exists
    if not os.path.exists(args[0]):
        raise FileNotFoundError(args[0])

    # Ensure directory path of output file exists
    # if not os.path.exists(os.path.dirname(args[1])):
        # os.system(f"mkdir {os.path.dirname(args[1])}")

    # Ensure output contains necessary identifiers
    # if "{theta}" not in args[1]:
        # raise ValueError("Output filepath must contain '{theta}'")

    return args


def main(filepath_input, filepath_output):
    # Bring multiplex into memory
    multiplex = mplxio.from_edgelist(filepath_input)

    # Cumulatively observe remnants
    remnant_multiplexes = remnants.cumulative_remnant_multiplexes(multiplex, THETAS)

    # Save each remnant to its own file
    for theta, remnant_multiplex in remnant_multiplexes.items():
        mplxio.to_edgelist(remnant_multiplex, filepath_output.format(theta=f"{theta:.2f}"))

    return


if __name__ == "__main__":
    # Check and parse args
    # args = _parse_args(sys.argv[1:])

    # main(*args)

    ntwks = [
        # "clean-multiplex-arxiv_l1-2_l2-6.mplx",
        # "clean-multiplex-arxiv_l1-2_l2-7.mplx",
        # "clean-multiplex-arxiv_l1-6_l2-7.mplx",
        # "clean-multiplex-celegans_l1-1_l2-2.mplx",
        # "clean-multiplex-celegans_l1-1_l2-3.mplx",
        # "clean-multiplex-celegans_l1-2_l2-3.mplx",
        # "clean-multiplex-drosophila_l1-1_l2-2.mplx",
        # "clean-multiplex-drosophila_l1-1_l2-3.mplx",
        # "clean-multiplex-drosophila_l1-2_l2-3.mplx",
        # "clean-multiplex-london_l1-1_l2-2.mplx",
        # "clean-multiplex-london_l1-1_l2-3.mplx",
        # "clean-multiplex-london_l1-2_l2-3.mplx",
        "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.1_prob-1.0.mplx",
        "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.2_prob-1.0.mplx",
        "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.3_prob-1.0.mplx",
        "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.4_prob-1.0.mplx",
        "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.5_prob-1.0.mplx",
        # "multiplex-LFR_N-10000_T1-4.0_T2-1.0_kavg-6.0_kmax-100_mu-0.1_prob-1.0.mplx",
        # "multiplex-LFR_N-10000_T1-2.1_T2-1.0_kavg-6.0_kmax-100_mu-0.5_prob-1.0.mplx",
        # "multiplex-LFR_N-10000_T1-2.7_T2-1.0_kavg-6.0_kmax-100_mu-0.3_prob-1.0.mplx",
    ]

    for ntwk in ntwks:
        print(f"Observing {ntwk}...")
        for rep in range(1, 6):
            args = [
                f"data/edgelists/{ntwk}",
                "data/remnants/remnants_theta-{theta}_rep-" + str(rep) + "_" + ntwk[:-5] + ".rmnt",
            ]

            main(*args)
