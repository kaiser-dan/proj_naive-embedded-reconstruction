import sys
import os

import numpy as np

from EMB import mplxio
from EMB import remnants

THETAS = np.linspace(0.00, 0.95, num=20, endpoint=True, dtype=float)

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
    regex = ""
    if len(sys.argv) > 1:
        regex = sys.argv[1]
    ntwks = [fh for fh in os.listdir(os.path.join("data", "edgelists", "")) if regex in fh]
    for ntwk in ntwks:
        filepath_input = os.path.join('data', 'edgelists', f"{ntwk}")
        print(f"Observing {filepath_input}...")
        for rep in range(1, 6):
            filepath_output = os.path.join('data', 'remnants', "remnants_theta-{theta}_rep-" + str(rep) + "_" + ntwk[:-5] + ".rmnt")
            print("Calculating remnants...")
            output_filehandles = [os.path.exists(filepath_output.format(theta=f"{THETAS[x]:.2f}")) for x in range(1, len(THETAS))]
            if all(output_filehandles):
                print("Remnants exists already! Skipping remnant calculations...")
            else:
                main(filepath_input, filepath_output)
