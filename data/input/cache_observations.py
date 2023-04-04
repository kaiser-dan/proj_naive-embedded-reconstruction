"""Script to precompute and cache observational data for reconstruction experiments.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
from itertools import product

# --- Scientific computing ---
import numpy as np

# --- Project source ---
# PATH adjustments
ROOT = "../../"
sys.path.append(f"{ROOT}/src/")

from data import dataio, preprocessing, observations, benchmarks
from utils import parameters as params

# --- Miscellaneous ---
from tqdm import tqdm

# ================ FUNCTIONS ======================
def main(args):
    # >>> Book-keeping >>>
    if len(args) == 1 or args[1] == "real":
        systems = [
            ("arxiv", 2, 6, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("arxiv", ROOT=ROOT).format(system="arxiv")), *[2, 6])),
            ("celegans", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("celegans", ROOT=ROOT).format(system="celegans")), *[1, 2])),
            ("drosophila", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("drosophila", ROOT=ROOT).format(system="drosophila")), *[1, 2])),
            ("london", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("london", ROOT=ROOT).format(system="london")), *[1, 2]))
        ]
    else:
        systems = [
            (
                f"LFR_mu-{mu}_prob-{prob}",
                1, 2,
                preprocessing.duplex_network(
                    benchmarks.lfr_multiplex(
                        10_000,
                        2.1, 1.0,
                        mu, 20,
                        np.sqrt(10_000),
                        1,
                        prob)[0],
                    1, 2
                )
            )
            for mu in [0.1] #, 0.2, 0.3, 0.4, 0.5]
            for prob in [0.0] #, 0.25, 0.5, 0.75, 1.0]
        ]


    reps = range(1, 1)
    thetas = np.linspace(0.05, 0.95, 11, endpoint=True)
    parameters, hyperparameters, _ = params.set_parameters_N2V(workers=32)
    # <<< Book-keeping <<<

    grid = product(systems, thetas, reps)

    for system_, theta, rep in tqdm(grid, desc="Caching observational data..."):
        system, l1, l2, (G, H) = system_

        observations.calculate_preprocessed_data(
            G, H,
            system, [l1, l2],
            theta, rep,
            parameters, hyperparameters,
            ROOT="preprocessed/"
        )


# ==================== MAIN =======================
if __name__ == "__main__":
    main(sys.argv)
