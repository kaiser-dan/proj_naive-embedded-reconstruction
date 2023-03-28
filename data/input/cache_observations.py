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

from data import dataio, preprocessing, observations
from utils import parameters as params

# --- Miscellaneous ---
from tqdm import tqdm

# ================ FUNCTIONS ======================
def main():
    # >>> Book-keeping >>>
    systems = [
        ("arxiv", 2, 6, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("arxiv", ROOT=ROOT).format(system="arxiv")), *[2, 6])),
        ("celegans", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("celegans", ROOT=ROOT).format(system="celegans")), *[1, 2])),
        ("drosophila", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("drosophila", ROOT=ROOT).format(system="drosophila")), *[1, 2])),
        ("london", 1, 2, preprocessing.duplex_network(dataio.read_file(dataio.get_input_filehandle("london", ROOT=ROOT).format(system="london")), *[1, 2]))
    ]
    reps = range(1, 11)
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
    main()
