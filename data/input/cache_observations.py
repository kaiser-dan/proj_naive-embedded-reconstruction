"""Script to precompute and cache observational data for reconstruction experiments.
"""
# ================= SET-UP =======================
# --- Standard library ---
import sys
from itertools import product
from enum import Enum

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


# --- Globals ---
class Config(float, Enum):
    # LFR params
    N = 1000.0  # convert to int
    AVG_K = 6.0
    T1 = 2.1
    T2 = 1.0
    PROB = 1.0
    MU = 0.1
    # Observations
    REPS = 1.0  # convert to int
    THETA_MIN = 0.05
    THETA_MAX = 0.95
    THETA_NUM = 37.0  # convert to int
    # Resources
    WORKERS = 60.0  # convert to int


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
                f"LFR_gamma-{T1}",
                1, 2,
                preprocessing.duplex_network(
                    benchmarks.lfr_multiplex(
                        int(Config.N.value),
                        T1, Config.T2.value,
                        Config.MU.value,
                        Config.AVG_K.value, np.sqrt(Config.N.value) if T1 <= 3 else np.power(Config.N.value, 1/(T1-1)),  # <k>, max(k)
                        1,  # min_community - ignored downstream
                        Config.PROB.value)[0],
                    1, 2
                )
            )
            for T1 in [2.1, 2.5, 2.9, 3.5]
        ]


    reps = range(1, int(Config.REPS.value)+1)
    thetas = np.linspace(Config.THETA_MIN.value, Config.THETA_MAX.value, int(Config.THETA_NUM.value), endpoint=True)
    parameters, hyperparameters, _ = params.set_parameters_N2V(workers=int(Config.WORKERS.value))
    # <<< Book-keeping <<<

    grid = list(product(systems, thetas, reps))
    print(f"Discovered {len(grid)} parameter grid vertices")

    for system_, theta, rep in tqdm(grid, desc="Caching observational data..."):
        system, l1, l2, (G, H) = system_

        # ! >>> DEBUG >>>
        print(f"DEBUG - System={system}; layers=({l1}, {l2}); rep={rep}")
        # ! <<< DEBUG <<<

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
