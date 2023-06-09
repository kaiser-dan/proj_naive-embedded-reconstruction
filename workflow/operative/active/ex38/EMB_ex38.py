"""Experiment script to reconstruct synthetic duplexes with N2V.

See `protocol_EMB_ex38.md` for additional details.
"""
# ========== SET-UP ==========
# --- Standard library ---
import sys
from datetime import datetime
from itertools import product
import yaml

# --- Scientific computing ---
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# --- Network science ---

# --- Data handling ---
import pandas as pd

# --- Project source ---
# PATH adjustments
ROOT = "../../../../"
sys.path.append(f"{ROOT}/")
sys.path.append(f"{ROOT}/src/")

# Primary modules
## Data
from data import dataio
from data import postprocessing
from data import observations

## Classifiers
from src.classifiers import features  # feature set helpers

## Utilities
from src.utils import parameters as params  # helpers for experiment parameters

# --- Miscellaneous ---
import logging
from datetime import datetime  # logging tag
from time import perf_counter, time  # simple pseudo-profiling
from tqdm.auto import tqdm  # progress bars

import warnings
warnings.filterwarnings("ignore")  # remove sklearn depreciation warnings

# ========== FUNCTIONS ==========
    



# ========== MAIN ==========
def main():
    # Load up config

    # 
    return 0


# -----------
if __name__ == "__main__":
    main()