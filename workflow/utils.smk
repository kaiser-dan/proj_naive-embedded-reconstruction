# ===============
# Python helpers
# ===============
import os

def basenames(filepaths):
    return [os.path.basename(fp) for fp in filepaths]