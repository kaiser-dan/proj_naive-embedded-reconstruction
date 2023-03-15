"""Project source code for common multiplex pre-processing utility.
"""
# ============= SET-UP =================
# --- Data handling ---
import pandas as pd

# =================== FUNCTIONS ===================
def df_from_records(records):
    return pd.DataFrame.from_records(records)