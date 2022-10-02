# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
import pickle  # For serializing output

# --- Scientific ---
from sklearn.metrics import accuracy_score, roc_auc_score  # Measuring classifier performance

# --- Data handling and visualization ---
import pandas as pd  # Dataframe tools

# --- Miscellaneous ---


# =================== FUNCTIONS ===================
# --- Drivers ---
def measure_performance(df_reconstruction):
    auroc = helper_auroc(df_reconstruction["Score"], df_reconstruction["Ground_Truth"])
    accuracy = helper_auroc(df_reconstruction["Classification"], df_reconstruction["Ground_Truth"])

    return auroc, accuracy

# --- Helpers ---
def helper_auroc(scores, gt):
    return roc_auc_score(gt, scores)

def helper_accuracy(classes, gt):
    return accuracy_score(gt, classes)


# ============== MAIN ===============
def main(reconstruction):
    auroc, accuracy = measure_performance(reconstruction)

    record = dict(snakemake.wildcards)
    record.update({
        "AUROC": auroc,
        "Accuracy": accuracy
    })

    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(record, _fh, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Load pickled duplex from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        reconstruction = pickle.load(_fh)

    # Run observation procedure
    main(reconstruction)
