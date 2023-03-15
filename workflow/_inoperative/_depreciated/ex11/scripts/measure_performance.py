# ============= SET-UP =================
# --- Standard library ---
import pickle  # For serializing output

# --- Scientific ---
from sklearn.metrics import accuracy_score, roc_auc_score  # Measuring classifier performance

# --- Data handling and visualization ---

# --- Miscellaneous ---


# =================== FUNCTIONS ===================
# --- Drivers ---
def measure_performance(reconstruction):
    auroc = \
        roc_auc_score(
            reconstruction["origination"], reconstruction["score"]
        )
    accuracy = \
        accuracy_score(
            reconstruction["origination"], reconstruction["classification"]
        )

    return auroc, accuracy


# ============== MAIN ===============
def main(reconstruction, record):
    auroc, accuracy = measure_performance(reconstruction)

    record.update({
        "AUROC": auroc,
        "Accuracy": accuracy
    })

    return record


if __name__ == "__main__":
    # Load reconstruction
    with open(snakemake.input["reconstruction"], "rb") as _fh:
        reconstruction = pickle.load(_fh)

    # Initialize record
    wc = dict(snakemake.wildcards)
    record = {
        "avg_k": wc["avg_k"],
        "gamma": wc["gamma"],
        "mu": wc["mu"],
        "metric": wc["metric"],
        "pfi": wc["pfi"],
        "rep": wc["rep"]
    }

    # Run observation procedure
    record = main(reconstruction, record)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(record, _fh, pickle.HIGHEST_PROTOCOL)
