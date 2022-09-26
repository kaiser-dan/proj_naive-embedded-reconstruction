# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
import pickle  # For serializing output

# --- Project code ---
sys.path.append("../../../src/")
from src.performance import measure_performance


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
