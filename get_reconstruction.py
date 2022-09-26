# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
import pickle  # For serializing output

# --- Project code ---
sys.path.append("../../../src/")
from src.reconstruct import reconstruct_system


# ============== FUNCTIONS ===========

# ============== MAIN ===============
def main(vectors, test_set, metric):
    # Book-keeping
    remnant_G_vectors, remnant_H_vectors = vectors

    # Reconstruct system
    reconstruction = reconstruct_system(remnant_G_vectors, remnant_H_vectors, test_set, metric)

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(reconstruction, _fh, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Load pickled system from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        test_set = pickle.load(_fh)["test_set"]

    # Load pickled vectors from Snakemake input
    with open(snakemake.input[1], "rb") as _fh:
        vectors = pickle.load(_fh)

    # Run observation procedure
    main(vectors, test_set, snakemake.wildcards["metric"])
