# ============= SET-UP =================
# --- Standard library ---
import sys  # For adding src to path
import pickle  # For serializing output

# --- Project code ---
sys.path.append("../../../src/")
from src.embed import embed_system


# ============== FUNCTIONS ===========

# ============== MAIN ===============
def main(system_, parameters):
    # Book-keeping
    remnant_G, remnant_H = system_["remnant_duplex"]

    # Embed vectors
    remnant_vectors = embed_system(remnant_G, remnant_H, parameters, embedding_method = "node2vec")

    # Save to disk
    with open(snakemake.output[0], "wb") as _fh:
        pickle.dump(remnant_vectors, _fh, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # Load pickled duplex from Snakemake input
    with open(snakemake.input[0], "rb") as _fh:
        system_ = pickle.load(_fh)

    # Run observation procedure
    main(system_, snakemake.params["hyperparams"])
