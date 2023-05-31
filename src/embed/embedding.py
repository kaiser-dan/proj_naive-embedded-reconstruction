"""
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import pickle

# --- Scientific computation ---
import numpy as np

# ============= CLASSES =================
class Embedding:
    def __init__(
            self,
            vectors: np.array,
            embedder: str,
            _aligned: bool = False,
            _scaled: bool = False):
        # Data assignment
        self.vectors = vectors
        self.embedder = embedder

        self._aligned = _aligned
        self._scaled = _scaled

        return


    # --- Private methods ---


    # --- Public methods ---
    # > Vector pre-processing >
    def align_vectors(self, components):
        for component in components:
            # Get component center of mass
            component_center = np.mean(self.vectors[component])

            # Translate each vector by center of mass
            # & Has effect of recentering vectors about the origin
            for vector_id in component:
                self.vectors[vector_id] -= component_center

        return

    def scale_vectors(self, components):
        for component in components:
            # Get total component norm
            component_total_norm = \
                np.sum([
                    np.linalg.norm(self.vectors[vector_id])
                    for vector_id in component
                ])

            # Normalize vectors
            # & Has effect that norms sum to unity
            for vector_id in component:
                self.vectors[vector_id] /= component_total_norm

    # > I/O >
    def save(self, filepath: str, only_vectors: bool = False):
        save_embedding(self, filepath, only_vectors)


# ============= FUNCTIONS =================
def save_embedding(embedding: Embedding, filepath: str, only_vectors: bool = False):
    if only_vectors:
        embedding.vectors.tofile(filepath)
    else:
        try:
            fh = open(filepath, "wb")
            pickle.dump(embedding, fh, pickle.HIGHEST_PROTOCOL)
        except Exception as err:
            sys.stderr.write(f"{err}\n Error serializing Embedding instance!")
        finally:
            fh.close()