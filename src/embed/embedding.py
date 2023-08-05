"""Source code for Embedding class object.
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import pickle

# --- Scientific computation ---
import numpy as np

# ============= CLASSES =================
class Embedding:
    """General class containing multiplex embedding data.

    Data
    ----
    vectors : np.array
        Embedded node vectors.
    embedder : str
        String labeling the graph embedding the vectors were obtained from.
    aligned : bool = False
        Indicator if the vector sets are aligned to origin.
    scaled : bool = False
        Indicator if vector sets are normalized by norm.


    Methods
    -------
    align_vectors(components)
        Aligns vector sets of each component to have center of mass at origin.
    scale_vectors(components)
        Scale the vector sets of each component to have norm density one.
    normalize(components)
        Aligns then scales vectors.
    save(filepath: str)
        Saves object to the given filepath.
    """
    def __init__(
            self,
            vectors: np.array,
            embedder: str,
            aligned: bool = False,
            scaled: bool = False):
        # Data assignment
        self.vectors = vectors
        self.embedder = embedder

        self._aligned = aligned
        self._scaled = scaled

        return


    # --- Properties ---
    @property
    def aligned(self):
        """Indicator if the vectors have been aligned to the origin."""
        return self._aligned

    @aligned.setter
    def aligned(self, value):
        if not isinstance(value, bool):
            raise TypeError("`aligned` property must be a boolean")
        else:
            self._aligned = value

    @property
    def scaled(self):
        """Indicator if the vectors have been scaled to cumulative unity norm."""
        return self._scaled

    @scaled.setter
    def scaled(self, value):
        if not isinstance(value, bool):
            raise TypeError("`scaled` property must be a boolean")
        else:
            self._scaled = value


    # --- Private methods ---
    def __eq__(self, other):
        return self.vectors == other.vectors


    # --- Public methods ---
    # > Vector pre-processing >
    def align_vectors(self, components):
        if self.aligned:
            print("Vectors already aligned!")
            return

        for component in components:
            # Get component center of mass
            component_center = np.mean([
                self.vectors[vector_id]
                for vector_id in component
                ])

            # Translate each vector by center of mass
            # & Has effect of recentering vectors about the origin
            for vector_id in component:
                self.vectors[vector_id] -= component_center

        self.aligned = True

        return

    def scale_vectors(self, components):
        if self.scaled:
            print("Vectors already scaled!")
            return

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

        self.scaled = True

        return

    def normalize(self, components):
        self.align_vectors(components)
        self.scale_vectors(components)

        return

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