"""
"""
# ============= SET-UP =================
# --- Standard library ---
import sys
import pickle

# ============= CLASSES =================
class ReconstructionModel:
    """General class for applying a multiplex reconstruction classifier.

    Data
    ----
    model_type : str
        Type of reconstruction classifier, e.g., "logistic regression"
    features : tuple
        Names of included features - considered as ordered!
    experiment_params : dict
        Experimental parameters and their values


    Methods
    -------
    save(filepath: str)
        Saves object to the given filepath.
    """
    def __init__(
            self,
            model_type: str,
            features: tuple,
            experiment_params: dict,
            ):
        # Data assignment
        self.model_type = model_type
        self.features = features
        self.experiment_params = experiment_params

        return


    # --- Private methods ---
    def __str__(self):
        return f"""Reconstruction Model Instance\n-----------------------------\nClassifier: {self.model_type}\nFeatures: {self.features}\nExperimental Parameters: {self.experiment_params}\n-----------------------------\n"""


    # --- Public methods ---
    # > I/O >
    def save(self, filepath: str):
        save_model(self, filepath)


# ============= FUNCTIONS =================
def save_model(model: ReconstructionModel, filepath: str):
    try:
        fh = open(filepath, "wb")
        pickle.dump(model, fh, pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        sys.stderr.write(f"{err}\n Error serializing ReconstructionModel instance!")
    finally:
        fh.close()