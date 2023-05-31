import sys
import pickle

class ReconstructionModel:
    def __init__(
            self,
            model_type: str,  # type of reconstruction classifier, e.g., "logistic regression"
            features: tuple,  # names of included features - considered as ordered!
            experiment_params: dict,  # experimental parameters and their values
            ):
        self.model_type = model_type
        self.features = features
        self.experiment_params = experiment_params

        return

    def __str__(self):
        return f"""Reconstruction Model Instance\n-----------------------------\nClassifier: {self.model_type}\nFeatures: {self.features}\nExperimental Parameters: {self.experiment_params}\n-----------------------------\n"""


def save_model(model: ReconstructionModel, filepath: str):
    try:
        fh = open(filepath, "wb")
        pickle.dump(model, fh, pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        sys.stderr.write(f"{err}\n Error serializing ReconstructionModel instance!")
    finally:
        fh.close()