"""Project source code for logistic regression edge classification for multiplex reconstruction
"""
# ========== SET-UP ==========
# --- Standard library ---
from collections import Counter

# --- Scientific computing ---
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Source code ---
from classifiers.model import ReconstructionModel
from classifiers.performance import performance


# ========== CLASSES ============
class LogReg(ReconstructionModel):
    def __init__(
            self,
            model_type: str, features: tuple, experiment_params: dict,
            training_data: np.array,
            training_labels: np.array,
            logreg_parameters: dict):
        # Inheritance initialization
        super().__init__(model_type, features, experiment_params)

        # Data assignment
        self.X = training_data
        self.Y = training_labels
        self.logreg_parameters = logreg_parameters

        # Model instantiation
        self.__start_sklearn_model()
        self.__train()

        return


    # --- Private methods ---
    def __start_sklearn_model(self):
        self._model = LogisticRegression(**self.logreg_parameters)
        return

    def __train(self):
        self._model.fit(self.X, self.Y)
        self.intercept = self._model.intercept_[0]
        self.coefficients = self._model.coef_[0]
        return

    # --- Public methods ---
    # > Examining given data >
    def get_data(self, label=None):
        if label is not None:
            indices = np.nonzero(self.Y == label)
            return self.X[indices]
        else:
            return self.X

    def label_distribution(self):
        return dict(Counter(self.Y))

    # > Model inspection >
    def get_coefficients(self):
        coefficients = {"intercept": self.intercept}
        coefficients.update({
            self.features[feature_idx]: self.coefficients[feature_idx]
            for feature_idx in range(len(self.features))
        })

        return coefficients

    def training_performance(self, measure="accuracy"):
        scored_labels = self.get_scores(self.X)
        predicted_labels = self.get_reconstruction(self.X)
        true_labels = self.Y
        return performance(scored_labels, predicted_labels, true_labels, measure=measure)

    # > Model application >
    def decision_function(self, X_):
        return self._model.decision_function(X_)

    def get_reconstruction(self, X_):
        return self._model.predict(X_)

    def get_scores(self, X_, class_label=1):
        # class_label = 1 indicates scoring for classification in G.
        return self._model.predict_proba(X_)[:, class_label]