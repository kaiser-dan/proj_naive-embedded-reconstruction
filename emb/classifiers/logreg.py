"""Logistic regression model for multiplex reconstruction.
"""
# ============= SET-UP =================
__all__ = ["train_model", "evaluate_model"]

# --- Imports ---
from loguru import logger as LOGGER

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc


# --- Globals ---
LOGREG_HYPERPARAMETERS = {
    # Functional form
    "fit_intercept": False,
    "penalty": None,
    # Regression mechanics
    "solver": "newton-cholesky",
    # Convergence hyperparameters
    "tol": 1e-4,
    "max_iter": 1000,
}

# =================== FUNCTIONS ===================
def train_model(X_train, y_train):
    model = LogisticRegression(**LOGREG_HYPERPARAMETERS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_=1):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, class_]

    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr = auc(recall, precision)

    return accuracy, auroc, pr
