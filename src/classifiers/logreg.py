"""Project source code for logistic regression edge classification for multiplex reconstruction
"""
# ========== SET-UP ==========
# --- Scientific computing ---
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ========== FUNCTIONS ==========
# --- Model training ---
def train_fit_logreg(
        X_train, y_train,
        kwargs):
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)

    return model

def get_model_fit(model):
    return model.intercept_, model.coef_

# --- Reconstruction ---
def get_reconstruction(model, X_test):
    return model.predict(X_test)

def get_scores(model, X_test, class_label=1):
    # class_label = 1 indicates scoring for classification in G.
    return model.predict_proba(X_test)[:, class_label]

# --- Performance measures ---
def get_model_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

def get_model_auroc(model, X_test, y_test):
    scores = get_scores(model, X_test)
    return roc_auc_score(y_test, scores)

def get_model_aupr(model, X_test, y_test):
    scores = get_scores(model, X_test)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    return auc(recall, precision)