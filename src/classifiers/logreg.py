"""Project source code for logistic regression edge classification for multiplex reconstruction
"""
# ========== SET-UP ==========
# --- Scientific computing ---
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ========== FUNCTIONS ==========
# --- Model training ---
def train_fit_logreg(X_train, y_train, seed=37):
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)

    return model

def get_model_fit(model):
    return model.intercept_[0], model.coef_[0][0]

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


# --- Helpers ---
def prepare_feature_matrix(
    distances_G, distances_H,
    degrees_i_G, degrees_i_H,
    degrees_j_G, degrees_j_H
):
    # >>> Book-keeping >>>
    NUM_FEATURES = 6  # number of features
    length = len(distances_G)  # number of observations in dataset
    feature_matrix = np.empty((length, NUM_FEATURES))  # initialize feature matrix
    # <<< Book-keeping <<<

    # >>> Format feature matrix >>>
    feature_matrix[:, 0] = distances_G
    feature_matrix[:, 1] = distances_H
    feature_matrix[:, 2] = degrees_i_G
    feature_matrix[:, 3] = degrees_i_H
    feature_matrix[:, 4] = degrees_j_G
    feature_matrix[:, 5] = degrees_j_H
    # <<< Format feature matrix <<<

    return feature_matrix