"""Project source code for logistic regression edge classification for multiplex reconstruction
"""
# ========== SET-UP ==========
# --- Scientific computing ---
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

def get_scores(model, X_test):
    return model.predict_proba(X_test)[:, 1]

# --- Performance measures ---
def get_model_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

def get_model_auroc(model, X_test, y_test):
    scores = get_scores(model, X_test)
    return roc_auc_score(y_test, scores)
