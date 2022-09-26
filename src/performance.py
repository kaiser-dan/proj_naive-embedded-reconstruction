# ============= SET-UP =================
# --- Standard library ---

# --- Scientific ---
from sklearn.metrics import accuracy_score, roc_auc_score  # Measuring classifier performance

# --- Data handling and visualization ---
import pandas as pd  # Dataframe tools

# --- Miscellaneous ---


# =================== FUNCTIONS ===================
# --- Drivers ---
def measure_performance(df_reconstruction):
    auroc = helper_auroc(df_reconstruction["Score"], df_reconstruction["Ground_Truth"])
    accuracy = helper_auroc(df_reconstruction["Classification"], df_reconstruction["Ground_Truth"])

    return auroc, accuracy

# --- Helpers ---
def helper_auroc(scores, gt):
    return roc_auc_score(gt, scores)

def helper_accuracy(classes, gt):
    return accuracy_score(gt, classes)