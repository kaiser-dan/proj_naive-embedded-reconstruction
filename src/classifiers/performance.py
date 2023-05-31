from sklearn import metrics

def performance(
        scored_labels,  # probability of belonging to class "1"
        predicted_labels,  # projection of scored_labels onto {0, 1}
        true_labels,  # true labels
        measure="accuracy",
        **kwargs):
    measure = measure.upper()  # remove ambiguity of input case
    if "ACC" in measure:
        return metrics.accuracy_score(true_labels, predicted_labels, **kwargs)
    elif ("AUC" in measure) \
        or ("ROC" in measure):
        return metrics.roc_auc_score(true_labels, scored_labels, **kwargs)
    elif "PR" in measure:
        scores = metrics.accuracy_score(true_labels, predicted_labels, **kwargs)
        precision, recall, _ = metrics.precision_recall_curve(true_labels, scores, **kwargs)
        return metrics.auc(recall, precision, **kwargs)
    else:
        raise NotImplementedError("Performance measure not implemented! Try \"accuracy\", \"auroc\", or \"aupr\".")
