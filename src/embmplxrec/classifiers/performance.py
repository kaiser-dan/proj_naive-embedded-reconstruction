from sklearn import metrics

def auroc(model, X, Y, class_ = 1):
    Yhat = model.predict_proba(X)[:, class_]
    return metrics.roc_auc_score(Y, Yhat)

def accuracy(model, X, Y):
    Yhat = model.predict(X)
    return metrics.accuracy_score(Y, Yhat)

def pr(model, X, Y, class_ = 1):
    Yhat = model.predict_proba(X)[:, class_]
    precision, recall, _ = metrics.precision_recall_curve(Y, Yhat)
    return metrics.auc(recall, precision)