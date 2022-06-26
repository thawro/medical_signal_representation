from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_classification_metrics(y_pred_proba, y_true, average="macro"):
    y_pred = y_pred_proba.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    fscore = f1_score(y_true, y_pred, average=average)
    auc = roc_auc_score(y_true, y_pred_proba, average=average, multi_class="ovr")
    return {"fscore": fscore, "acc": acc, "auc": auc}
