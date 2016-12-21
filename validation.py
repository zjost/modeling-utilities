"""
To help deliver functionality useful when prototyping in
a jupyter notebook sandbox
"""

from sklearn.model_selection import StratifiedKFold



def logreg_gridsearch(X, y, n_splits, c_list):
    # Run basic grid search
    skf = StratifiedKFold(n_splits=n_splits)

    score_summary = dict()
    for c in c_list:
        score_summary[c] = list()
        for train_idx, test_idx in skf.split(X, y.values):
            X_train, y_train = X.loc[train_idx,], y[train_idx]
            X_test, y_test = X.loc[test_idx,], y[test_idx]
            roc_auc = logreg(X_train, y_train, X_test, y_test, c)
            score_summary[c].append(roc_auc)

    return score_summary