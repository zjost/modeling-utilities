"""
To help deliver functionality useful when prototyping in
a jupyter notebook sandbox
"""
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV


def pretty_importances(clf, feature_list):
    """
    Returns a pretty dataframe of feature importances in
    decending order

    :param clf: trained sklearn estimator
    :param feature_list: list of features that match order
        of the dataframe used as the training data in clf
    :return:
    """

    try:
        importances = clf.__getattribute__('feature_importances')
    except AttributeError:
        print ("This estimator doesn't have feature importances!")
        return None

    return pd.DataFrame(
        {"feature": feature_list,
         "importance": importances,
         }
    ).sort_values(by='importance', ascending=False)


def stratified_test_train_split(X, y, test_size=0.3, random_state=None):
    """
    Creates a stratified test/train split

    :param X: <pd.DataFrame> Features
    :param y: <pd.DataFrame> Target to use stratification
    :param test_size: <float> Proportion of data to use for test set
    :param random_state: <int> Random state seed
    :return: a tuple of four dataframes that are the train/test stratified
        samples
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    for train_idx, test_idx in sss.split(X, y):
        X_train, y_train = X.loc[train_idx,], y[train_idx]
        X_test, y_test = X.loc[test_idx,], y[test_idx]

    return X_train, y_train, X_test, y_test


class BasicClassificationSearch(object):
    def __init__(self, model_list, X, y, **kwargs):
        self.model_list = model_list
        self.set_train_test(X, y, **kwargs)

        self.method_mapper = {
            'logistic_regression': self.log_reg_search,
            'adaboost': self.adaboost_search,
            'gradient_boosting': self.gradboosting_search,
        }
        self.cv_results_dict = dict()

        self.base_search_parameters = {
            "scoring": 'roc_auc',
            "refit": True,
            "cv": 3,
            "n_jobs": -1,
        }

    def run_search(self):
        for model_type in self.model_list:
            self.cv_results_dict.update({
                model_type: self.method_mapper[model_type]()
            })

    @property
    def results(self):
        if len(self.cv_results_dict) == 0:
            print 'Running search...'
            self.run_search()

        master_df = pd.DataFrame()

        for model, gridsearch in self.cv_results_dict.items():
            df = pd.DataFrame(gridsearch.cv_results_)
            df['model'] = model

            master_df.append(df)

        return master_df

    def set_train_test(self, X, y, **kwargs):
        arg_dict = {"X": X, "y": y}
        test_size = kwargs.get('test_size', None)
        if test_size:
            arg_dict.update({'test_size': test_size})
        random_state = kwargs.get('random_state', None)
        if random_state:
            arg_dict.update({'random_state': random_state})
        self.X_train, self.y_train, self.X_test, self.y_test = \
            stratified_test_train_split(**arg_dict)

    def base_search(self, clf, search_parameters):
        """
        Executes the actual grid search.  Reduces boiler-plate
        in the model-specific functions
        :param clf: <sklearn classifier> the estimator
        :param search_parameters: <dict> the model-specific grid search
            parameters
        :return: the GridSearchCV object
        """
        grid_search_parameters = {
            "estimator": clf,
            "param_grid": search_parameters,
        }
        # Add base parameters
        grid_search_parameters.update(self.base_search_parameters)

        grid_search = GridSearchCV(**grid_search_parameters)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search


    def log_reg_search(self):
        clf = LogisticRegression(
            solver='sag',
        )
        # Default search parameters
        search_parameters = {
            "class_weight": [None, 'balanced'],
            "C": [.01, .1, 1.0, 10],
        }
        grid_search = self.base_search(clf, search_parameters)
        return grid_search

    def adaboost_search(self):
        clf = AdaBoostClassifier()
        search_parameters = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [.1, 1],
        }
        grid_search = self.base_search(clf, search_parameters)
        return grid_search


    def gradboosting_search(self):
        clf = GradientBoostingClassifier(max_features='auto')
        search_parameters = {
            "learning_rate": [.1, 1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5]
        }
        grid_search = self.base_search(clf, search_parameters)
        return grid_search
