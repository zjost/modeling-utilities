import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def add_dummies(df, column_names, drop_first = False):
    """
    This function returns a new df with the original
    columns dropped and the dummies added

    Args:

        df (pd.DataFrame): The source dataframe
        column_names (list): A list of column names in df
            to be converted to dummies

    Output:
        df_copy (pd.DataFrame): The resulting dataframe
    """
    df_copy = df.copy()
    for column in column_names:
        dummies = pd.get_dummies(df[column], drop_first=drop_first)
        df_copy = df_copy.merge(dummies, left_index=True, right_index=True)
    return df_copy.drop(column_names, axis=1)


def describe(df):
    """
    Intended to return basic things, like number of missing...etc.
    :param df:
    :return:
    """
    pass


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


def get_sample_weights(y):
    """
    A helper function to calculate sample weights based
    on class frequencies.

    Args:
        y (pd.DataFrame): The samples to weight.  Assumes
            encoded as 0 and 1.

    Returns:
        sample_weights (np.array): An array of weights
            equal to inverse of class proportion

    """
    N = float(len(y))
    weight_0 = N / (y == 0).sum()
    weight_1 = N / (y == 1).sum()
    sample_weights = np.zeros((N,), dtype=np.float64)
    sample_weights[(y == 0).values] = weight_0
    sample_weights[(y == 1).values] = weight_1
    return sample_weights
