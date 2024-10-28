# Please, compare and analyze results. Add conclusions as comments here or to a readme file.

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_diabetes


def preprocess(X: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """
    Preprocesses the input data by scaling features and splitting into training and test sets.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    raise NotImplementedError


def get_regression_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the diabetes dataset for regression tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    raise NotImplementedError


def get_classification_data() -> list[np.ndarray]:
    """
    Loads and preprocesses the breast cancer dataset for classification tasks.

    Returns:
        list[np.ndarray]: List containing training and test sets for features and target.
    """
    raise NotImplementedError


def linear_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a linear regression model on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained linear regression model.
    """
    raise NotImplementedError


def ridge_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a ridge regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best ridge regression model found by GridSearchCV.
    """
    raise NotImplementedError


def lasso_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a lasso regression model with hyperparameter tuning using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best lasso regression model found by GridSearchCV.
    """
    raise NotImplementedError


def logistic_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model without regularization on the given data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Trained logistic regression model.
    """
    raise NotImplementedError


def logistic_l2_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L2 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L2 regularization found by GridSearchCV.
    """
    raise NotImplementedError


def logistic_l1_regression(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """
    Trains a logistic regression model with L1 regularization using GridSearchCV.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        BaseEstimator: Best logistic regression model with L1 regularization found by GridSearchCV.
    """
    raise NotImplementedError
