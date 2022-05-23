from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # that happens for each parameter Theta (estimator) we receive
    samples = np.column_stack((X, y))  # combine labels to samples
    partition_database = np.array_split(samples, cv,
                                        axis=0)  # split all samples into cv partitions

    validation_score, train_score = [], []
    for i in range(cv):
        fold = partition_database[:]
        del fold[i]   # look on samples without the i'th fold
        fold = np.vstack([fold[i] for i in range(len(fold))])
        train_y = fold[:, -1]  # splits to samples and responses
        train_x = fold[:, :-1]
        validate_x = partition_database[i][:, :-1]  # the i'th fold = test set
        validate_y = partition_database[i][:, -1]

        estimator.fit(train_x, train_y)  # fit a model using training set
        train_score.append(scoring(train_y, estimator.predict(
            train_x)))  # empiric loss - error_i
        validation_score.append(
            scoring(validate_y,
                    estimator.predict(validate_x)))  # generalization loss

    return np.mean(train_score), np.mean(validation_score)
