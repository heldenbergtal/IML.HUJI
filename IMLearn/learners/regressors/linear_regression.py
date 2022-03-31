from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        #my notes - We want to minimize the square error -
        the principle of learning is ERM
        The algorithm used to minimize RSS is by SVM decomposition

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # when there is an intercept included, to make the equation linear
        # (not affine) we add a zero-th coordinate with value 1
        if self.include_intercept_:
            X = np.insert(X, 0, np.ones(np.shape(X)[0], axis=1))
        # pinv X- Compute the (Moore-Penrose) pseudo-inverse of a matrix.
        # Calculate the generalized inverse of a matrix using its
        # singular-value decomposition (SVD) and including all
        # large singular values.
        self.coefs_ = np.linalg.pinv(X) @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        # we get samples and using the model we found (w hat) the result
        # is calculated
        # checks the intercept so the dimensions of the multiplication would
        # be ok
        if self.include_intercept_:
            X = np.insert(X, 0, np.ones(np.shape(X)[0], axis=1))
        # X multiply w hat = y
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # using the function I have implemented
        return mean_square_error(self.predict(X), y)
