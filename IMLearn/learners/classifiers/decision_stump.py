from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is
        about the threshold
    """
    SIGN = [-1, 1]

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        class_errors = []  # keep all errors
        best = np.infty
        for j in range(X.shape[1]):
            for sign in self.SIGN:
                threshold, thresh_err = self._find_threshold(X[:, j], y, sign)
                # class_errors.append((threshold, sign, j,
                #                      thresh_err))
                if (thresh_err < best):
                    self.threshold_, self.sign_, self.j_, best = threshold, sign, j, thresh_err
        # self.threshold_, self.sign_, self.j_, _ = min(class_errors,
        #                                               key=lambda err: err[
        #                                                   -1])

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array(
            [self.sign_ if X[i][self.j_] >= self.threshold_ else -self.sign_
             for i in range(X.shape[0])])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_val = values[values.argsort()]  # sorts array in ascending order
        class_err = []  # keeps all errors

        for thresh in sorted_val:
            y_pred = np.array(
                [sign if values[j] >= thresh else -sign for j in
                 range(len(values))])
            weighted_misclassification_error = self._weighted_misclassification_error(
                labels, y_pred)
            class_err.append((thresh, weighted_misclassification_error))
        return min(class_err, key=lambda err: err[1])  # return the threshold that causes the min error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))

    def _weighted_misclassification_error(self, y_true, y_pred):
        return np.sum(
            np.abs(y_true * (np.sign(y_true) != np.sign(y_pred)))) / len(
            y_true)
