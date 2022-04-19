from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.array([])
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        m = len(y)
        for k in self.classes_:
            nk = np.count_nonzero(y == k)
            self.pi_ = np.append(self.pi_, nk / m)
            idx = np.where(y == k)[0]
            xk = np.array(X[idx])
            mu_k = np.mean(xk, axis=0)
            self.mu_ = mu_k if self.mu_ is None else np.vstack((self.mu_, mu_k))
            cov = np.zeros((X.shape[1], X.shape[1]))
            for xi in xk:
                cov += np.outer(xi - mu_k, xi - mu_k)
            self.cov_ += cov
        self.cov_ /= (m - len(self.classes_))
        self._cov_inv = inv(self.cov_)

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
        likelihoods = self.likelihood(X)
        y_pred = np.array([])
        for sample in likelihoods:
            label = np.argmax(sample)
            y_pred = np.append(y_pred, label)
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        det_cov = np.linalg.det(self.cov_)
        d = X.shape[1]
        likelihoods = None
        for sample in X:
            each_label_likelihood = np.array([])
            for k in self.classes_:
                compute = ((sample - self.mu_[int(k)]).transpose()) @ self._cov_inv @ (sample - self.mu_[int(k)])
                each_label_likelihood = np.append(each_label_likelihood,
                                                  ((1/np.sqrt(((2*np.pi)**d) * det_cov))*np.exp(-0.5 * compute)) * self.pi_[int(k)])
            if likelihoods is None:
                likelihoods = each_label_likelihood
            else:
                likelihoods = np.vstack((likelihoods, each_label_likelihood))
        return likelihoods




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
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
