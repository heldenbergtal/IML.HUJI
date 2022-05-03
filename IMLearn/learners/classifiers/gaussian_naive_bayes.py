from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.array([])
        m = len(y)
        for k in range(len(self.classes_)):
            xk = X[y == k]
            self.pi_ = np.append(self.pi_, xk.shape[0] / m)
            mu_k = np.mean(xk, axis=0)
            if self.mu_ is None:
                self.mu_ = mu_k
            else:
                self.mu_ = np.vstack((self.mu_, mu_k))
            var = np.var(xk, ddof=1, axis=0)
            if self.vars_ is None:
                self.vars_ = var
            else:
                self.vars_ = np.vstack((self.vars_, var))

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
        d = X.shape[1]
        likelihoods = None
        for sample in X:
            each_label_likelihood = np.array([])
            for k in range(len(self.classes_)):
                var_mat = np.diag(self.vars_[int(k)])
                compute = ((sample - self.mu_[int(k)]).transpose()) @ inv(
                    var_mat) @ (sample - self.mu_[int(k)])
                each_label_likelihood = np.append(each_label_likelihood,
                                                  ((1 / np.sqrt(
                                                      ((2 * np.pi) ** d) * det(
                                                          var_mat))) * np.exp(
                                                      -0.5 * compute)) *
                                                  self.pi_[int(k)])
            likelihoods = each_label_likelihood if likelihoods is None else np.vstack(
                (likelihoods, each_label_likelihood))
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
