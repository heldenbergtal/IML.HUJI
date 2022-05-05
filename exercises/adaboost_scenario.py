import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(wl=lambda: DecisionStump(), iterations=250)
    ada_boost._fit(train_X, train_y)
    training_error = []
    test_error = []
    for T in range(1, n_learners + 1):
        training_error.append(ada_boost.partial_loss(train_X, train_y, T))
        test_error.append(ada_boost.partial_loss(test_X, test_y, T))

    fig_q1 = go.Figure()
    x_axis = [i for i in range(1, n_learners + 1)]
    fig_q1.add_trace(go.Scatter(x=x_axis, y=training_error,
                                mode='markers+lines',
                                name='training error'))
    fig_q1.add_trace(go.Scatter(x=x_axis, y=test_error,
                                mode='markers+lines',
                                name='test errors'))
    fig_q1.update_layout(
        title="Errors as a Function of the Number Of Fitted Leareners",
        xaxis_title="number of learners",
        yaxis_title="misclassification errors", width=1000, height=600)
    fig_q1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    print(lims[0])
    for t in T:
        predict = ada_boost.partial_predict(test_X, t)
        decision_surface(predict=predict, xrange=lims[0], yrange=lims[1]).show()

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0, 3, 100, 25)
