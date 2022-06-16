from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression, LassoRegression
from sklearn.linear_model import Lasso
from IMLearn.metrics.loss_functions import mean_square_error

from IMLearn.desent_methods.gradient_descent import GradientDescent

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MIN_X = -1.2
MAX_X = 2
NORMAL_MEAN = 0


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    train_proportion = 2 / 3
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(start=MIN_X, stop=MAX_X, num=n_samples)
    noiseless_y = f(X)
    noised_y = f(X) + np.random.normal(loc=NORMAL_MEAN, scale=noise,
                                       size=n_samples)

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(noised_y),
                                                        train_proportion)
    train_x, train_y, test_x, test_y = train_x.to_numpy().reshape(
        len(train_x)), \
                                       train_y.to_numpy().reshape(
                                           len(train_y)), test_x.to_numpy().reshape(
        len(test_x)), \
                                       test_y.to_numpy().reshape(len(test_y))

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=X, y=noiseless_y, mode="markers+lines", name="real"))
    fig1.add_trace(go.Scatter(x=test_x, y=test_y,
                              mode="markers", marker=dict(color='#ff6361'),
                              name="test"))
    fig1.add_trace(go.Scatter(x=train_x, y=train_y,
                              mode="markers", marker=dict(color='#58508d'),
                              name="train"))
    fig1.update_layout(title="Noise vs. Noiseless", xaxis_title="x",
                       yaxis_title="f(x)", height=700, width=1000)
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    polynomial_degrees = 11
    validation_error, training_error = [], []
    for deg in range(polynomial_degrees):
        ts, vs = cross_validate(PolynomialFitting(deg), train_x, train_y,
                                mean_square_error)
        validation_error.append(vs)
        training_error.append(ts)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[i for i in range(polynomial_degrees)],
                              y=validation_error,
                              mode="markers", marker=dict(color='#ff6361'),
                              name="validation"))
    fig2.add_trace(go.Scatter(x=[i for i in range(polynomial_degrees)],
                              y=training_error,
                              mode="markers", marker=dict(color='#58508d'),
                              name="training"))
    fig2.update_layout(
        title="Mean Error as Function of Polynom Degree",
        xaxis_title="degree",
        yaxis_title="validation error", height=700, width=1000)
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_error)
    print("k-star: ", k_star)
    p_reg = PolynomialFitting(k_star).fit(train_x, train_y)
    loss = p_reg.loss(test_x, test_y)
    print("loss on test set: ", loss)


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    train_x, train_y, test_x, test_y = X[:50], y[:50], X[50:], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lamds_range = np.linspace(0, 1, n_evaluations)
    validation_error_lasso, validation_error_ridge = [], []
    training_error_lasso, training_error_ridge = [], []
    for lam in lamds_range:
        lasso = sklearn.linear_model.Lasso(lam)
        ridge = RidgeRegression(lam)
        ts_lasso, vs_lasso = cross_validate(lasso, train_x, train_y,
                                            mean_square_error)
        ts_ridge, vs_ridge = cross_validate(ridge, train_x, train_y,
                                            mean_square_error)
        validation_error_ridge.append(vs_ridge)
        training_error_ridge.append(ts_ridge)
        validation_error_lasso.append(vs_lasso)
        training_error_lasso.append(ts_lasso)

    fig7 = make_subplots(rows=1, cols=2, shared_xaxes=True,
                         subplot_titles=("Lasso", "Ridge"))
    fig7.add_trace(go.Scatter(x=lamds_range,
                              y=validation_error_lasso,
                              mode="markers+lines",
                              marker=dict(color='#ff6361'), name="validation"),
                   row=1, col=1)
    fig7.add_trace(go.Scatter(x=lamds_range,
                              y=training_error_lasso,
                              mode="markers+lines",
                              marker=dict(color='#003f5c'), name="training"),
                   row=1, col=1)
    fig7.add_trace(go.Scatter(x=lamds_range,
                              y=validation_error_ridge,
                              mode="markers+lines",
                              marker=dict(color='#ff6361'), name="validation"),
                   row=1, col=2)
    fig7.add_trace(go.Scatter(x=lamds_range,
                              y=training_error_ridge,
                              mode="markers+lines",
                              marker=dict(color='#003f5c'), name="training"),
                   row=1, col=2)
    fig7.update_layout(
        title="Scoring in Different Models",
        xaxis_title="lambda",
        yaxis_title="scoring")
    fig7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = lamds_range[np.argmin(validation_error_ridge)]
    best_lasso = lamds_range[np.argmin(validation_error_lasso)]
    print(f"regularization best: ridge-{best_ridge}, lasso-{best_lasso}")

    lasso = sklearn.linear_model.Lasso(best_lasso)
    ridge = RidgeRegression(lam=best_ridge)
    linear = LinearRegression()

    lasso.fit(train_x, train_y)
    ridge.fit(train_x, train_y)
    linear.fit(train_x, train_y)

    lasso_loss = mean_square_error(lasso.predict(test_x), test_y)
    ridge_loss = mean_square_error(ridge.predict(test_x), test_y)
    linear_loss = linear.loss(test_x, test_y)
    print(
        f"Models errors: ridge-{ridge_loss}, lasso-{lasso_loss}, linear-{linear_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
