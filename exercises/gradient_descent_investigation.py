import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2, RegularizedModule, \
    LogisticModule
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import mean_square_error

import plotly.express as px
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weights = list()

    def append_to_lists(inst, l):
        values.append(l[1])
        weights.append(l[0])

    return append_to_lists, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        l1 = L1(init.copy())
        l2 = L2(init.copy())

        callback1, v1, w1 = get_gd_state_recorder_callback()
        callback2, v2, w2 = get_gd_state_recorder_callback()

        w1.append(init)
        w2.append(init)

        gd_l1 = GradientDescent(FixedLR(eta),
                                callback=callback1).fit(l1, None, None)
        gd_l2 = GradientDescent(FixedLR(eta),
                                callback=callback2).fit(l2, None, None)

        plot_descent_path(L1, np.array(w1),
                          title=f"l1 gd using fixed eta {eta}").show()
        plot_descent_path(L2, np.array(w2),
                          title=f"l2 gd fixed eta {eta}").show()

        q_3_1 = go.Figure()
        q_3_2 = go.Figure()

        q_3_1.add_trace(go.Scatter(x=[i for i in range(1, len(v1) + 1)], y=v1,
                                   mode='lines',
                                   name='l1 norm as function of gd iteration'))
        q_3_1.update_layout(
            title=f"l1 norm as function of gd iteration using eta {eta}",
            xaxis_title="iterations",
            yaxis_title="norm").show()

        q_3_2.add_trace(go.Scatter(x=[i for i in range(1, len(v2) + 1)], y=v2,
                                   mode='lines',
                                   name='l2 norm as function of gd iteration'))
        q_3_2.update_layout(
            title=f"l2 norm as function of gd iteration using eta {eta}",
            xaxis_title="iterations",
            yaxis_title="norm").show()


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    q_5 = go.Figure()
    for gamma in gammas:
        l1 = L1(init.copy())
        callback1, v1, w1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(ExponentialLR(eta, gamma),
                                callback=callback1)
        gd_l1.fit(l1, None, None)
        q_5.add_trace(
            go.Scatter(x=list(range(1, 1001)), y=v1, name=f"{gamma}"))

    q_5.update_layout(title="convergence rate as function of iterations",
                      xaxis_title="iterations",
                      yaxis_title="norm")
    q_5.show()

    # Plot algorithm's convergence for the different values of gamma

    # Plot descent path for gamma=0.95
    gamma = 0.95
    l1 = L1(init.copy())
    l2 = L2(init.copy())

    callback1, v1, w1 = get_gd_state_recorder_callback()
    callback2, v2, w2 = get_gd_state_recorder_callback()

    w1.append(init)
    w2.append(init)

    gd_l1 = GradientDescent(ExponentialLR(eta, gamma),
                            callback=callback1).fit(l1, None, None)
    gd_l2 = GradientDescent(ExponentialLR(eta, gamma),
                            callback=callback2).fit(l2, None, None)

    plot_descent_path(L1, np.array(w1),
                      title=f"l1 gd using exponential eta").show()
    plot_descent_path(L2, np.array(w2),
                      title=f"l2 gd exponential eta ").show()


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data

    logistic_model = LogisticRegression(solver=GradientDescent
    (callback=get_gd_state_recorder_callback()[0])).fit(np.array(X_train),
                                                        np.array(y_train))
    y_score = logistic_model.predict_proba(np.array(X_train))
    fpr, tpr, thresholds = roc_curve(y_train, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()

    a_max = np.argmax(tpr - fpr)
    print(thresholds[a_max])

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lamds = np.linspace(0, 1, 10000)
    tr, vr = [], []
    for lam in lamds:
        rgl1 = LogisticRegression(solver=GradientDescent
        (callback=get_gd_state_recorder_callback()[0]), penalty="l1",
                                  alpha=0.5, lam=lam)
        t, v = cross_validate(rgl1, np.array(X_train), np.array(y_train),
                            mean_square_error)
        tr.append(t)
        vr.append(v)
    best_lam = lamds[np.argmin(vr)]
    rgl1 = LogisticRegression(solver=GradientDescent
    (callback=get_gd_state_recorder_callback()[0]), penalty="l1",
                              alpha=0.5, lam=best_lam).fit(np.array(X_train), np.array(y_train))


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
