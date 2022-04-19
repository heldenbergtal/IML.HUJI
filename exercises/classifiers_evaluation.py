from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2
from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, 0:2], data[:, 2]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/{}".format(f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda perceptron, xi, yi: losses.append(
            perceptron.loss(X, y))).fit(X, y)

        # Plot figure
        x_axis = np.array([i for i in range(len(losses))])
        fig_preceptron = go.Figure()
        fig_preceptron.add_traces(
            [go.Scatter(x=x_axis, y=losses, mode='lines')])
        fig_preceptron.update_layout(
            title=rf"Perceptron Algorithm's Training Loss Values of {n}"
                  r" as sa Function of Training Iterations",
            xaxis_title="iterations", yaxis_title="loss", height=800)
        fig_preceptron.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    for n, f in [("gaussian1", "gaussian1.npy"),
                 ("gaussian2", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/{}".format(f))

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        y_gnb_pred = gnb.predict(X)

        lda = LDA()
        lda.fit(X, y)
        y_lda_pred = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        plot_models_scatter(X, y, y_gnb_pred, y_lda_pred, gnb.mu_, gnb.vars_,
                            lda.mu_, lda.cov_, n)


def plot_models_scatter(X, y, gnb_pred, lda_pred, gnb_mean,
                        gnb_var, lda_mean, lda_var, n):
    models = ["Gaussian Naive Bayes", "LDA"]
    fig_class = make_subplots(rows=1, cols=2,
                              subplot_titles=([
                                  rf"Model: {models[0]}, Accuracy: {accuracy(y, gnb_pred)}",
                                  rf"Model: {models[1]}, Accuracy: {accuracy(y, lda_pred)}"]))
    fig_class.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
            color=gnb_pred, symbol=y, size=7)),
        row=1, col=1)
    fig_class.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
            color=lda_pred, symbol=y, size=7)), row=1, col=2)
    create_ellipse_shapes(fig_class, gnb_mean, gnb_var, lda_mean, lda_var)
    fig_class.update_layout(showlegend=False, height=750, width=1300,
                            title_text=rf"Classifiers Results on {n} Database")
    fig_class.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipsis centered in Gaussian centers and shape dictated by fitted
    covariance matrix.
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
    scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * np.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def create_ellipse_shapes(fig, gnb_mean, gnb_var, lda_mean, lda_var):
    for i in range(3):
        fig.add_trace(get_ellipse(gnb_mean[i], np.diag(gnb_var[i])), row=1,
                      col=1)
        fig.add_trace(get_ellipse(lda_mean[i], lda_var), row=1, col=2)
        fig.add_trace(create_x_mark(gnb_mean[i]), row=1, col=1)
        fig.add_trace(create_x_mark(lda_mean[i]), row=1, col=2)


def create_x_mark(mu):
    """
    Drae a cross indicating the center of fitted gaussian
    ----------
    mu : ndarray of shape (2,)
        Center of samples
    Returns
    -------
        scatter: A plotly trace object of the X mark
    """
    return go.Scatter(x=[mu[0]], y=[mu[1]], mode='markers',
                      marker=dict(color='black', symbol='cross', size=5), showlegend=False)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
