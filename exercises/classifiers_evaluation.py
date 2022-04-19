import pandas as pd

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
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
    models = ["LDA", "Gaussian Naive Bayes"]

    for n, f in [("gaussian1", "gaussian1.npy"),
                 ("gaussian2", "gaussian2.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/{}".format(f))

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        y_lda_pred = lda.predict(X)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        y_gnb_pred = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            rf"Model: {models[0]}, Accuracy: {accuracy(y, y_lda_pred)}",
            rf"Model: {models[1]}, Accuracy: {accuracy(y, y_gnb_pred)}"])
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
                color=y_lda_pred, symbol=y), showlegend=False),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(
                color=y_gnb_pred, symbol=y), showlegend=True),
            row=1, col=2
        )

        fig.update_layout(height=900, width=1500,
                          title_text=fr"Classifiers Results on {n} Database")

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
