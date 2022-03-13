from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

X = np.random.normal(10, 1, 1000)  


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    # m = 1000
    global X

    fitted_X = UnivariateGaussian()
    fitted_X.fit(X)
    print(f"({fitted_X.mu_}, {fitted_X.var_})")

    fig = make_subplots(rows=5, cols=1, specs=[[{"rowspan": 4, "secondary_y": True}], [None], [None], [None], [{}]]) \
        .add_trace(go.Histogram(x=X, opacity=0.75, bingroup=1, nbinsx=200, marker_color='navy'), secondary_y=False)
    fig.update_layout(title_text="$\\text{ Histograms of Sample }X\\sim\\mathcal{N}\\left(10, 1\\right)$") \
        .update_yaxes(title_text="Number of occurrences", secondary_y=False, row=1, col=1) \
        .update_xaxes(title_text="Values", row=1, col=1)
    fig.show()

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 200).astype(int)
    estimated_mean = []
    fitted_X = UnivariateGaussian()
    for m in ms:
        fitted_X.fit(X[:m])
        distance = np.absolute(10 - fitted_X.mu_)
        estimated_mean.append(distance)

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$distance$')],
              layout=go.Layout(title=r"$\text{Absolute Error of the Estimator as Function of Number of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$distance$",
                               height=400)).show()



    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
