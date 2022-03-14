from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    num_of_samples = 1000
    X = np.random.normal(10, 1, num_of_samples)

    fitted_1 = UnivariateGaussian()
    fitted_1.fit(X)
    print(f"({fitted_1.mu_}, {fitted_1.var_})")

    # Question 2 - Empirically showing sample mean is consistent

    ms = np.arange(10, 1001, step=10).astype(int)
    fitted_2 = UnivariateGaussian()
    errors = []
    for m in ms:
        fitted_2.fit(X[:m])
        distance = np.absolute(10 - fitted_2.mu_)
        errors.append(distance)

    go.Figure([go.Scatter(x=ms, y=errors, mode='markers+lines', name=r'$distance$')],
              layout=go.Layout(title=r"$\text{Absolute Error of the Estimator as Function of Number of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$distance$",
                               height=400)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_1 = fitted_1.pdf(X)

    make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=X, y=pdf_1, mode='markers', marker=dict(color="red"), showlegend=False)]) \
        .update_layout(title_text=r"Empirical PDF Function under the Q1 Fitted Model",
                       xaxis_title=r"samples", yaxis_title=r"pdf values").show()


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model

    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = 1000
    X = np.random.multivariate_normal(mean, cov, samples)
    fitted_4 = MultivariateGaussian()
    fitted_4.fit(X)
    print(f"estimates expectation: {fitted_4.mu_}\nestimated covariance matrix:\n{fitted_4.cov_}")

    # Question 5 - Likelihood evaluation

    ms = np.linspace(-10, 10, 200)
    log_likelihood = []
    fitted_5 = MultivariateGaussian()
    for f1 in ms:
        sub_arr = []
        for f3 in ms:
            sub_arr.append(fitted_5.log_likelihood(np.array([f1, 0, f3, 0]), cov, X))
        log_likelihood.append(sub_arr)

    go.Figure(go.Heatmap(x=ms, y=ms, z=log_likelihood)).update_layout(title="Heatmap of Log-Likelihood Estimation"
                                                                      , xaxis_title="f1", yaxis_title="f3").show()

    # Question 6 - Maximum likelihood

    max_likelihood_value = np.amax(log_likelihood)
    result = np.where(log_likelihood == max_likelihood_value)
    print(f"f1: {round(ms[result[0]][0], 3)}, f3: {round(ms[result[1]][0], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
