from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.utils.utils import split_train_test

pio.templates.default = "simple_white"


def keep_last_selling_date(data):
    """
    This function removes duplicates of houses which have the same id and
    keeps only the data of the last selling
    """
    data.sort_values(by='date', ascending=False)
    data.drop_duplicates(subset='id', keep='first', inplace=True)


def remove_rows_containing_invalid_values(data):
    """
    This function clears invalid values.
    """
    data.drop(data[(data.bedrooms <= 0) | (data.bathrooms <= 0) |
                   (data.sqft_living <= 0) | (data.sqft_lot <= 0) |
                   (data.floors <= 0) | (data.condition <= 0) |
                   (data.sqft_above <= 0) |
                   (data.yr_built <= 0) | (data.price <= 0) | (
                           data.sqft_lot15 <= 0) |
                   (data.sqft_living15 <= 0)].index,
              inplace=True)

    data = data[data["grade"].isin(range(1, 14))]
    data = data[data["waterfront"].isin([0, 1])]
    data = data[data["view"].isin(range(0, 5))]
    data = data[data["condition"].isin(range(1, 5))]

    data.dropna(inplace=True)
    return data


def filter_data(data):
    """
    This function is preprocessing the data.
    """
    # clean samples with unrealistic values  - zeros in boxes that should
    # contain values larger than 0
    data = remove_rows_containing_invalid_values(data)

    # normalize the year of built
    min_year_to_remove = data['yr_built'].min()
    data['yr_built'] = data['yr_built'] - min_year_to_remove + 1

    # sort by date and remove duplicates
    keep_last_selling_date(data)

    # encode these categories in what is known as dummy variables or one-hot encoding
    data = pd.get_dummies(data, columns=['zipcode'], prefix='zipcode_')

    # split label vector from samples space
    y = data['price']

    # remove unnecessary columns
    data.drop(labels=['price', 'date', 'id', 'lat', 'long'], axis=1,
              inplace=True)
    return data, y


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = pd.read_csv(filename).drop_duplicates()  # converts csv to dataframe
    return filter_data(X)  # returns clean dataframe


def compute_pearson_correlation(data, response):
    p_by_feature = dict()
    std_of_X = data.std()  # series of std by col (features)
    std_of_y = response.std()
    for feature in data:
        if "zipcode" not in feature:
            p_by_feature[feature] = data[feature].cov(response) \
                                    / (std_of_y * std_of_X[feature])
    return p_by_feature


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # compute pearson correlation
    p_corr = compute_pearson_correlation(X, y)
    for feature in p_corr:
        fig = go.Figure()
        fig.add_traces(
            [go.Scatter(x=list(X[feature]), y=list(y), mode='markers')])
        fig.update_layout(title=fr"Pearson Correlation - {p_corr[feature]}",
                          xaxis_title=fr"feature {feature}",
                          yaxis_title=r"house price")
        fig.show()
        fig.write_image(output_path + fr"/{feature}" + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = r"house_prices.csv"
    X, y = load_data(path)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    w_hat = LinearRegression()
    avg = dict()
    std = dict()
    for i in range(10, 101):
        y_hat_mse = np.empty(10)

        for j in range(10):
            p_x_train, p_y_train, _, _ = split_train_test(train_x,
                                                          train_y,
                                                          i / 100)

            w_hat._fit(p_x_train.to_numpy(), p_y_train.to_numpy())

            y_hat_mse[j] = w_hat._loss(test_x.to_numpy(), test_y.to_numpy())
        avg[f"{i / 100}"] = np.mean(y_hat_mse)
        std[f"{i / 100}"] = np.std(y_hat_mse)

    x_axis = np.array(list(avg.keys()))
    y_axis = np.array(list(avg.values()))
    double_std_np = 2 * np.array(list(std.values()))

    fig_mse = go.Figure()
    fig_mse.add_traces(
        [go.Scatter(x=x_axis, y=(y_axis + double_std_np), mode='lines',
                    marker=dict(color='rgb(204, 221, 255)'),
                    showlegend=True,
                    fillcolor='rgb(204, 221, 255) ',
                    name="confidence interval"),
         go.Scatter(x=x_axis, y=(y_axis - double_std_np), mode='lines',
                    marker=dict(color='rgb(204, 221, 255)'),
                    showlegend=False, fill='tonexty',
                    fillcolor='rgb(204, 221, 255) ',
                    name="confidence interval")])
    fig_mse.add_traces([go.Scatter(x=x_axis, y=y_axis, mode="lines",
                                   marker=dict(color=' rgb(0,48, 153) '),
                                   name="avg mse")])
    fig_mse.update_layout(
        title="Loss as function of training wrapped in confidence interval",
        xaxis=dict(title="Percentage of Training Set"),
        yaxis=dict(title="MSE"))

    fig_mse.show()
