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


def convert_grade_to_categorical(data):
    data.loc[(data['grade'] > 0) & (data['grade'] < 4), 'grade'] = 1
    data.loc[(data['grade'] > 3) & (data['grade'] < 8), 'grade'] = 2
    data.loc[(data['grade'] > 7) & (data['grade'] < 11), 'grade'] = 3
    data.loc[(data['grade'] > 10) & (data['grade'] < 14), 'grade'] = 4


def keep_last_selling_date(data):
    data.sort_values(by='date', ascending=False)
    data.drop_duplicates(subset='id', keep='first', inplace=True)


def compute_relative_living_and_lot_size(data):
    data['sqft_living15'] = data['sqft_living'] / data['sqft_living15']
    data['sqft_lot15'] = data['sqft_lot'] / data['sqft_lot15']
    data.rename(columns={'sqft_living15': 'relative_living_to_area',
                         'sqft_lot15': 'relative_lot_to_area'}, inplace=True)


def filter_data(data):
    # clean samples with unrealistic values
    data = data.drop(data[(data.bedrooms <= 0) | (data.bathrooms <= 0) |
                          (data.sqft_living <= 0) | (data.sqft_lot <= 0) |
                          (data.floors <= 0) | (data.condition <= 0) |
                          (data.grade <= 0) | (data.sqft_above <= 0) |
                          (data.yr_built <= 0) | (
                                  data.price <= 0)].index)

    # categorize grade
    convert_grade_to_categorical(data)

    # sort by date and remove duplicates
    keep_last_selling_date(data)

    # change the sqft_living to the relative size to the area
    compute_relative_living_and_lot_size(data)

    # split label vector from samples space
    y = data['price']

    # remove unnecessary columns
    data = data.drop(labels=['price', 'date', 'id', 'zipcode'], axis=1)
    data.dropna()

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
    X = pd.read_csv(filename)  # converts csv to datafram
    return filter_data(X)


def compute_pearson_correlation(data, response):
    p_by_feature = dict()
    std_of_X = data.std()  # series of std by col (features)
    std_of_y = response.std()
    for i in range(np.shape(X)[1]):
        p_by_feature[data.columns.values[i]] = data.iloc[:, i].cov(
            response) / (std_of_y * std_of_X[i])
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
    # for feature in p_corr:
    #     # go.Figure([go.Scatter(x=list(X[feature]), y=list(y), mode='markers')],
    #     #           layout=go.Layout(
    #     #               title=fr"Pearson Correlation - {p_corr[feature]}",
    #     #               xaxis_title=fr"feature {feature}",
    #     #               yaxis_title=r"house price")).show()
    #     fig = go.Figure()
    #     fig.add_traces(
    #         [go.Scatter(x=list(X[feature]), y=list(y), mode='markers')])
    #     fig.update_layout(title=fr"Pearson Correlation - {p_corr[feature]}",
    #                       xaxis_title=fr"feature {feature}",
    #                       yaxis_title=r"house price")
    #     fig.show()
    #     fig.write_image(output_path + fr"/{feature}" + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    path = r"/Users/talheldenberg/IML.HUJI/datasets/house_prices.csv"
    X, y = load_data(path)
    feature_evaluation(X, y, "/Users/talheldenberg/Desktop/try1")
    train_x, train_y, test_x, test_y = split_train_test(X, y , 0.75)


    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
