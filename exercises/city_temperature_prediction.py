import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'],
                       dayfirst=True).drop_duplicates()

    # range of possible dates
    data = data[data["Year"] < 2023]
    data = data[data["Month"].isin(range(1, 13))]
    data = data[data["Day"].isin(range(1, 32))]
    data = data[(data["Temp"] < 55) & (data["Temp"] > -25)]

    # day of the year new column
    data['DayOfYear'] = data['Date'].dt.dayofyear

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    path = "/Users/talheldenberg/IML.HUJI/datasets/City_Temperature.csv"
    X = load_data(path)
    y = X['Temp']

    # Question 2 - Exploring data for specific country
    X_israel_database = X[X['Country'] == 'Israel']
    # makes the legend discrete
    X_israel_database = X_israel_database.astype({'Year': str})
    fig_israel = px.scatter(X_israel_database, x="DayOfYear", y="Temp",
                            color="Year",
                            title='Average Daily Temperature in Israel',
                            labels={"DayOfYear": 'Day Of Year',
                                    'Temp': 'Temperature'})
    fig_israel.show()

    # calculates the std of the daily temp every month
    israel_std = X_israel_database.groupby('Month').Temp.agg(['std'])
    fig_std = px.bar(israel_std, x=israel_std.index.to_numpy(), y='std',
                     color_discrete_sequence=(['firebrick']),
                     title='Monthly STD of the daily Temperature',
                     labels={"Month": 'Month', 'std': 'STD of the Daily Temp'})
    fig_std.show()

    # Question 3 - Exploring differences between countries
    mean_std_fulldatabase = X.groupby(['Country', 'Month']).Temp.agg(
        ['mean', 'std']).reset_index()
    fig_line = px.line(mean_std_fulldatabase, x='Month', y='mean',
                       color='Country', error_y='std',
                       title='Monthly Average of the daily Temperature',
                       labels={'Month': 'Month', 'mean': 'Avg'})
    fig_line.show()

    # Question 4 - Fitting model for different values of `k`
    X_israel = X_israel_database['DayOfYear']
    X_israel = X_israel.to_frame()
    y_israel = X_israel_database['Temp']
    train_il_X, train_il_y, test_il_X, test_il_y = split_train_test(X_israel,
                                                                    y_israel)
    models = dict()
    loss = dict()
    for k in range(1, 11):
        w_hat = PolynomialFitting(k)
        models[k] = w_hat.fit(train_il_X['DayOfYear'].to_numpy(),
                              train_il_y.to_numpy())
        loss[k] = round(
            w_hat._loss(test_il_X['DayOfYear'].to_numpy(),
                        test_il_y.to_numpy()), 2)
        print(f"loss when k is {k} : {loss[k]}")
    loss_pd = pd.DataFrame.from_dict(
        {'k': [i for i in range(1, 11)], 'loss': list(loss.values())})
    fig_il_fit_loss = px.bar(loss_pd, x='k', y='loss',
                             color_discrete_sequence=(['firebrick']),
                             title='IL Mse in Relation to Polynomial Degree',
                             labels={"k": 'k = degree',
                                     'std': 'loss'})
    fig_il_fit_loss.show()

    # Question 5 - Evaluating fitted model on different countries
    min_error_for_il = int(loss_pd[['loss']].idxmin())
    best_k_il = loss_pd.iat[min_error_for_il, 0]
    model_il = PolynomialFitting(best_k_il)
    model_il._fit(X_israel['DayOfYear'].to_numpy(), y_israel.to_numpy())
    countries_error = dict()
    for country in X.Country.unique():
        if country == 'Israel':
            continue
        c_data = X[X.Country == country]
        countries_error[country] = model_il._loss(c_data['DayOfYear'],
                                                  c_data['Temp'])
    countries_error = pd.DataFrame.from_dict(
        {'country': list(countries_error.keys()),
         'loss': list(countries_error.values())})
    fig_countries_loss = px.bar(countries_error, x='country', y='loss',
                                color_discrete_sequence=(['firebrick']),
                                title='Loss of Countries Fit by IL Model',
                                labels={"country": 'Country',
                                        'loss': 'Loss'})
    fig_countries_loss.show()
