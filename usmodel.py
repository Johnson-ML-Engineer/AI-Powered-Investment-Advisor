import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

def download_stocks(stock_names, start_date, end_date):
    data = yf.download(stock_names, start_date, end_date)
    return data["Close"]

def check_stationarity(data, max_diff=5, significance_level=0.05):
    p_value = adfuller(data)[1]

    if p_value > significance_level:
        print(f'Time series is non-stationary (p-value: {p_value:.4f}). Performing differencing...')

        for diff in range(1, max_diff + 1):
            differenced_data = data.diff(diff).dropna()
            p_value_diff = adfuller(differenced_data)[1]

            if p_value_diff <= significance_level:
                print(f'Differencing order {diff} makes the time series stationary (p-value: {p_value_diff:.4f}).')
                return differenced_data

        print(f'Differencing up to order {max_diff} did not achieve stationarity. Consider other transformations.')
        return None

    else:
        print(f'Time series is already stationary (p-value: {p_value:.4f}).')
        return data

def prepare_data_for_var(data, feature_column, look_back):
    dataset = data[feature_column].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    x, y = [], []
    for i in range(len(dataset) - look_back):
        x.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])

    x, y = np.array(x), np.array(y)
    return x, y, scaler

def fit_var_model(train_df, order):
    model = VAR(train_df)
    model_fitted = model.fit(order)
    return model_fitted

def forecast_var_model(model_fitted, forecast_input, n_steps):
    forecast = model_fitted.forecast(y=forecast_input, steps=n_steps)
    return forecast

def inverse_transform(predicted_prices_all, all_scaler, yahoo_finance_symbols):
    inverted_prices = {}

    for idx, feature_column in enumerate(yahoo_finance_symbols, start=1):
        predicted_prices = all_scaler[feature_column].inverse_transform(
            predicted_prices_all[:, idx - 1].reshape(-1, 1)
        )
        inverted_prices[feature_column] = predicted_prices.flatten()

    return pd.DataFrame(inverted_prices)

def roll_back_differences(original_data, forecasted_diff, diff_order):
    cumsum_values = original_data[-diff_order:].cumsum()

    if len(forecasted_diff) < len(cumsum_values):
        forecasted_diff = np.concatenate((np.zeros(len(cumsum_values) - len(forecasted_diff)), forecasted_diff))

    final_forecast = original_data[-1] + np.cumsum(forecasted_diff)
    return final_forecast

def main():
    company_name = ['Microsoft', 'Amazon', 'JPMorgan Chase', 'Home Depot', 'Adobe', 'Thermo Fisher Scientific',
                    'Abbott Laboratories', 'Intuit', 'Danaher', 'Texas Instruments']

    yahoo_finance_symbols = ['MSFT', 'AMZN', 'JPM', 'HD', 'ADBE', 'TMO', 'ABT', 'INTU', 'DHR', 'TXN']

    # Download training data
    start_date_train = "1998-01-01"
    end_date_train = date.today().strftime("%Y-%m-%d")
    train_data = download_stocks(yahoo_finance_symbols, start_date_train, end_date_train)
    train_data = train_data.fillna(method='ffill').fillna(method='bfill')

    # Loop through each stock, check stationarity, and apply differencing
    stationary_data = {}
    for i, symbol in enumerate(yahoo_finance_symbols):
        print(f"\nChecking stationarity for {company_name[i]} ({symbol})...")

        # Extract closing prices
        closing_prices = train_data[symbol]

        # Check and perform differencing if needed
        stationary_data[symbol] = check_stationarity(closing_prices)

    # Prepare data for VAR model
    look_back = 50
    data_x, data_y, all_scaler = {}, {}, {}

    for feature_column in yahoo_finance_symbols:
        x, y, scaler = prepare_data_for_var(stationary_data, feature_column, look_back)
        data_x[feature_column] = x
        data_y[feature_column] = y
        all_scaler[feature_column] = scaler

    # Combine data into a DataFrame
    train_df = pd.DataFrame(data_y, columns=yahoo_finance_symbols)

    # Fit VAR model
    lag_order = 74
    model_fitted = fit_var_model(train_df, lag_order)

    # Forecasting for test data
    forecast_input = train_df.values[-lag_order:]
    n_steps = 150
    forecast = forecast_var_model(model_fitted, forecast_input, n_steps)

    # Inverse transform the predicted values
    predicted_prices_all = inverse_transform(forecast, all_scaler, yahoo_finance_symbols)

    # Roll back differences and obtain final forecasted prices
    final_forecast_all = {}

    for symbol in yahoo_finance_symbols:
        print(f"\nRolling back differences for {symbol}...")

        # Get the original closing prices
        original_prices = train_data[symbol].values

        # Get the forecasted differences
        forecasted_diff = predicted_prices_all[symbol].values

        # Roll back differences using the function
        final_forecast = roll_back_differences(original_prices, forecasted_diff, look_back)

        # Store the final forecast in a dictionary
        final_forecast_all[symbol] = final_forecast

    # Combine final forecast into a DataFrame
    final_forecast_df = pd.DataFrame(final_forecast_all)
    
    business_days = pd.date_range(start=date.today().strftime("%Y-%m-%d"),periods=150,freq="B")
    final_forecast_df.set_index(business_days,inplace=True)
    train_data = pd.concat([train_data, final_forecast_df.iloc[[0]]])
    
    return final_forecast_df ,train_data




