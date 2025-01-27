import pandas as pd
import yfinance as yf

def load_historical_data(tickers, start_date, end_date):
    """
    Fetch historical data using Yahoo Finance API.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.ffill().bfill()  # Handle missing values
    return data

# Example usage:
# data = load_historical_data(['SPY', 'QQQ', 'TLT'], '2017-01-01', '2022-12-31')