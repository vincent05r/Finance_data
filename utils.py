import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
import csv

def download_and_save_data(ticker, interval, start_date, end_date, save_dir:str, sleep_time:int = 2, drop_last: bool = True, drop_NA:bool=True):
    try:
        c_ticker = yf.Ticker(ticker)
        data = c_ticker.history(start=start_date, end=end_date, interval=interval)
        if not data.empty:
            # Create directory for the index and interval if it doesn't exist
            dir_path = os.path.join(save_dir, interval)
            os.makedirs(dir_path, exist_ok=True)
            # Save to CSV
            file_path = os.path.join(dir_path, f"{ticker}.csv")
            #process df
            if drop_last:
                list_axis = ['Dividends','Stock Splits']
                data.drop(list_axis, axis=1, inplace=True)
            if drop_NA:
                data.dropna(inplace=True)

            data.to_csv(file_path)
            print(f"Data for {ticker} saved successfully.")
        else:
            print(f"No data found for {ticker}.")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
    # Sleep to respect rate limits
    sleep(sleep_time)


def download_and_save_data_period(ticker, interval, period, save_dir:str, sleep_time:int = 2, drop_last: bool = True, drop_NA:bool=True):
    '''
    period: current 8d for 1m interval, 730d for 1h interval max
    '''

    rt_string = None

    try:
        c_ticker = yf.Ticker(ticker)
        data = c_ticker.history(period=period, interval=interval)
        if not data.empty:
            # Create directory for the index and interval if it doesn't exist
            dir_path = os.path.join(save_dir, interval)
            os.makedirs(dir_path, exist_ok=True)
            # Save to CSV
            file_path = os.path.join(dir_path, f"{ticker}.csv")
            #process df
            if drop_last:
                list_axis = ['Dividends','Stock Splits']
                data.drop(list_axis, axis=1, inplace=True)
            if drop_NA:
                data.dropna(inplace=True)

            data.to_csv(file_path)
            print(f"Data for {ticker} saved successfully.")
            # rt_string = f"Data for {ticker} saved successfully."

        else:
            print(f"No data found for {ticker}.")
            rt_string = f"No data found for {ticker}."

    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        rt_string = f"Error downloading data for {ticker}: {e}"
    # Sleep to respect rate limits
    sleep(sleep_time)
    return rt_string

def csv_symbol(csv_path: str, skip_first: bool = True, symbol_index: int = 0):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        if skip_first:
            header = next(reader, None)  # Skip the header row

        stock_list = []
        for row in reader:
            stock_list.append(row[symbol_index])  # Adjust index if necessary

    return stock_list