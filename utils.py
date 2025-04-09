import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from time import sleep
import csv

def download_and_save_data(ticker, interval, start_date, end_date, save_dir:str, sleep_time:int = 2):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if not data.empty:
            # Create directory for the index and interval if it doesn't exist
            dir_path = os.path.join(save_dir, interval)
            os.makedirs(dir_path, exist_ok=True)
            # Save to CSV
            file_path = os.path.join(dir_path, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"Data for {ticker} saved successfully.")
        else:
            print(f"No data found for {ticker}.")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
    # Sleep to respect rate limits
    sleep(sleep_time)


def csv_symbol(csv_path: str, skip_first: bool = True, symbol_index: int = 0):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        if skip_first:
            header = next(reader, None)  # Skip the header row

        stock_list = []
        for row in reader:
            stock_list.append(row[symbol_index])  # Adjust index if necessary

    return stock_list