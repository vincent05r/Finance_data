import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime
from utils import *


# Configuration
interval = ['1m']
period_interval = ['8d']
sleep_time = 2  # seconds

main_save_dir_name = "FTS_data_snapshots"

# Create timestamp-based save directory using system time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(main_save_dir_name, f"snapshot_{timestamp}")
os.makedirs(save_dir, exist_ok=True)


symbol_path_pool = csv_symbol("all_symbol_c.csv")

summary_log = []


# Run download
for index_l in range(len(period_interval)): 
    for stock_symbol in symbol_path_pool:
        error_msg = download_and_save_data_period(
            ticker=stock_symbol,
            interval=interval[index_l],
            period=period_interval[index_l],
            save_dir=save_dir,
            sleep_time=sleep_time
        )
        if error_msg:
            summary_log.append(error_msg)


if len(summary_log) != 0:
    log_path = os.path.join(save_dir, "error_log.txt")
    with open(log_path,"w") as f:
        for line in summary_log:
            f.write(line + "\n" )
            
    print(f"\n⚠️ Summary log saved to: {log_path}")
else:
    print("\n✅ All tickers processed successfully. No errors.")