import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data(ticker): 
    base = f"../data/{ticker}"  # relative path
    try:
        files = [f for f in os.listdir(base) if f.endswith('.csv')]
        dfs = [] 
        for file in files: 
            file_path = os.path.join(base, file)
            df = pd.read_csv(file_path)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            dfs.append(df)
        
        if dfs: 
            return pd.concat(dfs).sort_values('open_time').reset_index(drop=True)
        else:
            print("Empty dataframe, load was not successful")
            return pd.DataFrame()
    except Exception as e:
        print(f"Data was not loaded successfully: {e}")


def transform_data(df):
    df['mid'] = (df['open'] + df['close']) / 2
    return df
