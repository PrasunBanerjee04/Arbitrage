import pandas as pd
import numpy as np

def set_up(series, window=1440, threshold=1.5):
    df = pd.DataFrame({'spread': series})
    df['mean'] = df['spread'].rolling(window=window).mean()
    df['std'] = df['spread'].rolling(window=window).std()
    df['z'] = (df['spread'] - df['mean']) / df['std']

    df['position'] = 0
    df.loc[df['z'] < -1*threshold, 'position'] = 1
    df.loc[df['z'] > threshold, 'position'] = -1
    df = df.dropna()
    return df

def calculate_pnl(df, transaction_cost=0.0005):
    df = df.copy()
    df['spread_return'] = df['spread'].diff()
    df['shifted_position'] = df['position'].shift(1).fillna(0)
    df['pnl'] = df['shifted_position'] * df['spread_return']

    df['position_change'] = df['position'].diff().abs()
    df['transaction_cost'] = df['position_change'] * df['spread'].abs() * transaction_cost
    df['pnl'] -= df['transaction_cost']
    
    df['cum_pnl'] = df['pnl'].cumsum()
    return df

def simulate_statarb(spread, window=1440, threshold=1.5, transaction_cost=0.0005):
    df = set_up(spread, window=window, threshold=threshold)
    df = calculate_pnl(df, transaction_cost=transaction_cost)
    return df