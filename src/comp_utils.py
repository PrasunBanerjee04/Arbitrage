import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations

def compute_spread(df1, df2, start=None, end=None):
    delta_S = None
    if (start and end):
        delta_S = df1.loc[start:end, 'mid'] - df2.loc[start:end, 'mid']
    else: 
        delta_S = df1['mid'] - df2['mid']
    
    return delta_S

def normalized_spread_sum(df1, df2):
    spread = compute_spread(df1, df2)
    z_spread = (spread - spread.mean()) / spread.std()
    return z_spread.abs().sum()

def find_best_pair_for_ticker(pairs_dict, target_key):
    if target_key not in pairs_dict:
        raise ValueError(f"{target_key} not found in pairs_dict.")

    target_df = pairs_dict[target_key]
    min_score = float('inf')
    best_pair = None

    for other_key, other_df in pairs_dict.items():
        if other_key == target_key:
            continue

        try:
            score = normalized_spread_sum(target_df, other_df)

            if score < min_score:
                min_score = score
                best_pair = other_key
        except Exception as e:
            print(f"Skipping pair {target_key}-{other_key}: {e}")

    return best_pair, min_score

def get_pairs(pairs_dict): 
    pairs = {}
    for security_i in pairs_dict.keys():
        best_j, score = find_best_pair_for_ticker(pairs_dict, security_i)
        pairs[(security_i, best_j)] = score
    return pairs

def get_residuals(df1, df2):
    y = df1['mid']
    x = df2['mid']

    y, x = y.align(x, join='inner')
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit() 
    residuals = model.resid
    alpha, beta = model.params 
    return residuals, alpha, beta 

def get_pair_stationary(ticker_dict):
    stationary_pairs = {}

    tickers = list(ticker_dict.keys())

    for ticker1, ticker2 in combinations(tickers, 2):
        df1 = ticker_dict[ticker1]
        df2 = ticker_dict[ticker2]

        try:
            residuals, _, _ = get_residuals(df1, df2)
            cadf = adfuller(residuals)

            score = cadf[0]
            cutoffs = cadf[4]

            if score < cutoffs['1%']:
                stationary_pairs[(ticker1, ticker2)] = f'1%, score: {score}'
            elif score < cutoffs['5%']:
                stationary_pairs[(ticker1, ticker2)] = '5%, score: {score}'
            elif score < cutoffs['10%']:
                stationary_pairs[(ticker1, ticker2)] = '10%, score: {score}'

        except Exception as e:
            print(f"Skipping pair {ticker1}-{ticker2}: {e}")

    return stationary_pairs

def test_adf_on_best_pairs(best_pairs, ticker_dict):
    adf_results = {}

    for pair in best_pairs.keys():
        t1, t2 = pair
        df1 = ticker_dict[t1]
        df2 = ticker_dict[t2]

        try:
            residuals, _, _ = get_residuals(df1, df2)
            cadf = adfuller(residuals)

            score = cadf[0]
            cutoffs = cadf[4]  # {'1%': ..., '5%': ..., '10%': ...}

            if score < cutoffs['1%']:
                adf_results[pair] = '1%'
            elif score < cutoffs['5%']:
                adf_results[pair] = '5%'
            elif score < cutoffs['10%']:
                adf_results[pair] = '10%'
            # else: don't include non-stationary pairs

        except Exception as e:
            print(f"ADF test failed for pair {t1}-{t2}: {e}")

    return adf_results

def get_Y(ticker_map):
    standardized_returns = {}
    window = 60

    for symbol, df in ticker_map.items():
        df = df.copy()
        df['log_return'] = np.log(df['mid']).diff()
        df['st_return'] = (
            df['log_return'] - df['log_return'].rolling(window=window).mean()
        ) / df['log_return'].rolling(window=window).std()
        
        standardized_returns[symbol] = df['st_return']

    Y_df = pd.concat(standardized_returns, axis=1)
    Y_df = Y_df.dropna()
    Y_matrix = Y_df.T.values

    return Y_matrix


    