import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go

def plot_candlestick(df, title="1-min Candles", pos_color='green', neg_color='red', start_idx=None, end_idx=None):
    # Handle slicing
    if start_idx is not None or end_idx is not None:
        df = df.iloc[start_idx:end_idx]

    fig = go.Figure(data=[go.Candlestick(
        x=df['open_time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color=pos_color,
        decreasing_line_color=neg_color
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="1 min candles",
        xaxis_rangeslider_visible=False,
        width=1200, 
        height=600

    )

    fig.show()


def plot_time_series(df, ticker="", color='green', start_idx=None, end_idx=None):
    if start_idx is not None or end_idx is not None:
        df = df.iloc[start_idx:end_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(df['open_time'], df['mid'], color=color, linewidth=1)

    plt.title(f"1 minute time-series {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_normalized_pair(df1, df2, title="Normalized Price Comparison", 
                         label1="Series 1", label2="Series 2", 
                         color1='blue', color2='green', 
                         start_idx=None, end_idx=None):
    
    if start_idx is not None or end_idx is not None:
        df1 = df1.iloc[start_idx:end_idx]
        df2 = df2.iloc[start_idx:end_idx]

    # Normalize close prices to start at 1
    norm1 = df1['mid'] / df1['mid'].iloc[0]
    norm2 = df2['mid'] / df2['mid'].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(df1['open_time'], norm1, label=label1, color=color1, linewidth=1.5)
    plt.plot(df2['open_time'], norm2, label=label2, color=color2, linewidth=1.5)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_series(S, color='blue', title=None):
    plt.figure(figsize=(12, 6))
    plt.plot(S.index, S.values, color=color, linewidth=1.5)

    if title:
        plt.title(title)
    else:
        plt.title("Series v. Time")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()