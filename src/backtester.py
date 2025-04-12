import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from comp_utils import get_residuals

def generate_signals(s_scores, s_entry=1.25, s_exit=0.5):
    N, M = s_scores.shape
    positions = np.zeros_like(s_scores)

    for i in range(N):
        pos = 0
        for t in range(M):
            s = s_scores[i, t]

            if pos == 0:
                if s < -s_entry:
                    pos = 1  # enter long
                elif s > s_entry:
                    pos = -1  # enter short
            elif pos == 1 and s > -s_exit:
                pos = 0  # exit long
            elif pos == -1 and s < s_exit:
                pos = 0  # exit short

            positions[i, t] = pos

    return positions


class Backtester: 

    def __init__(self, df1, df2, threshold=1.0, capital=100000, verbose=True, window=60, min_hold=5):
        self.df1 = df1
        self.df2 = df2
        self.threshold = threshold
        self.capital = capital
        self.verbose = verbose
        self.window = window
        self.min_hold = min_hold

        self.results = None
        self._prepare_data()

    def _prepare_data(self):
        # Run regression on mid prices
        residuals, alpha, beta = get_residuals(self.df1, self.df2)

        self.beta = beta
        self.spread = residuals
        self.zscore = (residuals - residuals.rolling(self.window).mean()) / residuals.rolling(self.window).std()

        if self.verbose:
            print(f"Estimated beta: {self.beta:.4f}")

    def run(self):
        position = 0  # 0 = flat, 1 = long spread, -1 = short spread
        entry_price_1 = entry_price_2 = None
        entry_index = None
        pnl = []
        trade_log = []

        for i in range(self.window, len(self.zscore)):
            z = self.zscore.iloc[i]
            t = self.zscore.index[i]
            price1 = self.df1['mid'].iloc[i]
            price2 = self.df2['mid'].iloc[i]

            # Entry logic
            if position == 0:
                if z > self.threshold:
                    position = -1  # short spread
                    entry_price_1 = price1
                    entry_price_2 = price2
                    entry_index = i
                    trade_log.append((t, 'Enter SHORT spread'))

                elif z < -self.threshold:
                    position = 1  # long spread
                    entry_price_1 = price1
                    entry_price_2 = price2
                    entry_index = i
                    trade_log.append((t, 'Enter LONG spread'))

            # Exit logic with min_hold restriction
            elif position != 0:
                if abs(z) < 0.1 and (i - entry_index) >= self.min_hold:
                    if position == 1:
                        pnl_val = (price1 - entry_price_1) - self.beta * (price2 - entry_price_2)
                    else:
                        pnl_val = (entry_price_1 - price1) - self.beta * (entry_price_2 - price2)

                    pnl.append(pnl_val)
                    trade_log.append((t, f'Exit: PnL = {pnl_val:.2f}'))
                    position = 0
                    entry_index = None

        self.results = {
            'total_pnl': np.sum(pnl),
            'n_trades': len(pnl),
            'avg_pnl': np.mean(pnl) if pnl else 0,
            'pnl_series': pnl,
            'log': trade_log
        }

        if self.verbose:
            print(f"\nTotal PnL: {self.results['total_pnl']:.2f}")
            print(f"Trades executed: {self.results['n_trades']}")
            print(f"Average PnL per trade: {self.results['avg_pnl']:.2f}")

        return self.results

    def summary(self):
        return self.results

    def plot_cumulative_pnl(self):
        if self.results is None or not self.results['pnl_series']:
            print("No results to plot. Run the backtester first.")
            return

        pnl_series = np.cumsum(self.results['pnl_series'])

        plt.figure(figsize=(12, 6))
        plt.plot(pnl_series, label='Cumulative PnL', linewidth=2)
        plt.axhline(0, linestyle='--', color='gray')
        plt.title("Cumulative PnL Over Time")
        plt.xlabel("Trades")
        plt.ylabel("Cumulative Profit")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_trade_signals_on_spread(self):
        if self.results is None or not self.results['log']:
            print("No trades to visualize. Run the backtester first.")
            return

        plt.figure(figsize=(14, 6))
        plt.plot(self.spread, label='Spread (Residual)', color='blue', alpha=0.7)

        # Prevent duplicate legend labels
        has_entry, has_exit = False, False

        for timestamp, note in self.results['log']:
            if "Enter" in note:
                plt.axvline(timestamp, color='green', linestyle='--', alpha=0.5,
                            label='Entry' if not has_entry else "")
                has_entry = True
            elif "Exit" in note:
                plt.axvline(timestamp, color='red', linestyle='--', alpha=0.5,
                            label='Exit' if not has_exit else "")
                has_exit = True

        plt.title("Spread with Entry/Exit Points")
        plt.xlabel("Time")
        plt.ylabel("Residual Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
