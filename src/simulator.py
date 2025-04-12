import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, data_dict, ret_window=1440, pca_window=1440, ou_window=1440, k=7, s_entry=1.25, s_exit=0.5):
        self.data_dict = data_dict
        self.ret_window = ret_window
        self.pca_window = pca_window
        self.ou_window = ou_window
        self.k = k
        self.s_entry = s_entry
        self.s_exit = s_exit

        self.asset_names = list(data_dict.keys())
        self.N = len(self.asset_names)

        price_dfs = []
        for name in self.asset_names:
            df = data_dict[name].copy()
            df = df[['midprice']]
            df.rename(columns={'midprice': name}, inplace=True)
            price_dfs.append(df)
        self.price_df = pd.concat(price_dfs, axis=1).sort_index()

        self.log_return_df = np.log(self.price_df).diff().dropna()
        self.times = self.log_return_df.index

    @staticmethod
    def compute_rolling_standardized_returns(df_returns, window):
        rolling_mean = df_returns.rolling(window=window).mean()
        rolling_std = df_returns.rolling(window=window).std()
        st_returns = (df_returns - rolling_mean) / rolling_std
        return st_returns.dropna()

    @staticmethod
    def compute_Q_matrix(Y, window, t, k):
        Y_window = Y[:, t - window:t]
        M = Y_window.shape[1]
        volatilities = np.std(Y_window, axis=1, ddof=1)
        P = (1 / (M - 1)) * (Y_window @ Y_window.T)
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        idx = np.argsort(eigenvalues)[::-1]
        top_k_eigenvectors = eigenvectors[:, idx[:k]]
        Q_matrix = top_k_eigenvectors / volatilities[:, np.newaxis]
        return Q_matrix

    @staticmethod
    def fit_ou_parameters(X):
        X_t = X[:-1]
        X_next = X[1:]
        A = np.vstack([X_t, np.ones_like(X_t)]).T
        b, a = np.linalg.lstsq(A, X_next, rcond=None)[0]
        residuals = X_next - (a + b * X_t)
        sigma_eps = np.std(residuals)
        kappa = -np.log(b) if 0 < b < 1 else 1e-6
        m = a / (1 - b) if abs(1 - b) > 1e-6 else 0.0
        sigma = sigma_eps * np.sqrt(2 * kappa / (1 - b ** 2)) if kappa > 0 else 1e-6
        sigma_eq = sigma / np.sqrt(2 * kappa) if kappa > 0 else 1e-6
        return kappa, m, sigma, sigma_eq

    def run_simulation(self):
        st_return_df = self.compute_rolling_standardized_returns(self.log_return_df, self.ret_window)
        Y = st_return_df.T.values
        M = Y.shape[1]
        start_t = max(self.ret_window, self.pca_window, self.ou_window)

        equity = 100000
        self.equity_curve = []
        self.pnl_series = []

        positions = np.zeros(self.N)
        cumulative_X = np.zeros((self.N, M))

        for t in range(start_t, M - 1):
            Q_t = self.compute_Q_matrix(Y, self.pca_window, t, self.k)
            F_t = Q_t.T @ Y[:, t]

            Y_window = Y[:, t - self.pca_window:t]
            F_window = np.array([Q_t.T @ Y[:, t - self.pca_window + i] for i in range(self.pca_window)]).T

            FF_inv = np.linalg.inv(F_window @ F_window.T + np.eye(self.k) * 1e-6)
            beta_matrix = (Y_window @ F_window.T) @ FF_inv

            fitted_return = np.sum(beta_matrix * F_t[np.newaxis, :], axis=1)
            actual_return = Y[:, t + 1]
            residual = actual_return - fitted_return
            cumulative_X[:, t + 1] = cumulative_X[:, t] + residual

            s_score = np.zeros(self.N)
            for i in range(self.N):
                if t >= self.ou_window:
                    X_series = cumulative_X[i, t - self.ou_window + 1:t + 1]
                    kappa, m, sigma, sigma_eq = self.fit_ou_parameters(X_series)
                    s_score[i] = (cumulative_X[i, t + 1] - m) / sigma_eq

            new_positions = np.zeros(self.N)
            for i in range(self.N):
                if positions[i] == 0:
                    if s_score[i] < -self.s_entry:
                        new_positions[i] = 1
                    elif s_score[i] > self.s_entry:
                        new_positions[i] = -1
                elif positions[i] == 1 and s_score[i] > -self.s_exit:
                    new_positions[i] = 0
                elif positions[i] == -1 and s_score[i] < self.s_exit:
                    new_positions[i] = 0
                else:
                    new_positions[i] = positions[i]
            positions = new_positions

            pnl = np.sum(positions * actual_return)
            equity += pnl
            self.pnl_series.append(pnl)
            self.equity_curve.append(equity)

        self.pnl_series = np.array(self.pnl_series)
        self.equity_curve = np.array(self.equity_curve)

    def plot_equity_curve(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.equity_curve, label='Equity Curve')
        plt.xlabel("Time steps")
        plt.ylabel("Equity")
        plt.title("Cumulative Equity Curve")
        plt.grid(True)
        plt.legend()
        plt.show()
