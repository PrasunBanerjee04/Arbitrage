import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, 
                 data_dict,
                 ret_window=1440,
                 pca_window=1440,
                 ou_window=1440,
                 k=7,
                 s_entry=1.25,
                 s_exit=0.5,
                 pca_update_interval=10,
                 start_index=None,
                 end_index=None):
        """
        A PCA-based stat-arb simulator with OU modeling for residuals, factor hedging, 
        and various performance optimizations.

        Parameters:
            data_dict (dict): {symbol: DataFrame with 'mid' prices, datetime index}
            ret_window (int): rolling window for standardized returns
            pca_window (int): lookback window for PCA and regression
            ou_window (int): lookback for OU process fitting
            k (int): number of eigenportfolios (principal components) to keep
            s_entry (float): s-score entry threshold
            s_exit (float): s-score exit threshold
            pca_update_interval (int): how often (in timesteps) to recompute PCA
            start_index (int): starting index of simulation window
            end_index (int): ending index of simulation window
        """
        self.data_dict = data_dict
        self.ret_window = ret_window
        self.pca_window = pca_window
        self.ou_window = ou_window
        self.k = k
        self.s_entry = s_entry
        self.s_exit = s_exit
        self.pca_update_interval = pca_update_interval
        self.start_index = start_index
        self.end_index = end_index

        self.asset_names = list(data_dict.keys())
        self.N = len(self.asset_names)

        # Build a combined price DataFrame
        price_dfs = []
        for name in self.asset_names:
            df = data_dict[name].copy()
            df = df[['mid']].rename(columns={'mid': name})
            price_dfs.append(df)
        self.price_df = pd.concat(price_dfs, axis=1).sort_index()

        # Compute log returns
        self.log_return_df = np.log(self.price_df).diff().dropna()
        self.times = self.log_return_df.index

    @staticmethod
    def compute_rolling_standardized_returns(df_returns, window):
        rolling_mean = df_returns.rolling(window=window).mean()
        rolling_std = df_returns.rolling(window=window).std()
        st_returns = (df_returns - rolling_mean) / rolling_std
        return st_returns.dropna()

    @staticmethod
    def compute_Q_matrix(Y_window, k):
        volatilities = np.std(Y_window, axis=1, ddof=1)
        M = Y_window.shape[1]
        P = (1 / (M - 1)) * (Y_window @ Y_window.T)
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        idx = np.argsort(eigenvalues)[::-1]
        top_k_eigenvectors = eigenvectors[:, idx[:k]]
        Q_matrix = top_k_eigenvectors / volatilities[:, np.newaxis]
        return Q_matrix

    @staticmethod
    def fit_ou_parameters(X):
        if len(X) < 2:
            return 1e-6, 0.0, 1e-6, 1e-6
        X_t = X[:-1]
        X_next = X[1:]
        A = np.vstack([X_t, np.ones_like(X_t)]).T
        b, a = np.linalg.lstsq(A, X_next, rcond=None)[0]
        residuals = X_next - (a + b * X_t)
        sigma_eps = np.std(residuals)
        if b <= 0 or b >= 1:
            return 1e-6, 0.0, 1e-6, 1e-6
        kappa = -np.log(b)
        m = a / (1 - b) if abs(1 - b) >= 1e-6 else 0.0
        denom = 1 - b**2
        sigma = sigma_eps * np.sqrt(2 * kappa / denom) if denom > 0 and kappa > 0 else 1e-6
        sigma_eq = sigma / np.sqrt(2 * kappa) if kappa > 0 and sigma > 0 else 1e-6
        return kappa, m, sigma, sigma_eq

    def run_simulation(self):
        st_return_df = self.compute_rolling_standardized_returns(self.log_return_df, self.ret_window)
        Y = st_return_df.T.values
        M = Y.shape[1]

        default_start = max(self.ret_window, self.pca_window, self.ou_window)
        start_t = self.start_index if self.start_index is not None else default_start
        end_t = self.end_index if self.end_index is not None else M - 1

        equity = 100_000.0
        self.equity_curve = []
        self.pnl_series = []
        positions = np.zeros(self.N)
        cumulative_X = np.zeros((self.N, M))
        cached_Q = None

        for t in range(start_t, end_t):
            if (t % self.pca_update_interval == 0) or (cached_Q is None):
                Y_window = Y[:, t - self.pca_window : t]
                cached_Q = self.compute_Q_matrix(Y_window, self.k)

            Q_t = cached_Q
            F_t = Q_t.T @ Y[:, t]
            Y_window = Y[:, t - self.pca_window : t]
            F_window = Q_t.T @ Y_window
            FF = F_window @ F_window.T + np.eye(self.k) * 1e-9
            FF_inv = np.linalg.inv(FF)
            beta_matrix = (Y_window @ F_window.T) @ FF_inv
            fitted_return = beta_matrix @ F_t
            actual_return = Y[:, t+1]
            residual = actual_return - fitted_return
            cumulative_X[:, t+1] = cumulative_X[:, t] + residual

            s_score = np.zeros(self.N)
            if t >= self.ou_window:
                for i in range(self.N):
                    X_series = cumulative_X[i, t - self.ou_window + 1 : t + 1]
                    kappa, m, sigma, sigma_eq = self.fit_ou_parameters(X_series)
                    s_score[i] = (cumulative_X[i, t+1] - m) / sigma_eq if sigma_eq > 1e-9 else 0.0

            new_positions = np.zeros(self.N)
            for i in range(self.N):
                old_pos = positions[i]
                if old_pos == 0:
                    if s_score[i] < -self.s_entry:
                        new_positions[i] = 1.0
                    elif s_score[i] > self.s_entry:
                        new_positions[i] = -1.0
                elif old_pos == 1.0 and s_score[i] > -self.s_exit:
                    new_positions[i] = 0.0
                elif old_pos == -1.0 and s_score[i] < self.s_exit:
                    new_positions[i] = 0.0
                else:
                    new_positions[i] = old_pos
            positions = new_positions

            final_positions = np.zeros(self.N)
            for i in range(self.N):
                p_i = positions[i]
                if abs(p_i) > 1e-9:
                    factor_exposure_i = beta_matrix[i, :] * p_i
                    hedge_i = np.sum(-factor_exposure_i[j] * Q_t[:, j] for j in range(self.k))
                    final_positions[i] += p_i
                    final_positions += hedge_i

            pnl = np.sum(final_positions * actual_return)
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
