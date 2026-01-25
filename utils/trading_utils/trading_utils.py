import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_random_clusters(TICKERS, n_clusters=3, seed=42):
    df_clusters = pd.DataFrame(columns=["asset", "cluster"])
    df_clusters["asset"] = TICKERS
    np.random.seed(seed)
    df_clusters["cluster"] = np.random.randint(1, n_clusters+1, size=len(TICKERS))
    return df_clusters

class Pair:
    def __init__(self, stock_A, stock_B, index_A, index_B, returns_A, returns_B, 
                 prices_A, prices_B, spreads_A, spreads_B):
        self.stock_A = stock_A 
        self.index_A = index_A
        self.returns_A = returns_A
        self.stock_B = stock_B
        self.index_B = index_B
        self.returns_B = returns_B
        self.prices_A = prices_A
        self.prices_B = prices_B
        self.spreads_A = spreads_A
        self.spreads_B = spreads_B
        self.entered = False # True only when pair is entered
        self.invert = None # baseline: longA shortB, invert: shortA longB
        self.current_t_in = None
        self.plot_windows = []
        self.R_in = None
        self.cumulR_in = None
        self.patience = 0

    def enter(self, invert, t_in, R_in, cumulR_in):
        self.entered = True
        self.invert = invert
        self.current_t_in = t_in
        self.R_in = R_in
        self.cumulR_in = cumulR_in

    def exit(self, t_out, emergency):
        self.entered = False
        self.plot_windows.append([self.current_t_in, t_out, self.invert, emergency])
        self.invert = None
        self.current_t_in = None
        self.R_in = None
        self.patience = 0

    def evaluate(self, t,
                 window_lookback, lambda_in, lambda_out, lambda_emergency, patience_max,
                 np_positions):
        
        returns_diff_window = self.returns_A.iloc[t - window_lookback: t+1] - self.returns_B.iloc[t - window_lookback: t+1]
        rolling_sigma = returns_diff_window.std()
        latest_return_diff = self.returns_A.iloc[t] - self.returns_B.iloc[t]

        cumul_returns_diff = (1 + returns_diff_window).cumprod() - 1

        if(not self.entered):
            
            # Calculate total spread cost for entering the pair (one-way cost)
            spread_cost_A = (self.spreads_A.iloc[t] / (2 * self.prices_A.iloc[t]))
            spread_cost_B = (self.spreads_B.iloc[t] / (2 * self.prices_B.iloc[t]))
            total_spread_cost = spread_cost_A + spread_cost_B
            
            # Adjust entry threshold to account for spread
            effective_lambda_in = lambda_in * rolling_sigma + total_spread_cost
            
            # Check for SIGNAL D'ENTREE (with spread adjustment)
            if latest_return_diff > effective_lambda_in:  # Enter pair, short A long B
                self.enter(invert = True,
                           t_in = t,
                           R_in = latest_return_diff,
                           cumulR_in = cumul_returns_diff.iloc[-1])

            elif latest_return_diff < -effective_lambda_in:  # Enter pair, long A short B
                self.enter(invert = False,
                           t_in = t,
                           R_in = latest_return_diff, 
                           cumulR_in = cumul_returns_diff.iloc[-1])

        else:
            # Check for SIGNAL DE SORTIE
        
            if self.invert == True:
                # Sortie bénéfique
                if cumul_returns_diff.iloc[-1] < lambda_out * self.cumulR_in:
                    self.exit(t_out = t, emergency = 'No')
                    return
                # Sortie d'urgence
                elif cumul_returns_diff.iloc[-1] > lambda_emergency * self.cumulR_in:
                    self.exit(t_out = t, emergency = 'onDrift')
                    return


            else:
                # Sortie bénéfique
                if cumul_returns_diff.iloc[-1] > lambda_out * self.cumulR_in:
                    self.exit(t_out = t, emergency = 'No')
                    return
                # Sortie d'urgence
                elif cumul_returns_diff.iloc[-1] < lambda_emergency * self.cumulR_in:
                    self.exit(t_out = t, emergency = 'onDrift')
                    return
            
            # Sortie sur patience
            if self.invert == True:
                # If divergence is getting worse or staying flat, increment patience
                if cumul_returns_diff.iloc[-1] >= self.cumulR_in:
                    self.patience += 1
                else:
                    # If improving (converging), reset patience
                    self.patience = 0
            else:
                # If divergence is getting worse or staying flat, increment patience
                if cumul_returns_diff.iloc[-1] <= self.cumulR_in:
                    self.patience += 1
                else:
                    # If improving (converging), reset patience
                    self.patience = 0
            
            # Exit if patience exceeded
            if self.patience >= patience_max:
                self.exit(t_out = t, emergency = 'onPatience')

        # update positions dataframe
        if(self.entered):
            np_positions[t, self.index_A] += (1 if not self.invert else -1)
            np_positions[t, self.index_B] += (-1 if not self.invert else 1)
            

    def info(self):
        print("PAIR==================")
        print("Stocks: " + self.stock_A + ", " + self.stock_B)
    

    def plot_lifetime_last(self, df_returns, df_cum_returns):
        fig, ax = plt.subplots(figsize = (22, 4))
        ax.set_title("Lifetime of pair (" + self.stock_A + ", "  + self.stock_B + ")", fontsize = 14) 
        ax.plot(df_cum_returns[self.stock_A], color = "purple", label = "(A) " + self.stock_A)
        ax.plot(df_cum_returns[self.stock_B], color = "darkgreen", label = "(B) " + self.stock_B)
        ax.set_xticks([])
        #plt.plot(df_returns[self.stock_A] - df_returns[self.stock_B], label = "A - B")
        ax.grid(alpha = 0.5)
        
        
        if len(self.plot_windows) > 0:
            for i in range(len(self.plot_windows)):
        
                t_in, t_out, invert, emergency = self.plot_windows[i]
        
                date_in = df_returns.iloc[t_in].name
                date_out = df_returns.iloc[t_out].name
                ax.axvline(date_out,                 color = 'black' if emergency == 'No' else ('magenta' if emergency == 'onDrift' else 'red'), linewidth=0.3)
                ax.axvspan(date_in, date_out,        color = 'blue' if not invert else 'orange', alpha = 0.2)
            if(self.entered):
                ax.axvline(date_in, linestyle = ':', color = 'gray')
        
        ax.legend()
        plt.show()


def compute_strategy_pnl(returns_df, positions_df, prices_df, spreads_df):
    
	# Position at previous tick (shift forward by 1)
	# We hold the position from t-1 during period t, earning returns[t]
	lagged_positions = positions_df.shift(1).fillna(0)
	
	# Trading PnL: position held * returns earned during that period
	# Each asset contributes: position_held * return_of_asset
	trading_pnl_per_asset = lagged_positions * returns_df
	trading_pnl = trading_pnl_per_asset.sum(axis=1)
	
	# Calculate position changes (where transactions occur)
	position_changes = positions_df.diff().fillna(positions_df.iloc[0])
	
	# Transaction cost per trade = (spread / 2) / price
	transaction_cost_pct = (spreads_df / (2 * prices_df))
	
	transaction_costs_per_tick = (position_changes.abs() * transaction_cost_pct).sum(axis=1)
	
	net_pnl_per_tick = trading_pnl - transaction_costs_per_tick
	
	cumulative_pnl = net_pnl_per_tick.cumsum()
	        
	return cumulative_pnl