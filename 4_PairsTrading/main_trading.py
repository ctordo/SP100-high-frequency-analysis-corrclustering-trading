import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import importlib
import trading_utils, trading_visuals
importlib.reload(trading_utils)
importlib.reload(trading_visuals)
from trading_utils import generate_random_clusters, Pair, construct_positions, compute_strategy_pnl
from trading_visuals import barplot_distrib_clusters, draw_positions_table, plot_cumulative_pnl

data_local = "../../FBD_local_data/"
data_repo = "../Data/"

def main_4(window, lambda_in, lambda_out, lambda_emergency, patience_max,
           use_SUBDATA:bool = True,
           nb_pairs_cap:int=-1,
           display_figures:bool = False):

    # Initial message
    print("="*70)
    print("ENTERING PHASE 4: TRADING STRATEGY IMPELMENTATION")
    print("="*70 + "\n")

    # ============ LOAD PHASE INPUTS ============ 
    print("4.1 Loading phase inputs...")
    
    # Stock prices
    df_prices = pd.read_csv(data_local + "stock_prices.csv")
    print("  'stock_prices.csv' loaded as 'df_prices'")
    # Tickers
    TICKERS = list(df_prices.columns[1:])
    tickers_index_map = {v: i for i, v in enumerate(TICKERS)}
    
    # Clusters
    # df_clusters = pd.read_csv...
    df_clusters = generate_random_clusters(TICKERS, n_clusters = 10)
    n_clusters = len(df_clusters["cluster"].unique())
    print(f"  Random clusters generated. Assets are partitioned in {n_clusters} different clusters.")
    if(display_figures):
        barplot_distrib_clusters(df_clusters, n_clusters)
    
    # ============  BUILD MASTER "DATA" DICT ============ 
    print("4.2 Building DATA master dictionary...")
    
    df_prices.set_index("timestamp", inplace=True, drop=True)
    df_prices = df_prices.interpolate(method="linear", limit_direction="both")
    df_returns = df_prices.pct_change()
    df_cum_returns = (1 + df_returns).cumprod() - 1
    
    DATA = {}
    if(not use_SUBDATA):
        print("  use_SUBDATA = False -> Using full data range")
        DATA["PRICES"] = df_prices
        DATA["RETURNS"] = df_returns
        DATA["CUMRETURNS"] = df_cum_returns
    
    else:
        print("  use_SUBDATA = True -> Using restricted range (approx. 3 weeks) only")
        start_date = "2008-09-02"
        end_date = "2008-09-25"
        df_prices_window = df_prices[(df_prices.index >= start_date) & (df_prices.index <= end_date)]
        df_returns_window = df_returns[(df_returns.index >= start_date) & (df_returns.index <= end_date)]
        df_cum_returns_window = (1 + df_returns_window).cumprod() - 1
        DATA["PRICES"] = df_prices_window
        DATA["RETURNS"] = df_returns_window
        DATA["CUMRETURNS"] = df_cum_returns_window
    
    print("  Succesfully built DATA dictionary.")
    
    # SUBDATA Building - Sanity check 
    if(display_figures):
        print("  (Sanity check) Plotting cumulative returns of 5 random assets:")
        fig, ax = plt.subplots(figsize = (22, 4))
        ax.plot(DATA["CUMRETURNS"].sample(5, axis=1), label = DATA["CUMRETURNS"].sample(5, axis=1).columns)
        ax.set_title("5 random samples of DATA[CUMRETURNS]")
        ax.set_xticks([])
        ax.legend()
        ax.axhline(0, color='black')
        ax.grid(axis = 'y', alpha =0.5)
        plt.show()

    # ============ CONSTRUCT INTRA-CLUSTERS PAIRS ============ 
    print("4.3 Constructing intra-clusters pairs...")
    pairs = []
    for cluster in range(1, n_clusters+1):
        stocks_in_cluster = list(df_clusters[df_clusters["cluster"] == cluster]["asset"].values)
        for i, ticker1 in enumerate(stocks_in_cluster):
            for ticker2 in stocks_in_cluster[i+1:]:
                pairs.append(Pair(ticker1, ticker2,
                                  tickers_index_map[ticker1], tickers_index_map[ticker2],
                                  df_returns = DATA["RETURNS"]))

    nb_pairs = len(pairs)
    print(f"  Pairs successfully constructed. Number of pairs: {nb_pairs}")

    # NUMBER OF PAIRS TO EVALUATE 
    nb_eval = min(nb_pairs, nb_pairs_cap) if nb_pairs_cap > 0 else nb_pairs
    print(f"4.4 Evaluation of {nb_eval} pairs...")

    np_positions = construct_positions(pairs, nb_eval, nb_pairs, TICKERS,
                                       window, lambda_in, lambda_out, lambda_emergency, patience_max,
                                       DATA["RETURNS"], DATA["CUMRETURNS"],
                                       display_figures)

    # Do a column for sum of positions to check = 0
    np_positions[:, -1] = np_positions.sum(axis=1)
    
    print(f"  Positions array filled, shape: {np_positions.shape} (columns allocated for {len(TICKERS)} assets, +1 for row-wise sum).")

    if(display_figures):
        draw_positions_table(np_positions, TICKERS)

    # CONVERT POSITIONS TO DATAFRAME
    df_positions = pd.DataFrame(data=np_positions[:,:-1],
                            columns=TICKERS,
                            index=DATA["RETURNS"].index)
    
    # ============ COMPUTE STRATEGY PNL ============ 
    print("4.5 Computing PNL series...")
    pnl_curve = compute_strategy_pnl(df_returns_window, df_positions, transaction_cost=0.0001)
    if(display_figures):
        plot_cumulative_pnl(pnl_curve)

    # Final message
    print("\n" + "="*70)
    print("PHASE 4 CORRECTLY TERMINATED")
    print("="*70 + "\n")
