import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import importlib
import utils.trading_utils as trading_utils, utils.trading_visuals as trading_visuals
importlib.reload(trading_utils)
importlib.reload(trading_visuals)
from utils.trading_utils import generate_random_clusters, Pair, compute_strategy_pnl
from utils.trading_visuals import barplot_distrib_clusters, draw_positions_table, plot_cumulative_pnl

from main_clustering import run_louvain

data_local = "../FBD_local_data/"

def main_4(window_clustering, window_lookback, lambda_in, lambda_out, lambda_emergency, patience_max,
           use_SUBDATA:bool = True,
           nb_pairs_cap:int=-1,
           plot_pairs_lifetimes:bool = False,
           display_figures:bool = False):

    # Initial message
    print("="*70)
    print("ENTERING PHASE 4: TRADING STRATEGY IMPELMENTATION")
    print("="*70 + "\n")

    # ============ LOAD AND REFORMAT PHASE INPUTS ============ 
    print("4.1 Loading phase inputs...")
    
    # Stock prices
    print("  Loading + structuring 'stock_prices.csv'...")
    df_prices = pd.read_csv(data_local + "stock_prices.csv")
    df_prices.set_index("timestamp", inplace=True, drop=True)
    # Stock returns
    print("  Loading + structuring 'stock_returns.csv'...")
    df_returns = pd.read_csv(data_local + "stock_returns.csv")
    df_returns.set_index("timestamp", inplace=True, drop=True)
    # Stock spreads
    print("  Loading + structuring 'stock_spreads.csv'...")
    df_spreads = pd.read_csv(data_local + "stock_spreads.csv")
    df_spreads.set_index("timestamp", inplace=True, drop=True)

    # Compute cumulative returns
    df_cum_returns = (1 + df_returns).cumprod() - 1
    
    # Tickers
    TICKERS = list(df_prices.columns)
    tickers_index_map = {v: i for i, v in enumerate(TICKERS)}
    print(tickers_index_map)
    

    # ============  BUILD MASTER "DATA" DICT ============ 
    print("4.2 Building DATA master dictionary...")
        
    DATA = {}
    if(not use_SUBDATA):
        print("  use_SUBDATA = False -> Using full data range")
        DATA["PRICES"] = df_prices
        DATA["RETURNS"] = df_returns
        DATA["CUMRETURNS"] = df_cum_returns
        DATA["SPREADS"] = df_spreads
    
    else:
        print("  Argument 'use_SUBDATA' = True -> Using restricted range (approx. 3 weeks) only")
        start_date = "2008-09-02"
        end_date = "2008-09-25"
        df_prices_window = df_prices[(df_prices.index >= start_date) & (df_prices.index <= end_date)]
        df_returns_window = df_returns[(df_returns.index >= start_date) & (df_returns.index <= end_date)]
        df_cum_returns_window = (1 + df_returns_window).cumprod() - 1
        df_spreads_window = df_spreads[(df_spreads.index >= start_date) & (df_spreads.index <= end_date)]
        DATA["PRICES"] = df_prices_window
        DATA["RETURNS"] = df_returns_window
        DATA["CUMRETURNS"] = df_cum_returns_window
        DATA["SPREADS"] = df_spreads_window
        print(f"  --> gives {len(df_prices_window)} timestamps.")
        
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

    # ============ ROLLING CLUSTERS AND PAIRS CONSTRUCTION ============ 
    print("4.3 Implementing rolling clustering approach...")
    print(f"  Clustering will be recalculated every {window_clustering} timestamps")

    # Initialize positions array
    np_positions = np.zeros((len(DATA["RETURNS"].index), len(TICKERS)+1))
    print(f"  Successfully initialized empty positions table, shape:{np_positions.shape}")

    # Determine the number of rolling clustering windows
    total_timestamps = len(DATA["RETURNS"])
    num_windows = (total_timestamps - window_clustering) // window_clustering + 1
    print(f"  Total timestamps: {total_timestamps}, Number of rolling windows: {num_windows}")

    # Iterate through rolling windows
    for window_idx in range(num_windows):
        clustering_start = window_idx * window_clustering
        clustering_end = clustering_start + window_clustering
        trading_start = clustering_end
        trading_end = min(trading_start + window_clustering, total_timestamps)
        
        print(f"\n  === Rolling Window {window_idx + 1}/{num_windows} ===")
        print(f"  Clustering period: t={clustering_start} to t={clustering_end-1}")
        print(f"  Trading period: t={trading_start} to t={trading_end-1}")
        
        # Get clustering data (past window timestamps)
        df_returns_clustering = DATA["RETURNS"].iloc[clustering_start:clustering_end]
        
        # Generate clusters based on clustering period data
        ### df_clusters = generate_random_clusters(TICKERS, n_clusters=10, seed=42 + window_idx)
        df_clusters = run_louvain(df_returns_clustering)

        n_clusters = len(df_clusters["cluster"].unique())
        print(f"  Generated {n_clusters} clusters for this window")
        
        if(window_idx == 0 and display_figures):
            barplot_distrib_clusters(df_clusters, n_clusters)
        
        # Construct intra-cluster pairs for this window
        pairs = []
        for cluster in range(1, n_clusters+1):
            stocks_in_cluster = list(df_clusters[df_clusters["cluster"] == cluster]["asset"].values)
            for i, ticker1 in enumerate(stocks_in_cluster):
                for ticker2 in stocks_in_cluster[i+1:]:
                    pairs.append(Pair(ticker1, ticker2,
                                      tickers_index_map[ticker1], tickers_index_map[ticker2],
                                      returns_A = DATA["RETURNS"][ticker1],
                                      returns_B = DATA["RETURNS"][ticker2],
                                      prices_A = DATA["PRICES"][ticker1],
                                      prices_B = DATA["PRICES"][ticker2],
                                      spreads_A = DATA["SPREADS"][ticker1],
                                      spreads_B = DATA["SPREADS"][ticker2]))
        
        nb_pairs = len(pairs)
        nb_eval = min(nb_pairs, nb_pairs_cap) if nb_pairs_cap > 0 else nb_pairs
        print(f"  Constructed {nb_pairs} pairs, evaluating {nb_eval} pairs")
        
        # Trade pairs during the trading period
        count = 0
        for pair in pairs[:nb_eval]:
            count += 1
            if count % 5 == 0 or count == nb_eval:
                print(f"    Evaluating pair {count}/{nb_eval}: {pair.stock_A}-{pair.stock_B}")
            
            # Evaluate pair for each timestamp in trading period
            for t in range(trading_start, trading_end):
                # Only evaluate if we have enough history (window timestamps before t)
                if t >= window_lookback:
                    pair.evaluate(t,
                                  window_lookback, lambda_in, lambda_out, lambda_emergency, patience_max,
                                  np_positions)
        
        # Plot one random pair of each window
        if(display_figures and plot_pairs_lifetimes):
            print("  Plotting one random pair of that window:")
            pair_to_plot = np.random.randint(0, nb_eval-1)
            pairs[pair_to_plot].plot_lifetime_last(df_returns = DATA["RETURNS"],
                                    df_cum_returns = DATA["CUMRETURNS"])
        
        # Force exit all positions at the end of trading period
        print(f"  Forcing exit of all positions at end of window (t={trading_end-1})")
        if trading_end < total_timestamps:
            np_positions[trading_end, :] = 0

    # Do a column for sum of positions to check = 0
    np_positions[:, -1] = np_positions.sum(axis=1)
    
    print(f"\n4.4 All rolling windows completed.")
    print(f"  Final positions array shape: {np_positions.shape} (columns allocated for {len(TICKERS)} assets, +1 for row-wise sum).")

    if(display_figures):
        draw_positions_table(np_positions, TICKERS)

    # CONVERT POSITIONS TO DATAFRAME
    df_positions = pd.DataFrame(data=np_positions[:,:-1],
                            columns=TICKERS,
                            index=DATA["RETURNS"].index)
    
    # ============ COMPUTE STRATEGY PNL ============ 
    print("4.5 Computing PNL series...")
    pnl_curve = compute_strategy_pnl(DATA["RETURNS"], df_positions, DATA["PRICES"], DATA["SPREADS"])
    if(display_figures):
        plot_cumulative_pnl(pnl_curve)

    # Final message
    print("\n" + "="*70)
    print("PHASE 4 CORRECTLY TERMINATED")
    print("="*70 + "\n")
