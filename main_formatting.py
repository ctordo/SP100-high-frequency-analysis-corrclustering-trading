import pandas as pd

from utils.formatting_utils.formatting_utils import panel_to_table, plot_daily_nan_proportion_heatmap

data_local = "../FBD_local_data/"

def main_2(display_figures:bool = False):

    # Initial message
    print("="*70)
    print("ENTERING PHASE 2: FORMATTING")
    print("="*70)

    # ============ LOAD PHASE INPUTS ============ 
    print("="*50 + "\n2.1 Loading phase inputs")
    
    # Panel
    df_panel1min = pd.read_parquet(data_local + "panel_data_1min.parquet")
    ("  panel_data_1min.parquet loaded as DataFrame.")
    
    if(display_figures):
        display(df_panel1min)
    

    # ============ FORMAT STOCK PRICES ============ 
    print("="*50 + "\n2.2 Table restructuring")
    
    print("  Build DataFrame of stock prices (pivoting panel by ticker aggregation)...")
    df_prices = panel_to_table(df_panel1min, attribute="mid-price", aggfunc="last", disp=False)
    print("  Build DataFrame of stock spreads (pivoting panel by ticker aggregation)...")
    df_spreads = panel_to_table(df_panel1min, attribute="spread", aggfunc="last", disp=False)

    if(display_figures):
        print("  Resulting DataFrames:")
        display(df_prices)
        display(df_spreads)
    
    #Retrieve TICKERS
    TICKERS = df_prices.columns.to_list()[1:]

    # ============ NAN EVALUATION ============ 
    print("="*50 + "\n2.3 Missing values evaluation")
    if(display_figures):
        plot_daily_nan_proportion_heatmap(df_prices, "PRICE", TICKERS, gradient_count="Blues", gradient_prop="Greens")
        plot_daily_nan_proportion_heatmap(df_spreads, "PRICE", TICKERS, gradient_count="Reds", gradient_prop="Greens")

    print("  Forward-filling NaN values. Warning: Keeps first-entry NaN values.")
    df_prices = df_prices.ffill()
    df_spreads = df_spreads.ffill()

    # Sanity check: replot NaN heatmaps to verify filling
    if(display_figures):
        plot_daily_nan_proportion_heatmap(df_prices, "PRICE", TICKERS, gradient_count="Blues", gradient_prop="Greens")
        plot_daily_nan_proportion_heatmap(df_spreads, "SPREADS", TICKERS, gradient_count="Reds", gradient_prop="Greens")

    # Cut off the first day to remove first-entry NaN values
    print(f"  Cutting off the first day to remove first-entry NaN values...")
    cutoff_date = "2008-09-03"
    df_prices = df_prices[(df_prices.index >= cutoff_date)]
    df_spreads = df_spreads[(df_spreads.index >= cutoff_date)]

    # Sanity check: replot NaN heatmaps to verify cutoff
    if(display_figures):
        plot_daily_nan_proportion_heatmap(df_prices, "PRICE", TICKERS, gradient_count="Blues", gradient_prop="Greens")
        plot_daily_nan_proportion_heatmap(df_spreads, "SPREADS", TICKERS, gradient_count="Reds", gradient_prop="Greens")


    # ============ COMPUTING RETURNS ============ 
    print("="*50 + "\n2.4 Build DataFrame of stock returns (pct changes of prices)")
    df_returns = df_prices.pct_change()


    # ============ FILES WRITING ============ 
    print("="*50 + f"\n2.5 Writing .csv files at [{data_local}]")
    count=0  
    df_prices.to_csv(data_local + "stock_prices.csv")
    count+=1
    df_returns.to_csv(data_local + "stock_returns.csv")
    count+=1
    df_spreads.to_csv(data_local + "stock_spreads.csv")
    count+=1
    print(f"  {count} files successfully written.")

    # Final message
    print("\n" + "="*70)
    print("PHASE 2 CORRECTLY TERMINATED")
    print("="*70 + "\n")
