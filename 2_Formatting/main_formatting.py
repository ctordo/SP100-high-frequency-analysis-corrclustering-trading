import pandas as pd

import importlib
import formatting_utils
importlib.reload(formatting_utils)
from formatting_utils import panel_to_table, plot_daily_nan_proportion_heatmap

data_local = "../../FBD_local_data/"
data_repo = "../Data/"

def main_2(display_figures:bool = False):

    # Initial message
    print("="*70)
    print("ENTERING PHASE 2: FORMATTING")
    print("="*70 + "\n")

    # ============ LOAD PHASE INPUTS ============ 
    print("2.1 Loading phase inputs...")
    
    # Panel
    df_panel1min = pd.read_parquet(data_local + "panel_data_1min.parquet")
    ("  panel_data_1min.parquet loaded as DataFrame. Overview:")
    display(df_panel1min)
    

    # ============ FORMAT STOCK PRICES ============ 
    print("2.2 Table restructuring")
    
    print("  Apply pivoting on prices by ticker aggregation...")
    df_price = panel_to_table(df_panel1min, attribute="mid-price", aggfunc="last", disp=False)

    print("  Resulting DataFrame:")
    display(df_price)
    
    #Retrieve TICKERS
    TICKERS = df_price.columns.to_list()[1:]

    # Write csv in repo
    print(f"  Writing new dataframe at {data_local}...")
    df_price.to_csv(data_local + "stock_prices.csv")
    print(f"  File successfully written.")
    
    # ============ NAN EVALUATION ============ 
    print("2.3 Missing values evaluation")
    if(display_figures):
        plot_daily_nan_proportion_heatmap(df_price, "PRICE", TICKERS, gradient_count="Blues", gradient_prop="Greens")

    # Final message
    print("\n" + "="*70)
    print("PHASE 2 CORRECTLY TERMINATED")
    print("="*70 + "\n")
