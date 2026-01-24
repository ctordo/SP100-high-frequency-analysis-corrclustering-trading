"""
Test script to run the Polars-based data preprocessing pipeline
and compare performance with Pandas version
"""
from preprocessing_utils import DataPreprocessingPolars
import time

def main_1(start_date='2008-09-01', end_date='2008-12-31', chosen_interval='1min'):
    """
    Main function to run the complete preprocessing pipeline.
    
    Args:
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        chosen_interval: Resampling interval (e.g., '1min', '5min', '15min')
    
    Returns:
        panel: The final panel data DataFrame
    """
    data_local = "../../FBD_local_data/"

    #Original raw parquet path 
    raw_parquet_path = data_local + "Data_parquet/"

    #Clean parquet path 
    clean_parquet_path = data_local+ "Clean_Data_parquet/"

    timestamp_path = data_local+ "sp100_timestamps.csv"

    #Initialize preprocessor object
    print("Initializing data preprocessor...")

    preprocessing = DataPreprocessingPolars(
            folder_path= raw_parquet_path,
            output_path= clean_parquet_path,
            timestamp_path=timestamp_path,
            start_date=start_date,
            end_date=end_date)

    # Process all assets
    #Step 1: Process all assets (if not already done)
    print("\nStep 1: Processing/Loading cleaned assets...")
    preprocessing.process_all_assets()

    #Step 2: Create panel data with chosen interval
    print("Step 2: Creating final panel data with chosen interval")
    panel = preprocessing.create_panel_data(
            resample_interval=chosen_interval)

    #Save the panel data
    #Save in local_data + "panel_data/" + "panel_data_{chosen_interval}.parquet"
    if panel is not None:
        output_file = data_local +  f"panel_data_{chosen_interval}.parquet"
        print(f"\nSaving panel data to {output_file}...")
        panel.write_parquet(output_file)
        print(f"Saved {panel.height:,} rows to {output_file}")
    else:
        print("\nNo panel data created - check that data files exist in the specified folder")
    
    return panel


if __name__ == "__main__":
    # Run with default parameters when executed as a script
    panel = main_1()



