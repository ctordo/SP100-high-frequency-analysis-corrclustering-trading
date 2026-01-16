"""
Test script to run the Polars-based data preprocessing pipeline
and compare performance with Pandas version
"""
from datapreprocessing_polars import DataPreprocessingPolars
import time

data_local = "../../FBD_local_data/"
data_repo = "../Data/"

#Industry Mapping path
industry_mapping = data_local + 'industry_mapping.csv'

#Original raw parquet path 
raw_parquet_path = data_local + "Data_parquet/"

#Clean parquet path 
clean_parquet_path = data_local+ "Clean_Data_parquet/"

# Date filter
start_date = '2008-09-01'
end_date = '2008-12-31'

#Initialize preprocessor object
print("Initializing data preprocessor...")

preprocessing = DataPreprocessingPolars(
        folder_path= raw_parquet_path,
        output_path= clean_parquet_path,
        start_date=start_date,
        end_date=end_date)

# Process all assets
#Step 1: Process all assets (if not already done)
print("\nStep 1: Processing/Loading cleaned assets...")
preprocessing.process_all_assets()

#Step 2: Compare different resampling intervals
print("Step 2: Comparing different resampling intervals") 
intervals_to_test = ['1min']#, '5min', '15min', '30min']   

comparison_results = preprocessing.compare_resampling_intervals(industry_mapping_path=industry_mapping,
                                                                intervals=intervals_to_test)

#Step 3: Create panel data with chosen interval
print("Step 3: Creating final panel data with chosen interval")
chosen_interval = '1min'
panel = preprocessing.create_panel_data(
        industry_mapping_path=industry_mapping,
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
"""
#==========================================================
#---------Covariances : dimensionality reduction ----------
input_path = './panel_data_1min.parquet'
window_ret, cov_denoised, info, tickers = run_covariance_denoise(input_path= input_path)

#==========================================================
"""


