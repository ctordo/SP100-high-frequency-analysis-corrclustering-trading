import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from datapreprocessing import DataPreprocessing
from covariances import CovarianceDenoiser
from covariances import run_covariance_denoise

# Define paths
# Define paths

data_local = "../../FBD_local_data/"
data_repo = "../Data/"

# Date filter (optional - if None, will use first 3 months from each asset)
start_date = '2008-09-01'  # Or None for auto
end_date = '2008-12-31'    # Or None for auto


#==========================================================
#----------------Data Preprocessing-----------------
# Initialize preprocessor object
preprocessing = DataPreprocessing(folder_path=raw_data_path, 
                                  output_path=clean_data_path,
                                  start_date=start_date,
                                  end_date=end_date)

# Process all assets 
preprocessing.process_all_assets()

# Load cleaned data
data_dict = preprocessing.load_cleaned_data()

print(f"\nLoaded {len(data_dict)} cleaned assets")
print(f"Example ticker: {list(data_dict.keys())[0]}")
print(f"\nExample data:")
print(data_dict[list(data_dict.keys())[0]].head())

#==========================================================

#==========================================================
#---------Covariances : dimensionality reduction ----------
input_path = './panel_data_1min.parquet'
covariance_denoise = CovarianceDenoiser(input_path=input_path)
run_covariance_denoise(input_path= input_path)

#==========================================================



#TODO : 
# - Time aggregation : volatility ? Resample 1,5,10,30min --> optimal granularity : chosen when we have least NaNs 
# - Epps effect 
# - How to estimate correlation matrix --> dimensionality reduction ? factors ? 
# - Investigate autocorrelation for intraday data