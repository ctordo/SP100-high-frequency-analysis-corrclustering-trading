import numpy as np
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict


class DataPreprocessing:
    def __init__(self, folder_path, output_path, start_date=None, end_date=None):
        self.folder_path = folder_path  # Raw parquet data location
        self.output_path = output_path  # Clean output data location
        self.start_date = start_date  # Optional: filter start date
        self.end_date = end_date  # Optional: filter end date
    
    def process_single_asset(self, file_path):
        """
        Process a single asset through the cleaning pipeline
        
        Steps:
        1. Load raw data
        2. Convert datetime and set as index
        3. Filter by date 
        4. Keep only relevant columns (bid price, bid volume, ask price, ask volume)
        5. Calculate spread, mid price (volume weighted), classical mid price and volume imbalance
        6. VWAP aggregation for duplicate timestamps (when we have multiple trading occuring at same time, assemble data to same timestamp)
        7. Drop NaN rows
        8. Drop zero-volume rows
        
        Args:
            file_path: Path to raw parquet file
            
        Returns:
            Cleaned DataFrame
        """
        ticker = file_path.stem
        
        #Load data
        df = pd.read_parquet(file_path)
        
        #Convert to datetime and set it as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        #Filter by Date : [start_date:end_date] (3 last months)
        original_rows = len(df)

        df = df.loc[self.start_date:self.end_date]
        
        filtered_rows = len(df)
        
        #Keep only relevant columns
        columns_to_keep = ['bid-price', 'bid-volume', 'ask-price', 'ask-volume']
        df = df[columns_to_keep]
        
        #Drop zero volume rows : avoids division by zero in mid price volume weighted 
        #as well as avoiding / 0 in volume imbalance
        rows_before_zero = len(df)
        zero_volume_mask = (df['bid-volume'] == 0) | (df['ask-volume'] == 0)
        df = df[~zero_volume_mask]
        zero_volume_removed = rows_before_zero - len(df)
        
        #drop NaN rows
        rows_before_nan = len(df)
        df = df.dropna()
        nan_removed_early = rows_before_nan - len(df)
        
        #Create mid-price (volume-weighted average)
        df['mid-price (weighted-av)'] = (
            (df['bid-price'] * df['bid-volume'] + df['ask-price'] * df['ask-volume']) / 
            (df['bid-volume'] + df['ask-volume']))
        
        #VWAP aggregation for duplicate timestamps : 
        #If there are multiple rows with same timestamps, aggregate
        #them by doing an average
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            try:
                df = df.groupby(df.index).apply(lambda group: pd.Series({
                    'bid-price': np.average(group['bid-price'], weights=group['bid-volume']) if len(group) > 1 else group['bid-price'].iloc[0],
                    'ask-price': np.average(group['ask-price'], weights=group['ask-volume']) if len(group) > 1 else group['ask-price'].iloc[0],
                    'bid-volume': group['bid-volume'].sum(),
                    'ask-volume': group['ask-volume'].sum()}), include_groups=False)
            except Exception as e:
                # If VWAP fails, skip aggregation but continue processing
                print(f"  Warning: VWAP aggregation failed ({e}), skipping...")
        
        df['spread'] = df['ask-price'] - df['bid-price']
        df['mid-price'] = (df['bid-price'] + df['ask-price']) / 2
        df['volume_imbalance'] = (df['bid-volume'] - df['ask-volume']) / (df['bid-volume'] + df['ask-volume'])
        
        #Drop NaN
        rows_before_nan_final = len(df)
        df = df.dropna()
        nan_removed_final = rows_before_nan_final - len(df)
        
        final_columns = ['ask-price', 'ask-volume', 'bid-price', 'bid-volume','spread', 'mid-price', 'volume_imbalance']
        df = df[final_columns]
        
        return df, {'ticker': ticker,
            'original_rows': original_rows,
            'filtered_rows': filtered_rows,
            'duplicates_removed': duplicates,
            'nan_removed': nan_removed_early + nan_removed_final,
            'zero_volume_removed': zero_volume_removed,
            'final_rows': len(df)}
        
    
    def process_all_assets(self):
        """
        Process all assets in folder_path and save to output_path
        
        Creates clean parquet files with naming: {ticker}_clean.parquet
        """
        folder = Path(self.folder_path)
        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        parquet_files = list(folder.glob('*.parquet'))
        
        if not parquet_files:
            print(f"No parquet files found in '{self.folder_path}'")
            return
        
        print(f"{'='*70}")
        print(f"PROCESSING {len(parquet_files)} ASSETS")
        if self.start_date and self.end_date:
            print(f"Date filter: {self.start_date} to {self.end_date}")
        else:
            print(f"Date filter: Auto (first 3 months from each asset's start)")
        print(f"{'='*70}\n")
        
        results_summary = []
        
        for i, file in enumerate(parquet_files, 1):
            df_clean = None
            stats = None
            try:
                print(f"[{i}/{len(parquet_files)}] Processing {file.stem}...")
                
                # Process the asset
                df_clean, stats = self.process_single_asset(file)
                
                # Save cleaned data (even if it has issues)
                output_file = output_path / f"{file.stem}_clean.parquet"
                df_clean.to_parquet(output_file)
                
                # Print stats
                print(f"  Date range: {df_clean.index.min()} to {df_clean.index.max()}")
                print(f"  Rows: {stats['original_rows']:,} → {stats['filtered_rows']:,} (filtered) → {stats['final_rows']:,} (final)")
                print(f"  Removed - Duplicates: {stats['duplicates_removed']:,}, "
                      f"NaN: {stats['nan_removed']:,}, "
                      f"Zero-volume: {stats['zero_volume_removed']:,}")
                print(f"  Saved: {output_file.name}\n")
                
                results_summary.append(stats)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                
                # Try to save whatever data we have
                if df_clean is not None:
                    try:
                        output_file = output_path / f"{file.stem}_clean.parquet"
                        df_clean.to_parquet(output_file)
                        print(f"  Saved partial data to: {output_file.name}\n")
                        if stats:
                            results_summary.append(stats)
                    except Exception as save_error:
                        print(f"  Could not save file: {save_error}\n")
                else:
                    print(f"  No data to save\n")
        
        # Final summary
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {len(results_summary)}/{len(parquet_files)} assets")
        print(f"Output folder: {output_path}")
        
        total_rows = sum(s['final_rows'] for s in results_summary)
        total_duplicates = sum(s['duplicates_removed'] for s in results_summary)
        total_nan = sum(s['nan_removed'] for s in results_summary)
        total_zero = sum(s['zero_volume_removed'] for s in results_summary)
        
        print(f"\nTotal cleaned rows: {total_rows:,}")
        print(f"Total removed - Duplicates: {total_duplicates:,}, "
              f"NaN: {total_nan:,}, "
              f"Zero-volume: {total_zero:,}")
    
    def load_cleaned_data(self):
        """
        Load all cleaned data from output_path
        
        Returns:
            Dictionary with {ticker: DataFrame}
        """
        output_path = Path(self.output_path)
        
        if not output_path.exists():
            print(f"Error: Folder '{self.output_path}' does not exist!")
            print("Run process_all_assets() first.")
            return {}
        
        parquet_files = list(output_path.glob('*_clean.parquet'))
        
        if not parquet_files:
            print(f"No cleaned files found in '{self.output_path}'")
            return {}
        
        print(f"Loading {len(parquet_files)} cleaned assets...")
        
        data = {}
        for i, file in enumerate(parquet_files, 1):
            try:
                # Extract ticker (remove '_clean' suffix)
                ticker = file.stem.replace('_clean', '')
                
                # Load data
                df = pd.read_parquet(file)
                data[ticker] = df
                
                if (i % 10 == 0) or (i == len(parquet_files)):
                    print(f"  Loaded {i}/{len(parquet_files)} assets...")
                    
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        print(f"\n✓ Successfully loaded {len(data)} assets")
        
        return data
