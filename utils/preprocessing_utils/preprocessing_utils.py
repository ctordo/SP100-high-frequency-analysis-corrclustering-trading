import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import date


class DataPreprocessingPolars:
    def __init__(self, folder_path, output_path, timestamp_path,  start_date=None, end_date=None):
        self.folder_path = Path(folder_path)  # Raw parquet data location
        self.output_path = Path(output_path)  # Clean output data location
        self.timestamp_path = Path(timestamp_path)
        self.start_date = start_date  # Filter start date
        self.end_date = end_date  # Filter end date
        self.assets_to_skip = None
    
    def process_all_assets(self):
        """
        Process all assets in folder_path and save to output_path
        
        Skips assets that already have cleaned files in output_path.
        Creates clean parquet files with naming: {ticker}_clean.parquet
        """
        

        self.output_path.mkdir(parents=True, exist_ok=True)
        
        parquet_files = list(self.folder_path.glob('*.parquet'))
        
        if not parquet_files:
            print(f"No parquet files found in '{self.folder_path}'")
            return
        
        # Check which files already exist
        existing_cleaned = set(f.stem.replace('_clean', '') for f in self.output_path.glob('*_clean.parquet'))
        files_to_process = [f for f in parquet_files if f.stem not in existing_cleaned]
        files_to_load = [f for f in parquet_files if f.stem in existing_cleaned]
        
        print(f"{'='*70}")
        print(f"PROCESSING {len(parquet_files)} ASSETS")
        if self.start_date and self.end_date:
            print(f"Date filter: {self.start_date} to {self.end_date}")
        else:
            print(f"Date filter: Auto (first 3 months from each asset's start)")
        print(f"{'='*70}")
        print(f"Files already cleaned: {len(files_to_load)}")
        print(f"Files to process: {len(files_to_process)}")
        print(f"{'='*70}\n")
        
        results_summary = []
        loaded_count = 0
        processed_count = 0
        
        # Check timestamps for all raw data files
        timestamp_csv_path = self.timestamp_path
        if not timestamp_csv_path.exists():
            print("Checking timestamps in raw data files...")
            df_timestamps = self.check_timestamps(self.folder_path)
            with pl.Config(tbl_rows=-1):
                print(df_timestamps)
            df_timestamps.write_csv(timestamp_csv_path)
            print(f"\n{'='*70}\n")
        else:
            print("Timestamp file already exists, loading it...")
            df_timestamps = pl.read_csv(timestamp_csv_path)
            print(f"\n{'='*70}\n")
        
        # Identify assets to skip based on incomplete data coverage
        # Expected full coverage: 2004-01-02 to 2008-12-31
 
        expected_start = date(2004, 1, 2)
        expected_end = date(2008, 12, 31)
        
        assets_to_skip = []
        for row in df_timestamps.iter_rows(named=True):
            asset = row['asset']
            first_ts = row['first_timestamp']
            last_ts = row['last_timestamp']
            
            # Convert to date
            if isinstance(first_ts, str):
                first_date = pl.Series([first_ts]).str.to_datetime().dt.date()[0]
                last_date = pl.Series([last_ts]).str.to_datetime().dt.date()[0]
            else:
                first_date = first_ts.date() if hasattr(first_ts, 'date') else first_ts
                last_date = last_ts.date() if hasattr(last_ts, 'date') else last_ts
            
            if first_date != expected_start or last_date != expected_end:
                assets_to_skip.append(asset)
        
        self.assets_to_skip = assets_to_skip
        print(f"Assets to skip due to incomplete data coverage: {len(assets_to_skip)}")
        print(f"Assets: {assets_to_skip}")
        print(f"\n{'='*70}\n")
      
        # Check missing data per asset per year
        if not self.timestamp_path.exists():
            print("Checking missing data per asset per year...")
            df_missing = self.check_missing_data_per_asset(self.folder_path, self.assets_to_skip)
            if df_missing is not None:
                with pl.Config(tbl_rows=-1):
                    print(df_missing)
                df_missing.write_csv(self.timestamp_path)
                print(f"Missing data report saved to {self.timestamp_path}")
            print(f"\n{'='*70}\n")
        else:
            print("Missing data report already exists, skipping analysis...")
            print(f"\n{'='*70}\n")
       
        
        for i, file in enumerate(parquet_files, 1):
            # Skip assets with incomplete data
            if file.stem in self.assets_to_skip:
                print(f"[{i}/{len(parquet_files)}] Skipping {file.stem} (incomplete data coverage)\n")
                continue
            
            df_clean = None
            stats = None
            try:
                output_file = self.output_path / f"{file.stem}_clean.parquet"
                
                # Check if file exists
                if output_file.exists():
                    print(f"[{i}/{len(parquet_files)}] Loading {file.stem}... (already cleaned)")
                    df_clean = pl.read_parquet(output_file)
                    stats = {
                        'ticker': file.stem,
                        'original_rows': None,
                        'filtered_rows': None,
                        'duplicates_removed': None,
                        'nan_removed': None,
                        'zero_volume_removed': None,
                        'final_rows': df_clean.height
                    }
                    loaded_count += 1
                    print(f"  Loaded: {stats['final_rows']:,} rows\n")
                else:
                    print(f"[{i}/{len(parquet_files)}] Processing {file.stem}...")
                    
                    # Process the asset
                    df_clean, stats = self.process_single_asset(file)
                  
                    # Save cleaned data
                    df_clean.write_parquet(output_file)
                    
                    # Print stats
                    timestamp_min = df_clean.select(pl.col('timestamp').min()).item()
                    timestamp_max = df_clean.select(pl.col('timestamp').max()).item()
                    
                    print(f"  Date range: {timestamp_min} to {timestamp_max}")
                    print(f"  Rows: {stats['original_rows']:,} → {stats['filtered_rows']:,} (filtered) → {stats['final_rows']:,} (final)")
                    print(f"  Removed - Duplicates: {stats['duplicates_removed']:,}, "
                          f"NaN: {stats['nan_removed']:,}, "
                          f"Zero-volume: {stats['zero_volume_removed']:,}")
                    print(f"  Saved: {output_file.name}\n")
                    
                    processed_count += 1
                
                results_summary.append(stats)
                
            except Exception as e:
                print(f"  ERROR: {e}")
                
                # Try to save whatever data we have
                if df_clean is not None:
                    try:
                        output_file = self.output_path / f"{file.stem}_clean.parquet"
                        df_clean.write_parquet(output_file)
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
        print(f"Total assets: {len(parquet_files)}")
        print(f"  - Loaded (already cleaned): {loaded_count}")
        print(f"  - Processed (newly cleaned): {processed_count}")
        print(f"Output folder: {self.output_path}")
        
        processed_results = [s for s in results_summary if s.get('original_rows') is not None]
        
        if processed_results:

            total_original = sum(s['original_rows'] for s in processed_results)
            total_filtered = sum(s['filtered_rows'] for s in processed_results)
            total_final = sum(s['final_rows'] for s in processed_results)
            total_duplicates = sum(s['duplicates_removed'] for s in processed_results)
            total_nan = sum(s['nan_removed'] for s in processed_results)
            total_zero = sum(s['zero_volume_removed'] for s in processed_results)
            
            # Calculate average percentage removed
            total_removed = total_original - total_final
            avg_pct_removed = (total_removed / total_original) * 100 if total_original > 0 else 0
            
            print(f"\nProcessed files statistics:")
            print(f"Total original rows: {total_original:,}")
            print(f"Total after date filter: {total_filtered:,}")
            print(f"Total cleaned rows: {total_final:,}")
            print(f"Total rows removed: {total_removed:,} ({avg_pct_removed:.2f}%)")
            print(f"Total removed - Duplicates: {total_duplicates:,}, "
                  f"NaN: {total_nan:,}, "
                  f"Zero-volume: {total_zero:,}")
    
    def process_single_asset(self, file_path):
        """
        Process a single asset through the cleaning pipeline using Polars
        
        Steps:
        1. Load raw data
        2. Convert datetime and set as index
        3. Filter by date
        4. Keep only relevant columns
        5. VWAP aggregation for duplicate timestamps
        6. Calculate spread and volume imbalance
        7. Drop NaN rows
        8. Drop zero-volume rows
        
        Args:
            file_path: Path to raw parquet file
            
        Returns:
            Cleaned DataFrame and stats dictionary
        """
        ticker = file_path.stem
        
        #Load data with Polars
        df = pl.read_parquet(file_path)
        
        #replace the column 'timestamp' by the same column timestamp in datetime format
        df = df.with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f'))
        
        # Filter by Date: [start_date:end_date]
        original_rows = df.height #df.shape[0] == vertical length
        #Original_rows == How many rows we have at the beginning

                    
        #Condition if we did not specify any start and end_date : takes first of the column and add 3 months ~ 90 days
        if self.start_date and self.end_date:
            # Convert date strings to datetime
            start = pl.lit(self.start_date).str.to_datetime(format='%Y-%m-%d')
            end = pl.lit(self.end_date).str.to_datetime(format='%Y-%m-%d')
            df = df.filter((pl.col('timestamp') >= start) & (pl.col('timestamp') <= end))
        else:
            first_date = df.select(pl.col('timestamp').min()).item()
            end_date_auto = first_date + pl.duration(days=90)  # ~3 months
            df = df.filter((pl.col('timestamp') >= first_date) & (pl.col('timestamp') <= end_date_auto))
        #Filtered_rows == How many rows we have after selecting 3 months only 
        filtered_rows = df.height
        
        # Keep only relevant columns
        columns_to_keep = ['timestamp', 'bid-price', 'bid-volume', 'ask-price', 'ask-volume']
        df = df.select(columns_to_keep)
        
        # Drop zero volume rows: avoids division by zero when averaging by volume
        rows_before_zero = df.height
        df = df.filter(
            (pl.col('bid-volume') != 0) & 
            (pl.col('ask-volume') != 0)
        )
        zero_volume_removed = rows_before_zero - df.height
        
        # Drop NaN rows
        rows_before_nan = df.height
        df = df.drop_nulls()
        nan_removed_early = rows_before_nan - df.height
        
        # Create mid-price (volume-weighted average)
        df = df.with_columns([
            ((pl.col('bid-price') * pl.col('bid-volume') + 
              pl.col('ask-price') * pl.col('ask-volume')) / 
             (pl.col('bid-volume') + pl.col('ask-volume')))
            .alias('mid-price (weighted-av)')
        ])
        
        # VWAP aggregation for duplicate timestamps
        duplicates = df.height - df.select(pl.col('timestamp')).unique().height
        
        if duplicates > 0:
            try:
                df = df.group_by('timestamp').agg([
                    # Weighted averages for bid and ask prices
                    (pl.col('bid-price') * pl.col('bid-volume')).sum() / pl.col('bid-volume').sum()
                    .alias('bid-price'),
                    
                    (pl.col('ask-price') * pl.col('ask-volume')).sum() / pl.col('ask-volume').sum()
                    .alias('ask-price'),
                    
                    # Sum volumes
                    pl.col('bid-volume').sum().alias('bid-volume'),
                    pl.col('ask-volume').sum().alias('ask-volume'),
                ])
                
                # Recalculate mid-price (weighted-av) after aggregation
                df = df.with_columns([
                    ((pl.col('bid-price') * pl.col('bid-volume') + 
                      pl.col('ask-price') * pl.col('ask-volume')) / 
                     (pl.col('bid-volume') + pl.col('ask-volume')))
                    .alias('mid-price (weighted-av)')
                ])
                
            except Exception as e:
                print(f"  Warning: VWAP aggregation failed ({e}), skipping...")
        
        # Calculate spread, mid-price, and volume imbalance
        df = df.with_columns([
            (pl.col('ask-price') - pl.col('bid-price')).alias('spread'),
            ((pl.col('bid-price') + pl.col('ask-price')) / 2).alias('mid-price'),
            ((pl.col('bid-volume') - pl.col('ask-volume')) / 
             (pl.col('bid-volume') + pl.col('ask-volume'))).alias('volume_imbalance')
        ])
        
        # Drop NaN (final cleanup)
        rows_before_nan_final = df.height
        df = df.drop_nulls()
        nan_removed_final = rows_before_nan_final - df.height
        
        # Reorder columns (timestamp first, then others)
        final_columns = ['timestamp', 'ask-price', 'ask-volume', 'bid-price', 'bid-volume',
                        'spread', 'mid-price', 'volume_imbalance']
        df = df.select(final_columns)
        
        # Sort by timestamp
        df = df.sort('timestamp')
        
        return df, {
            'ticker': ticker,
            'original_rows': original_rows,
            'filtered_rows': filtered_rows,
            'duplicates_removed': duplicates,
            'nan_removed': nan_removed_early + nan_removed_final,
            'zero_volume_removed': zero_volume_removed,
            'final_rows': df.height
        }
    

    def check_timestamps(self, raw_data_path):
        """
        Check first and last timestamps for all parquet files in the raw data folder.
        Useful to detect whether some assets left or entered the SP100 during the period of interests
        
        Args:
            raw_data_path: Path to folder containing raw parquet files
            
        Returns:
            DataFrame with columns: asset, first_timestamp, last_timestamp
        """
        folder = Path(raw_data_path)
        parquet_files = list(folder.glob('*.parquet'))
        
        timestamp_data = []
        
        for file in parquet_files:
            try:
                df = pl.read_parquet(file)
                # Convert timestamp to datetime if it's a string
                if df.schema['timestamp'] == pl.Utf8:
                    df = df.with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f'))
                
                first_ts = df.select(pl.col('timestamp').min()).item()
                last_ts = df.select(pl.col('timestamp').max()).item()
                
                timestamp_data.append({
                    'asset': file.stem,
                    'first_timestamp': first_ts,
                    'last_timestamp': last_ts
                })
            except Exception as e:
                print(f"Error reading {file.stem}: {e}")
                timestamp_data.append({
                    'asset': file.stem,
                    'first_timestamp': None,
                    'last_timestamp': None
                })
        
        # Create DataFrame from collected data
        df_timestamps = pl.DataFrame(timestamp_data)
        return df_timestamps.sort('asset')

    
    def check_missing_data_per_asset(self, raw_data_path, assets_to_skip):
        """
        Check for missing dates in each asset per year (2004-2008).
        
        For each asset, we identify the expected trading dates based on the union
        of all dates present across all assets, then count how many dates are missing.
        
        Args:
            raw_data_path: Path to folder containing raw parquet files
            assets_to_skip: List of asset names to exclude from the check
            
        Returns:
            DataFrame with columns: asset_name, 2004, 2005, 2006, 2007, 2008
        """
        folder = Path(raw_data_path)
        parquet_files = [f for f in folder.glob('*.parquet') if f.stem not in assets_to_skip]
        
        if not parquet_files:
            print("No valid parquet files found for missing data check")
            return None
        
        print(f"Analyzing missing data across {len(parquet_files)} assets...")
        
        #First we collect all dates from all assets to determine "expected" trading dates
        # By that we mean : If Asset A has January 2,3,5 and Asset B has January 1,2,3,4,5,6 
        # then expected trading dates are 1,2,3,4,5,6 and we can determine that A has some missing values

        all_dates_union = set()
        asset_dates = {}
        
        for file in parquet_files:
            try:
                df = pl.read_parquet(file)
                # Convert timestamp to datetime if needed
                if df.schema['timestamp'] == pl.Utf8:
                    df = df.with_columns(pl.col('timestamp').str.to_datetime(format='%Y-%m-%d %H:%M:%S%.f'))
                
                # Extract unique dates (date only, without time)
                unique_dates = set(df.select(pl.col('timestamp').dt.date()).unique().to_series().to_list())
                asset_dates[file.stem] = unique_dates
                all_dates_union.update(unique_dates)
            except Exception as e:
                print(f"Error reading {file.stem}: {e}")
        
        # Group expected dates by year
        expected_dates_by_year = defaultdict(set)
        for date in all_dates_union:
            if 2004 <= date.year <= 2008:
                expected_dates_by_year[date.year].add(date)
        
        # Second pass: count missing dates per asset per year
        missing_data_results = []
        
        for asset_name, dates in asset_dates.items():
            row = {'asset_name': asset_name}
            
            for year in [2004, 2005, 2006, 2007, 2008]:
                expected = expected_dates_by_year[year]
                actual = {d for d in dates if d.year == year}
                missing = expected - actual
                row[str(year)] = len(missing)
            
            missing_data_results.append(row)
        
        # Create DataFrame
        df_missing = pl.DataFrame(missing_data_results)
        df_missing = df_missing.sort('asset_name')
        
        return df_missing

    

    def resample_with_vwap(self, df, interval='1min'):
        """
        Resample a DataFrame to a specified time interval using VWAP aggregation
        
        This function aggregates high-frequency tick data into lower frequency bars
        using volume-weighted average prices (VWAP) to preserve price-volume information.
        
        Args:
            df: Polars DataFrame with 'timestamp' column and price/volume columns
            interval: Time interval for resampling. Options:
                     '1min', '5min', '15min', '30min', '1h', etc.
                     
        Returns:
            Resampled Polars DataFrame with VWAP-aggregated values
        """
        # Parse interval to polars duration
        interval_map = {
            '1min': '1m',
            '5min': '5m', 
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '1hour': '1h'
        }
        
        pl_interval = interval_map.get(interval, interval)
        
        # Sort by timestamp first
        df = df.sort('timestamp')
        
        # Group by time interval and aggregate using VWAP
        df_resampled = df.group_by_dynamic(
            'timestamp',
            every=pl_interval
        ).agg([
            # Volume-weighted average prices (VWAP)
            (pl.col('bid-price') * pl.col('bid-volume')).sum() / pl.col('bid-volume').sum()
            .alias('bid-price'),
            
            (pl.col('ask-price') * pl.col('ask-volume')).sum() / pl.col('ask-volume').sum()
            .alias('ask-price'),
            
            # Sum volumes across the interval
            pl.col('bid-volume').sum().alias('bid-volume'),
            pl.col('ask-volume').sum().alias('ask-volume'),
            
            # Count number of observations in each interval
            pl.count().alias('n_obs')
        ])
        
        # Recalculate derived features (spread, mid-price, volume_imbalance)
        df_resampled = df_resampled.with_columns([
            (pl.col('ask-price') - pl.col('bid-price')).alias('spread'),
            ((pl.col('bid-price') + pl.col('ask-price')) / 2).alias('mid-price'),
            ((pl.col('bid-volume') - pl.col('ask-volume')) / 
             (pl.col('bid-volume') + pl.col('ask-volume'))).alias('volume_imbalance')
        ])
        
        # Drop rows with NaN (can happen if no data in that interval)
        df_resampled = df_resampled.drop_nulls()
        
        return df_resampled
    
    def create_panel_data(self, resample_interval=None):
        """
        Concatenate all cleaned assets into a single panel DataFrame
        
        Creates a long-format DataFrame with:
        - timestamp: datetime
        - ticker: asset ticker (without .N suffix)
        - ask-price, ask-volume, bid-price, bid-volume, spread, mid-price, volume_imbalance
        
        Args:
            resample_interval: Optional time interval for VWAP resampling before concatenation.
                             Options: '1min', '5min', '15min', '30min', '1h'
                             If None, no resampling is performed (uses original tick data)            
        Returns:
            Polars DataFrame in long format (timestamp x ticker)
        """
        
        
        if not self.output_path.exists():
            print(f"Error: Folder '{self.output_path}' does not exist!")
            print("Run process_all_assets() first.")
            return None
        
        parquet_files = list(self.output_path.glob('*_clean.parquet'))
        
        if not parquet_files:
            print(f"No cleaned files found in '{self.output_path}'")
            return None
        
        # Filter out assets that should be skipped
        if self.assets_to_skip:
            parquet_files = [f for f in parquet_files if f.stem.replace('_clean', '') not in self.assets_to_skip]
            print(f"Excluding {len(self.assets_to_skip)} assets from panel: {self.assets_to_skip}")
        
        print(f"{'='*70}")
        print(f"CREATING PANEL DATA FROM {len(parquet_files)} ASSETS")
        if resample_interval:
            print(f"With VWAP resampling to {resample_interval} intervals")
        print(f"{'='*70}\n")
        
        # Load and concatenate all assets
        panel_frames = []
        
        for i, file in enumerate(parquet_files, 1):
            try:
                # Extract ticker (remove '_clean' and '.N' suffixes)
                ticker_with_suffix = file.stem.replace('_clean', '')
                ticker = ticker_with_suffix.replace('.N', '')
                
                # Load data
                df = pl.read_parquet(file)
                
                # Apply resampling if specified
                if resample_interval is not None:
                    original_rows = df.height
                    df = self.resample_with_vwap(df, resample_interval)
                    if (i % 10 == 0) or (i == len(parquet_files)):
                        print(f"  [{file.stem}] Resampled: {original_rows:,} → {df.height:,} rows")
                
                # Ensure consistent data types (cast all numeric columns to Float64)
                df = df.with_columns([
                    pl.col('ask-price').cast(pl.Float64),
                    pl.col('ask-volume').cast(pl.Float64),
                    pl.col('bid-price').cast(pl.Float64),
                    pl.col('bid-volume').cast(pl.Float64),
                    pl.col('spread').cast(pl.Float64),
                    pl.col('mid-price').cast(pl.Float64),
                    pl.col('volume_imbalance').cast(pl.Float64)
                ])
                
                # Add ticker column
                df = df.with_columns(pl.lit(ticker).alias('ticker'))
                
                panel_frames.append(df)
                
                if (i % 10 == 0) or (i == len(parquet_files)):
                    print(f"  Loaded {i}/{len(parquet_files)} assets...")
                    
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        # Concatenate all assets
        print(f"\nConcatenating {len(panel_frames)} assets...")
        panel_df = pl.concat(panel_frames, how='vertical_relaxed')
        
        # Sort by timestamp and ticker
        panel_df = panel_df.sort(['timestamp', 'ticker'])
        
        # Reorder columns: timestamp, ticker, features
        column_order = [
            'timestamp', 'ticker',
            'ask-price', 'ask-volume', 
            'bid-price', 'bid-volume',
            'spread', 'mid-price', 'volume_imbalance'
        ]
        panel_df = panel_df.select(column_order)
        
        print(f"\n{'='*70}")
        print("PANEL DATA CREATED")
        print(f"{'='*70}")
        print(f"Shape: {panel_df.shape}")
        print(f"Date range: {panel_df.select(pl.col('timestamp').min()).item()} to {panel_df.select(pl.col('timestamp').max()).item()}")
        print(f"Number of assets: {panel_df.select(pl.col('ticker').n_unique()).item()}")
        
        return panel_df

