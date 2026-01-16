import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict


class DataPreprocessingPolars:
    def __init__(self, folder_path, output_path, start_date=None, end_date=None):
        self.folder_path = folder_path  # Raw parquet data location
        self.output_path = output_path  # Clean output data location
        self.start_date = start_date  # Optional: filter start date
        self.end_date = end_date  # Optional: filter end date
    
    def process_single_asset(self, file_path):
        """
        Process a single asset through the cleaning pipeline using Polars
        
        Steps:
        1. Load raw data
        2. Convert datetime and set as index
        3. Filter by date (first 3 months if no dates provided)
        4. Keep only relevant columns
        5. Create mid-price (volume-weighted)
        6. VWAP aggregation for duplicate timestamps
        7. Calculate spread and volume imbalance
        8. Drop NaN rows
        9. Drop zero-volume rows
        
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
    
    def _process_or_load_asset(self, file, output_path):
        """
        Helper function: Check if cleaned file exists, load it if yes, process if no
        
        Args:
            file: Path to raw parquet file
            output_path: Path object to output directory
            
        Returns:
            Tuple of (df_clean, stats, was_loaded)
        """
        output_file = output_path / f"{file.stem}_clean.parquet"
        
        # Check if cleaned file already exists
        if output_file.exists():
            try:
                # Load existing cleaned file
                df_clean = pl.read_parquet(output_file)
                
                # Create stats from loaded file
                stats = {
                    'ticker': file.stem,
                    'original_rows': None,  # Not available for loaded files
                    'filtered_rows': None,
                    'duplicates_removed': None,
                    'nan_removed': None,
                    'zero_volume_removed': None,
                    'final_rows': df_clean.height
                }
                
                return df_clean, stats, True  # True = was loaded
                
            except Exception as e:
                print(f"  Warning: Could not load existing file, will reprocess. Error: {e}")
                # Fall through to processing
        
        # Process the asset (file doesn't exist or loading failed)
        df_clean, stats = self.process_single_asset(file)
        
        # Save cleaned data
        df_clean.write_parquet(output_file)
        
        return df_clean, stats, False  # False = was processed
    
    def process_all_assets(self):
        """
        Process all assets in folder_path and save to output_path
        
        Skips assets that already have cleaned files in output_path.
        Creates clean parquet files with naming: {ticker}_clean.parquet
        """
        folder = Path(self.folder_path)
        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        parquet_files = list(folder.glob('*.parquet'))
        
        if not parquet_files:
            print(f"No parquet files found in '{self.folder_path}'")
            return
        
        # Check which files already exist
        existing_cleaned = set(f.stem.replace('_clean', '') for f in output_path.glob('*_clean.parquet'))
        files_to_process = [f for f in parquet_files if f.stem not in existing_cleaned]
        files_to_load = [f for f in parquet_files if f.stem in existing_cleaned]
        
        print(f"{'='*70}")
        print(f"PROCESSING {len(parquet_files)} ASSETS WITH POLARS")
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
        
        for i, file in enumerate(parquet_files, 1):
            df_clean = None
            stats = None
            try:
                output_file = output_path / f"{file.stem}_clean.parquet"
                
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
                        output_file = output_path / f"{file.stem}_clean.parquet"
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
        print(f"Output folder: {output_path}")
        
        # Calculate totals only from processed files (loaded files don't have these stats)
        processed_results = [s for s in results_summary if s.get('original_rows') is not None]
        
        if processed_results:
            total_rows = sum(s['final_rows'] for s in processed_results)
            total_duplicates = sum(s['duplicates_removed'] for s in processed_results)
            total_nan = sum(s['nan_removed'] for s in processed_results)
            total_zero = sum(s['zero_volume_removed'] for s in processed_results)
            
            print(f"\nProcessed files statistics:")
            print(f"Total cleaned rows: {total_rows:,}")
            print(f"Total removed - Duplicates: {total_duplicates:,}, "
                  f"NaN: {total_nan:,}, "
                  f"Zero-volume: {total_zero:,}")
    
    def load_cleaned_data(self):
        """
        Load all cleaned data from output_path
        
        Returns:
            Dictionary with {ticker: Polars DataFrame}
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
        
        print(f"Loading {len(parquet_files)} cleaned assets with Polars...")
        
        data = {}
        for i, file in enumerate(parquet_files, 1):
            try:
                # Extract ticker (remove '_clean' suffix)
                ticker = file.stem.replace('_clean', '')
                
                # Load data with Polars
                df = pl.read_parquet(file)
                data[ticker] = df
                
                if (i % 10 == 0) or (i == len(parquet_files)):
                    print(f"  Loaded {i}/{len(parquet_files)} assets...")
                    
            except Exception as e:
                print(f"  Error loading {file.name}: {e}")
        
        print(f"\n✓ Successfully loaded {len(data)} assets")
        
        return data
    
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
    
    def create_panel_data(self, industry_mapping_path='industry_mapping.csv', resample_interval=None):
        """
        Concatenate all cleaned assets into a single panel DataFrame
        
        Creates a long-format DataFrame with:
        - timestamp: datetime
        - ticker: asset ticker (without .N suffix)
        - ask-price, ask-volume, bid-price, bid-volume, spread, mid-price, volume_imbalance
        - industry: industry classification from CSV
        
        Args:
            industry_mapping_path: Path to CSV with ticker-industry mapping            resample_interval: Optional time interval for VWAP resampling before concatenation.
                             Options: '1min', '5min', '15min', '30min', '1h'
                             If None, no resampling is performed (uses original tick data)            
        Returns:
            Polars DataFrame in long format (timestamp x ticker)
        """
        output_path = Path(self.output_path)
        
        if not output_path.exists():
            print(f"Error: Folder '{self.output_path}' does not exist!")
            print("Run process_all_assets() first.")
            return None
        
        parquet_files = list(output_path.glob('*_clean.parquet'))
        
        if not parquet_files:
            print(f"No cleaned files found in '{self.output_path}'")
            return None
        
        print(f"{'='*70}")
        print(f"CREATING PANEL DATA FROM {len(parquet_files)} ASSETS")
        if resample_interval:
            print(f"With VWAP resampling to {resample_interval} intervals")
        print(f"{'='*70}\n")
        
        # Load industry mapping
        print(f"Loading industry mapping from {industry_mapping_path}...")
        industry_df = pl.read_csv(industry_mapping_path)
        ticker_col = industry_df.columns[0]
        industry_col = industry_df.columns[1]
        
        # Create mapping dictionary (ticker -> industry)
        industry_map = dict(zip(
            industry_df[ticker_col].to_list(),
            industry_df[industry_col].to_list()
        ))
        
        print(f"Loaded {len(industry_map)} industry mappings\n")
        
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
                
                # Add industry column
                industry = industry_map.get(ticker, 'Unknown')
                df = df.with_columns(pl.lit(industry).alias('industry'))
                
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
        
        # Reorder columns: timestamp, ticker, features, industry
        column_order = [
            'timestamp', 'ticker',
            'ask-price', 'ask-volume', 
            'bid-price', 'bid-volume',
            'spread', 'mid-price', 'volume_imbalance',
            'industry'
        ]
        panel_df = panel_df.select(column_order)
        
        print(f"\n{'='*70}")
        print("PANEL DATA CREATED")
        print(f"{'='*70}")
        print(f"Shape: {panel_df.shape}")
        print(f"Date range: {panel_df.select(pl.col('timestamp').min()).item()} to {panel_df.select(pl.col('timestamp').max()).item()}")
        print(f"Number of assets: {panel_df.select(pl.col('ticker').n_unique()).item()}")
        print(f"Number of industries: {panel_df.select(pl.col('industry').n_unique()).item()}")
        print(f"\nIndustries:")
        industry_counts = panel_df.group_by('industry').agg(
            pl.col('ticker').n_unique().alias('n_assets')
        ).sort('industry')
        print(industry_counts)
        
        return panel_df
    
    def create_wide_panel_data(self, industry_mapping_path='industry_mapping.csv'):
        """
        Create wide-format panel data where each asset's features become separate columns
        
        Creates a wide DataFrame with:
        - timestamp: datetime (index)
        - For each asset: {ticker}_ask-price, {ticker}_ask-volume, {ticker}_bid-price, etc.
        - {ticker}_industry for each asset
        
        Args:
            industry_mapping_path: Path to CSV with ticker-industry mapping
            
        Returns:
            Polars DataFrame in wide format (timestamp as rows, assets as column groups)
        """
        # First create long format
        panel_long = self.create_panel_data(industry_mapping_path)
        
        if panel_long is None:
            return None
        
        print(f"\n{'='*70}")
        print("CONVERTING TO WIDE FORMAT")
        print(f"{'='*70}\n")
        
        # Pivot to wide format for each feature
        features = ['ask-price', 'ask-volume', 'bid-price', 'bid-volume', 
                   'spread', 'mid-price', 'volume_imbalance']
        
        # Get unique timestamps
        timestamps = panel_long.select('timestamp').unique().sort('timestamp')
        wide_df = timestamps
        
        # For each feature, pivot and join
        for feature in features:
            print(f"Pivoting {feature}...")
            pivot = panel_long.pivot(
                values=feature,
                index='timestamp',
                columns='ticker'
            )
            
            # Rename columns to {ticker}_{feature}
            pivot = pivot.rename({
                col: f"{col}_{feature}" if col != 'timestamp' else col
                for col in pivot.columns
            })
            
            # Join with main DataFrame
            wide_df = wide_df.join(pivot, on='timestamp', how='left')
        
        # Add industry columns (one per ticker)
        print("Adding industry columns...")
        industry_df = panel_long.select(['ticker', 'industry']).unique()
        
        for row in industry_df.iter_rows(named=True):
            ticker = row['ticker']
            industry = row['industry']
            wide_df = wide_df.with_columns(
                pl.lit(industry).alias(f"{ticker}_industry")
            )
        
        print(f"\n{'='*70}")
        print("WIDE PANEL DATA CREATED")
        print(f"{'='*70}")
        print(f"Shape: {wide_df.shape}")
        print(f"Date range: {wide_df.select(pl.col('timestamp').min()).item()} to {wide_df.select(pl.col('timestamp').max()).item()}")
        
        return wide_df
    
    def compare_resampling_intervals(self, industry_mapping_path='industry_mapping.csv',
                                    intervals=['1min', '5min', '15min', '30min']):
        """
        Compare different resampling intervals to see which produces the least NaN values
        
        This function creates panel data with different time aggregations and reports:
        - Total rows in the panel
        - Number of NaN values per column
        - Percentage of data retention
        - Memory usage
        
        Args:
            industry_mapping_path: Path to CSV with ticker-industry mapping
            intervals: List of time intervals to test (e.g., ['1min', '5min', '15min', '30min'])
            
        Returns:
            Dictionary with comparison statistics for each interval
        """
        print(f"{'='*70}")
        print(f"COMPARING RESAMPLING INTERVALS: {', '.join(intervals)}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for interval in intervals:
            print(f"\n{'='*70}")
            print(f"Testing interval: {interval}")
            print(f"{'='*70}")
            
            try:
                # Create panel data with this interval
                panel = self.create_panel_data(
                    industry_mapping_path=industry_mapping_path,
                    resample_interval=interval
                )
                
                if panel is None:
                    print(f"Failed to create panel for {interval}")
                    continue
                
                # Calculate statistics
                total_rows = panel.height
                total_cells = total_rows * len(panel.columns)
                
                # Count nulls per column
                null_counts = {}
                for col in panel.columns:
                    if col not in ['timestamp', 'ticker', 'industry']:
                        n_nulls = panel[col].null_count()
                        null_counts[col] = n_nulls
                
                total_nulls = sum(null_counts.values())
                null_percentage = (total_nulls / total_cells) * 100
                
                # Get unique timestamps and tickers
                n_timestamps = panel.select(pl.col('timestamp').n_unique()).item()
                n_tickers = panel.select(pl.col('ticker').n_unique()).item()
                
                # Estimate memory (approximate)
                memory_mb = panel.estimated_size('mb')
                
                # Store results
                results[interval] = {
                    'total_rows': total_rows,
                    'n_timestamps': n_timestamps,
                    'n_tickers': n_tickers,
                    'total_cells': total_cells,
                    'total_nulls': total_nulls,
                    'null_percentage': null_percentage,
                    'memory_mb': memory_mb,
                    'null_counts_by_column': null_counts
                }
                
                # Print summary
                print(f"\nResults for {interval}:")
                print(f"  Total rows: {total_rows:,}")
                print(f"  Unique timestamps: {n_timestamps:,}")
                print(f"  Unique tickers: {n_tickers:,}")
                print(f"  Total NaN values: {total_nulls:,} ({null_percentage:.2f}%)")
                print(f"  Memory usage: {memory_mb:.2f} MB")
                print(f"\n  NaN counts by column:")
                for col, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / total_rows) * 100
                    print(f"    {col}: {count:,} ({pct:.2f}%)")
                
                # Free memory
                del panel
                
            except Exception as e:
                print(f"\nError testing interval {interval}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print comparison summary
        print(f"\n\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Interval':<10} {'Rows':<15} {'Timestamps':<12} {'NaN %':<10} {'Memory (MB)':<12}")
        print(f"{'-'*70}")
        
        for interval in intervals:
            if interval in results:
                r = results[interval]
                print(f"{interval:<10} {r['total_rows']:<15,} {r['n_timestamps']:<12,} "
                      f"{r['null_percentage']:<10.2f} {r['memory_mb']:<12.2f}")
        
        # Recommend best interval (least NaN percentage)
        if results:
            best_interval = min(results.items(), key=lambda x: x[1]['null_percentage'])
            print(f"\nRecommendation: {best_interval[0]} has the lowest NaN percentage ({best_interval[1]['null_percentage']:.2f}%)")
        
        return results
