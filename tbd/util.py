import numpy as np
import pandas as pd

def create_volatility_bars(df, vol_threshold, price_col = 'bid-price'): 
    """
        Create bars based on cumulative volatility instead of time
        
        Args:
            df: DataFrame with price data (datetime index)
            vol_threshold: Volatility threshold (1%)
            price_col: Column to use for volatility calculation
            
        Returns:
            DataFrame with volatility-based bars
    """
    # Calculate returns (log returns for better statistical properties)
    df = df.copy()
    df['price'] = df[price_col]
    df['returns'] = np.log(df['price'] / df['price'].shift(1))
    
    # Calculate absolute returns as volatility proxy
    df['abs_returns'] = df['returns'].abs()
    
    # Remove NaN values from the first row
    df = df.dropna(subset=['abs_returns'])
    
    # Calculate cumulative volatility
    df['cum_vol'] = df['abs_returns'].cumsum()
    
    # Create bar IDs based on volatility threshold
    # Each time cum_vol increases by vol_threshold, new bar starts
    df['bar_id'] = (df['cum_vol'] / vol_threshold).astype(int)
    
    # Group by bar_id and aggregate
    vol_bars = df.groupby('bar_id').agg({
        'bid-price': ['first', 'max', 'min', 'last', 'mean'],
        'ask-price': ['first', 'max', 'min', 'last', 'mean'],
        'bid-volume': 'sum',
        'ask-volume': 'sum',
        'abs_returns': 'sum',  # Total volatility in this bar
        'ticker': 'first',
        'cum_vol': 'last'
    })
    
    # Flatten column names
    vol_bars.columns = ['_'.join(col).strip() for col in vol_bars.columns.values]
    
    # Add timestamp (use last timestamp in each bar)
    vol_bars['timestamp'] = df.groupby('bar_id').apply(lambda x: x.index[-1])
    vol_bars = vol_bars.set_index('timestamp')
    
    # Add bar duration (time span of each bar)
    vol_bars['bar_duration'] = df.groupby('bar_id').apply(
        lambda x: (x.index[-1] - x.index[0]).total_seconds()
    ).values
    
    return vol_bars