"""
Script to analyze factor loadings from denoised covariance matrix

This demonstrates how to:
1. Load saved factor loadings
2. Identify ticker clusters based on factor exposures
3. Find pairs with similar/opposite factor exposures
4. Visualize factor structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_factor_analysis():
    """Load saved factor loadings and covariance matrices"""
    loadings = pd.read_csv('factor_loadings.csv', index_col=0)
    cov_raw = np.load('covariance_raw.npy')
    cov_denoised = np.load('covariance_denoised.npy')
    
    return loadings, cov_raw, cov_denoised

def cluster_tickers_by_factor(loadings_df: pd.DataFrame, factor_name: str, n_groups: int = 3):
    """
    Cluster tickers into groups based on loading on a specific factor
    
    Args:
        loadings_df: DataFrame with factor loadings
        factor_name: Which factor to use for clustering (e.g., 'Factor_1')
        n_groups: Number of groups (e.g., 3 = high/medium/low)
    
    Returns:
        Dictionary of {group_name: [list of tickers]}
    """
    factor_loadings = loadings_df[factor_name].sort_values(ascending=False)
    
    # Split into quantiles
    groups = {}
    quantiles = np.linspace(0, len(factor_loadings), n_groups + 1, dtype=int)
    
    for i in range(n_groups):
        start, end = quantiles[i], quantiles[i+1]
        group_tickers = factor_loadings.iloc[start:end].index.tolist()
        groups[f'Group_{i+1}_{"high" if i==0 else "low" if i==n_groups-1 else "mid"}'] = group_tickers
    
    return groups

def find_opposite_loading_pairs(loadings_df: pd.DataFrame, factor_name: str, n_pairs: int = 10):
    """
    Find pairs of stocks with opposite loadings on a factor (for market-neutral pairs)
    
    Args:
        loadings_df: DataFrame with factor loadings
        factor_name: Which factor to analyze
        n_pairs: Number of pairs to return
    
    Returns:
        List of (ticker_positive, ticker_negative, loading_diff) tuples
    """
    factor_loadings = loadings_df[factor_name].sort_values()
    
    # Get most negative and most positive
    n_extreme = min(20, len(factor_loadings) // 2)
    negative = factor_loadings.head(n_extreme)
    positive = factor_loadings.tail(n_extreme)
    
    pairs = []
    for neg_ticker, neg_load in negative.items():
        for pos_ticker, pos_load in positive.items():
            diff = abs(pos_load - neg_load)
            pairs.append((pos_ticker, neg_ticker, diff))
    
    # Sort by largest difference and return top pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:n_pairs]

def find_similar_loading_pairs(loadings_df: pd.DataFrame, n_pairs: int = 20, min_correlation: float = 0.7):
    """
    Find pairs of stocks with similar factor exposures (for cointegration candidates)
    
    Uses Euclidean distance in factor space to find similar stocks
    
    Args:
        loadings_df: DataFrame with factor loadings
        n_pairs: Number of pairs to return
        min_correlation: Minimum correlation in factor space
    
    Returns:
        List of (ticker_A, ticker_B, similarity_score) tuples
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Compute pairwise distances in factor space
    distances = pdist(loadings_df.values, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # Convert to similarity (inverse of distance)
    # Add small epsilon to avoid division by zero
    similarity_matrix = 1 / (dist_matrix + 1e-10)
    
    # Zero out diagonal
    np.fill_diagonal(similarity_matrix, 0)
    
    # Find top pairs
    pairs = []
    tickers = loadings_df.index.tolist()
    
    for i in range(len(tickers)):
        for j in range(i+1, len(tickers)):
            similarity = similarity_matrix[i, j]
            
            # Also check actual correlation of loadings
            corr = np.corrcoef(loadings_df.iloc[i], loadings_df.iloc[j])[0, 1]
            
            if corr >= min_correlation:
                pairs.append((tickers[i], tickers[j], similarity, corr))
    
    # Sort by similarity and return top pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(t1, t2, sim, corr) for t1, t2, sim, corr in pairs[:n_pairs]]

def visualize_factor_heatmap(loadings_df: pd.DataFrame, save_path: str = 'factor_heatmap.png'):
    """
    Visualize factor loading heatmap
    
    Args:
        loadings_df: DataFrame with factor loadings
        save_path: Where to save the plot
    """
    plt.figure(figsize=(12, 16))
    
    # Sort tickers by their loading on Factor_1 for better visualization
    sorted_loadings = loadings_df.sort_values('Factor_1', ascending=False)
    
    sns.heatmap(sorted_loadings, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Factor Loading'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Factor Loadings Heatmap\n(Sorted by Factor_1)', fontsize=14, pad=20)
    plt.xlabel('Factors', fontsize=12)
    plt.ylabel('Tickers', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {save_path}")
    plt.show()

def analyze_factor_correlations(loadings_df: pd.DataFrame):
    """
    Check if factors are orthogonal (they should be from eigendecomposition)
    
    Args:
        loadings_df: DataFrame with factor loadings
    """
    factor_corr = loadings_df.corr()
    
    print("\nFactor Correlation Matrix:")
    print("(Should be near-identity since eigenvectors are orthogonal)")
    print(factor_corr.round(3))
    
    # Check orthogonality
    off_diagonal = factor_corr.values[~np.eye(factor_corr.shape[0], dtype=bool)]
    max_off_diag = np.abs(off_diagonal).max()
    print(f"\nMax off-diagonal correlation: {max_off_diag:.6f}")
    print("✓ Factors are orthogonal" if max_off_diag < 0.01 else "⚠️  Warning: factors may not be perfectly orthogonal")

if __name__ == '__main__':
    print("="*80)
    print("FACTOR LOADING ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading factor loadings and covariance matrices...")
    loadings, cov_raw, cov_denoised = load_factor_analysis()
    print(f"Loaded loadings for {len(loadings)} tickers across {loadings.shape[1]} factors")
    
    # Analyze factor orthogonality
    analyze_factor_correlations(loadings)
    
    # Example 1: Cluster by Factor_1 (likely market factor)
    print("\n" + "="*80)
    print("EXAMPLE 1: Clustering by Factor_1 (Market Factor)")
    print("="*80)
    clusters = cluster_tickers_by_factor(loadings, 'Factor_1', n_groups=3)
    for group_name, tickers in clusters.items():
        print(f"\n{group_name}: {len(tickers)} stocks")
        print(f"  {tickers[:5]}...")  # Show first 5
    
    # Example 2: Find market-neutral pairs (opposite loadings on Factor_1)
    print("\n" + "="*80)
    print("EXAMPLE 2: Market-Neutral Pair Candidates")
    print("="*80)
    print("(Stocks with opposite loadings on Factor_1 for hedging)")
    opposite_pairs = find_opposite_loading_pairs(loadings, 'Factor_1', n_pairs=10)
    print(f"\n{'Rank':<6} {'Long':<10} {'Short':<10} {'Loading Diff':<15}")
    print("-"*50)
    for rank, (pos, neg, diff) in enumerate(opposite_pairs, 1):
        print(f"{rank:<6} {pos:<10} {neg:<10} {diff:>14.4f}")
    
    # Example 3: Find similar factor exposure pairs (cointegration candidates)
    print("\n" + "="*80)
    print("EXAMPLE 3: Similar Factor Exposure Pairs")
    print("="*80)
    print("(Stocks with similar factor loadings - good cointegration candidates)")
    similar_pairs = find_similar_loading_pairs(loadings, n_pairs=20, min_correlation=0.7)
    print(f"\n{'Rank':<6} {'Ticker A':<10} {'Ticker B':<10} {'Similarity':<12} {'Corr':<8}")
    print("-"*60)
    for rank, (t1, t2, sim, corr) in enumerate(similar_pairs, 1):
        print(f"{rank:<6} {t1:<10} {t2:<10} {sim:>11.4f} {corr:>7.3f}")
    
    # Example 4: Visualize factor loadings
    print("\n" + "="*80)
    print("EXAMPLE 4: Visualizing Factor Structure")
    print("="*80)
    visualize_factor_heatmap(loadings)
    
    print("\n✓ Analysis complete!")
    print("\nNext steps:")
    print("  1. Use similar_pairs for cointegration testing")
    print("  2. Use opposite_pairs for market-neutral strategies")
    print("  3. Examine factor_heatmap.png to identify sector patterns")
