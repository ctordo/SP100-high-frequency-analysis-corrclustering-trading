import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.clustering_utils.Leiden_clustering import LeidenCorrelationClustering
from utils.clustering_utils.Louvain_clustering import LouvainCorrelationClustering
from utils.clustering_utils.Marsili_Giada_clustering import MarsiliGiadaCorrelationClustering
from utils.clustering_utils.Utils import cluster_industry, industry_mapping
from utils.clustering_utils.plots import (
    plot_all_clustering_methods,
    plot_all_cluster_correlation_graphs,
    plot_industry_distribution_by_cluster,
    plot_cluster_correlation_graph
)

data_local = "../FBD_local_data/"

def main_3(industry_mapping_dict=None, create_plots=True, show_individual_plots=False):
    """
    Run complete clustering pipeline on stock returns data.
    
    This function performs four different clustering methods:
    1. Leiden Clustering - Community detection based on correlation matrix
    2. Louvain Clustering - Modularity-based community detection
    3. Marsili-Giada Clustering - Maximum likelihood hierarchical clustering
    4. Industry Clustering - Clustering based on industry classification
    
    Each method uses its own built-in correlation matrix cleaning approach.
    
    Parameters:
    -----------
    df_returns : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    industry_mapping_dict : dict, optional
        Dictionary mapping ticker symbols to industry names. 
        If None, uses the default industry_mapping from Utils.
    create_plots : bool, default=True
        Whether to create and display plots
    show_individual_plots : bool, default=False
        Whether to show individual plots for each clustering method (only if create_plots=True)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - Index: Asset ticker symbols
        - 'Leiden_Cluster': Cluster assignments from Leiden method
        - 'Louvain_Cluster': Cluster assignments from Louvain method
        - 'Marsili_Giada_Cluster': Cluster assignments from Marsili-Giada method
        - 'Industry_Cluster': Cluster assignments based on industry
    """
    
    # Initial message
    print("="*70)
    print("ENTERING PHASE 3: CLUSTERING PIPELINE")
    print("="*70)

    # ============ LOAD AND REFORMAT PHASE INPUTS ============ 
    print("="*50 + "\n3.1 Loading stock returns...")
    
    # Stock returns
    print("  Loading + structuring 'stock_returns.csv'...")
    df_returns = pd.read_csv(data_local + "stock_returns.csv")
    df_returns.set_index("timestamp", inplace=True, drop=True)

    print(f"  Input data shape: {df_returns.shape}")
    print(f"  Number of stocks: {df_returns.shape[1]}")
    print(f"  Number of time periods: {df_returns.shape[0]}")
    
    # Use provided industry mapping or default
    if industry_mapping_dict is None:
        industry_mapping_dict = industry_mapping
    
    # Get tickers
    TICKERS = list(df_returns.columns)
    
    # ============ RUN CLUSTERING METHODS ============ 
    print("="*50 + "\n3.2 Running clustering methods")
    print("  Each method uses its own built-in correlation matrix cleaning.")
    
    # 1. Leiden Clustering
    print("\n  1. Running Leiden Clustering...")
    leiden_df = LeidenCorrelationClustering(df_returns)
    leiden_clusters = leiden_df['cluster'].values
    print(f"   Leiden: {leiden_df['cluster'].nunique()} clusters identified")
    
    # 2. Louvain Clustering
    print("\n  2. Running Louvain Clustering...")
    louvain_df = LouvainCorrelationClustering(df_returns)
    louvain_clusters = louvain_df.iloc[:, 0].values
    print(f"   Louvain: {louvain_df.iloc[:, 0].nunique()} clusters identified")
    
    # 3. Marsili-Giada Clustering
    print("\n  3. Running Marsili-Giada Clustering...")
    marsili_df = MarsiliGiadaCorrelationClustering(df_returns)
    marsili_clusters = marsili_df['cluster'].values
    print(f"   Marsili-Giada: {marsili_df['cluster'].nunique()} clusters identified")
    
    # 4. Industry-based Clustering
    print("\n  4. Running Industry-based Clustering...")
    industry_clusters_series = cluster_industry(TICKERS)
    industry_clusters = industry_clusters_series.values
    print(f"   Industry: {industry_clusters_series.nunique()} clusters identified")
    
    # ============ CREATE OUTPUT DATAFRAME ============ 
    print("="*50 + "\n3.3 Create output DataFrame")
    
    clustering_results = pd.DataFrame({
        'Leiden_Cluster': leiden_clusters,
        'Louvain_Cluster': louvain_clusters,
        'Marsili_Giada_Cluster': marsili_clusters,
        'Industry_Cluster': industry_clusters
    }, index=TICKERS)
    
    print("\nClustering Results Summary:")
    print(clustering_results.head(10))
    
    # Print summary statistics
    print("---"*10 + "CLUSTERING STATISTICS" + "---"*10)
    
    for method in ['Leiden_Cluster', 'Louvain_Cluster', 'Marsili_Giada_Cluster', 'Industry_Cluster']:
        n_clusters = clustering_results[method].nunique()
        print(f"\n{method}:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster sizes:")
        cluster_sizes = clustering_results[method].value_counts().sort_index()
        for cluster_id, size in cluster_sizes.items():
            print(f"    Cluster {cluster_id}: {size} stocks")
    
    # ============ CREATE PLOTS ============ 
    print("="*50 + "\n3.4 Create plots")
    if create_plots:
        print("---"*10 + "VISUALIZATIONS" + "---"*10)
        
        # Plot 1: Industry distribution across all clustering methods
        print("  1. Creating industry distribution plots...")
        fig_industry_dist = plot_all_clustering_methods(clustering_results)
        plt.show()
        
        # Plot 2: Correlation graphs for all clustering methods
        print("  2. Creating cluster correlation graphs...")
        fig_corr = plot_all_cluster_correlation_graphs(df_returns, clustering_results)
        plt.show()
        
        # Individual plots (optional)
        if show_individual_plots:
            print("  3. Creating individual plots...")
            
            # Individual industry distribution plots
            for method in ['Leiden_Cluster', 'Louvain_Cluster', 'Marsili_Giada_Cluster', 'Industry_Cluster']:
                print(f"\n   - {method} industry distribution...")
                fig, ax = plot_industry_distribution_by_cluster(clustering_results, cluster_column=method)
                plt.show()
            
            # Individual correlation graphs
            for method in ['Leiden_Cluster', 'Louvain_Cluster', 'Marsili_Giada_Cluster', 'Industry_Cluster']:
                print(f"\n   - {method} correlation graph...")
                fig, ax = plot_cluster_correlation_graph(df_returns, clustering_results, cluster_column=method)
                plt.show()
        
        print("\nAll visualizations complete!")
    else:
        print("  Argument 'create_plots' set to False.")
    
    # Final message
    print("\n" + "="*70)
    print("PHASE 3 CORRECTLY TERMINATED")
    print("="*70 + "\n")
    
    return clustering_results


def main_marsili_clustering(df_returns):
    """
    Apply Marsili-Giada clustering.
    
    This is a simplified function that performs only Marsili-Giada clustering
    using its built-in correlation matrix cleaning method.
    
    Parameters:
    -----------
    df_returns : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - Index: Asset ticker symbols
        - 'Marsili_Giada_Cluster': Cluster assignments from Marsili-Giada method
    
    """
    
    print("="*80)
    print("MARSILI-GIADA CLUSTERING")
    print("="*80)
    print(f"\nInput data shape: {df_returns.shape}")
    print(f"Number of stocks: {df_returns.shape[1]}")
    print(f"Number of time periods: {df_returns.shape[0]}")
    
    # Get tickers
    tickers = df_returns.columns
    
    # -----------------
    # Run Marsili-Giada Clustering
    # -----------------
    print("\n" + "-"*80)
    print("RUNNING MARSILI-GIADA CLUSTERING")
    print("-"*80)
    
    print("\nRunning Marsili-Giada Clustering...")
    print("(Using built-in correlation matrix cleaning method)")
    marsili_df = MarsiliGiadaCorrelationClustering(df_returns)
    marsili_clusters = marsili_df['cluster'].values
    
    n_clusters = marsili_df['cluster'].nunique()
    print(f"\nMarsili-Giada: {n_clusters} clusters identified")
    
    # -----------------
    # Create Output DataFrame
    # -----------------
    print("\n" + "-"*80)
    print("CREATING OUTPUT DATAFRAME")
    print("-"*80)
    
    results = pd.DataFrame({
        'Marsili_Giada_Cluster': marsili_clusters
    }, index=tickers)
    
    print("\nClustering Results Summary:")
    print(results.head(10))
    
    # Print cluster statistics
    print("\n" + "="*80)
    print("CLUSTERING SUMMARY STATISTICS")
    print("="*80)
    print(f"\nNumber of clusters: {n_clusters}")
    print(f"\nCluster sizes:")
    cluster_sizes = results['Marsili_Giada_Cluster'].value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        print(f"  Cluster {cluster_id}: {size} stocks")
    
    print("\n" + "="*80)
    print("MARSILI-GIADA CLUSTERING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return results


def run_marsili(df_returns):
    """
    Run Marsili-Giada clustering without any console output.
    
    This is a silent version of main_marsili_clustering() that returns results
    without printing any status messages or statistics.
    
    Parameters:
    -----------
    df_returns : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - Index: Asset ticker symbols
        - 'Marsili_Giada_Cluster': Cluster assignments from Marsili-Giada method
    
    """
    print('Starting Marsili-Giada clustering...')
    # Get tickers
    tickers = df_returns.columns
    
    # Run Marsili-Giada Clustering
    marsili_df = MarsiliGiadaCorrelationClustering(df_returns)
    marsili_clusters = marsili_df['cluster'].values
    
    # Create Output DataFrame
    results = pd.DataFrame({
        'Marsili_Giada_Cluster': marsili_clusters
    }, index=tickers)
    print('Marsili-Giada clustering completed.')
    
    return results


def run_louvain(df_returns):
    """
    Run Louvain clustering with minimal console output.
    
    This function runs Louvain clustering and returns results with only
    a message at the beginning and end.
    
    Parameters:
    -----------
    df_returns : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - Index: Asset ticker symbols
        - 'Louvain_Cluster': Cluster assignments from Louvain method
    
    """
    print('  Starting Louvain clustering...')
    
    # Get tickers
    tickers = df_returns.columns
    
    # Run Louvain Clustering
    louvain_df = LouvainCorrelationClustering(df_returns)
    louvain_clusters = louvain_df[0].values
    
    # Create Output DataFrame
    results = pd.DataFrame({
        'asset': tickers,
        'cluster': louvain_clusters
    })
    
    return results


