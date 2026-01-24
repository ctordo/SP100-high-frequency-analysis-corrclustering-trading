import pandas as pd


industry_mapping = {
    # Technology
    'IBM': 'Technology', 'HPQ': 'Technology', 'EMC': 'Technology',
    'TXN': 'Technology', 'XRX': 'Technology',
    
    # Healthcare/Pharma
    'ABT': 'Healthcare', 'BAX': 'Healthcare', 'BMY': 'Healthcare', 'JNJ': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'MDT': 'Healthcare', 'UNH': 'Healthcare',
    'CVS': 'Healthcare',
    
    # Financial
    'AXP': 'Financial', 'BAC': 'Financial', 'BK': 'Financial', 'C': 'Financial',
    'COF': 'Financial', 'GS': 'Financial', 'JPM': 'Financial', 'MS': 'Financial',
    'USB': 'Financial', 'WFC': 'Financial', 'MA': 'Financial', 'V': 'Financial',
    'MET': 'Financial',
    
    # Energy
    'APA': 'Energy', 'BHI': 'Energy', 'COP': 'Energy', 'CVX': 'Energy', 'DVN': 'Energy',
    'FCX': 'Energy', 'HAL': 'Energy', 'NOV': 'Energy', 'OXY': 'Energy', 'SLB': 'Energy',
    'XOM': 'Energy', 'WMB': 'Energy',
    
    # Consumer Goods
    'PG': 'Consumer Goods', 'CL': 'Consumer Goods', 'KO': 'Consumer Goods',
    'PEP': 'Consumer Goods', 'PM': 'Consumer Goods', 'MO': 'Consumer Goods',
    'KFT': 'Consumer Goods', 'NKE': 'Consumer Goods',
    
    # Retail
    'WMT': 'Retail', 'TGT': 'Retail', 'HD': 'Retail', 'LOW': 'Retail',
    'WAG': 'Retail', 'S': 'Retail',
    
    # Industrial
    'CAT': 'Industrial', 'DD': 'Industrial', 'DOW': 'Industrial', 'EMR': 'Industrial',
    'GE': 'Industrial', 'HON': 'Industrial', 'MMM': 'Industrial', 'UTX': 'Industrial',
    'GD': 'Industrial', 'LMT': 'Industrial', 'RTN': 'Industrial', 'BA': 'Industrial',
    
    # Transportation/Logistics
    'FDX': 'Transportation', 'UPS': 'Transportation', 'UNP': 'Transportation',
    'NSC': 'Transportation',
    
    # Utilities
    'AEP': 'Utilities', 'ETR': 'Utilities', 'EXC': 'Utilities', 'SO': 'Utilities',
    
    # Telecom
    'T': 'Telecom', 'VZ': 'Telecom',
    
    # Media/Entertainment
    'DIS': 'Media', 'TWX': 'Media',
    
    # Food/Beverage
    'MCD': 'Food & Beverage', 'HNZ': 'Food & Beverage', 'MON': 'Food & Beverage',
    
    # Automotive
    'F': 'Automotive',
    
    # Materials
    'WY': 'Materials', 'AVP': 'Materials', 'ALL': 'Materials',
}


def cluster_industry(tickers):
    ticker_industry = pd.Series({ticker: industry_mapping.get(ticker, 'Other') for ticker in tickers})
    ticker_industry.name = 'Industry'
    unique_industries = sorted(ticker_industry.unique())
    industry_to_cluster = {industry: idx for idx, industry in enumerate(unique_industries)}
    industry_clusters = ticker_industry.map(industry_to_cluster)
    industry_clusters.name = 'Industry_Cluster'
    return industry_clusters


def create_output_df(louvain_clusters,leiden_clusters,industry_clusters,tickers):
    clustering_results_df = pd.DataFrame({
    'Louvain_Cluster': louvain_clusters,
    'Leiden_Cluster': leiden_clusters,
    'Industry_Cluster': industry_clusters
    }, index=tickers)  
    return clustering_results_df


def run_all_clustering_methods(returns_df, correlation_matrix=None):
    """
    Run all clustering methods (Leiden, Louvain, Marsili-Giada, and Industry-based) on the returns data.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    correlation_matrix : pd.DataFrame or np.ndarray, optional
        Pre-computed cleaned correlation matrix (e.g., after Marchenko-Pastur denoising).
        If None, each method will compute its own correlation matrix from returns_df.
    
    Returns:
    --------
    clustering_results_df : pd.DataFrame
        DataFrame with clustering results containing columns:
        - 'Louvain_Cluster': cluster assignments from Louvain method
        - 'Leiden_Cluster': cluster assignments from Leiden method
        - 'Marsili_Giada_Cluster': cluster assignments from Marsili-Giada method
        - 'Industry_Cluster': cluster assignments based on industry classification
    """
    from Leiden_clustering import LeidenCorrelationClustering
    from Louvain_clustering import LouvainCorrelationClustering
    from Marsili_Giada_clustering import MarsiliGiadaCorrelationClustering
    
    # Get tickers
    tickers = returns_df.columns
    
    # Run Leiden clustering
    leiden_df = LeidenCorrelationClustering(returns_df, correlation_matrix=correlation_matrix)
    leiden_clusters = leiden_df['cluster'].values
    
    # Run Louvain clustering
    louvain_df = LouvainCorrelationClustering(returns_df, correlation_matrix=correlation_matrix)
    louvain_clusters = louvain_df.iloc[:, 0].values
    
    # Run Marsili-Giada clustering
    marsili_df = MarsiliGiadaCorrelationClustering(returns_df, correlation_matrix=correlation_matrix)
    marsili_clusters = marsili_df['cluster'].values
    
    # Run Industry-based clustering
    industry_clusters_series = cluster_industry(tickers)
    industry_clusters = industry_clusters_series.values
    
    # Create output DataFrame
    clustering_results_df = pd.DataFrame({
        'Louvain_Cluster': louvain_clusters,
        'Leiden_Cluster': leiden_clusters,
        'Marsili_Giada_Cluster': marsili_clusters,
        'Industry_Cluster': industry_clusters
    }, index=tickers)
    
    return clustering_results_df

