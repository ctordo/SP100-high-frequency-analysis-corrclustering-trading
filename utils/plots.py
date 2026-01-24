import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from utils.Utils import industry_mapping


def plot_industry_distribution_by_cluster(clustering_df, cluster_column='Leiden_Cluster'):
    """
    Creates a stacked bar chart showing the proportion of each industry within each cluster.
    
    Parameters:
    -----------
    clustering_df : pd.DataFrame
        DataFrame containing clustering results with columns for clusters and an index of ticker symbols
    cluster_column : str
        Name of the column containing cluster assignments (e.g., 'Leiden_Cluster', 'Louvain_Cluster')
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Map tickers to industries
    industries = clustering_df.index.map(lambda ticker: industry_mapping.get(ticker, 'Other'))
    
    # Create a temporary dataframe with cluster and industry
    temp_df = pd.DataFrame({
        'Cluster': clustering_df[cluster_column],
        'Industry': industries
    })
    
    # Calculate counts for each industry in each cluster
    industry_cluster_counts = temp_df.groupby(['Cluster', 'Industry']).size().unstack(fill_value=0)
    
    # Calculate proportions (percentage)
    industry_cluster_proportions = industry_cluster_counts.div(industry_cluster_counts.sum(axis=1), axis=0) * 100
    
    # Sort clusters
    industry_cluster_proportions = industry_cluster_proportions.sort_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each industry (using a colormap)
    industries_list = industry_cluster_proportions.columns.tolist()
    colors = plt.cm.tab20(np.linspace(0, 1, len(industries_list)))
    
    # Create stacked bar chart
    x_positions = np.arange(len(industry_cluster_proportions.index))
    bottom = np.zeros(len(industry_cluster_proportions.index))
    
    bars = []
    for idx, industry in enumerate(industries_list):
        values = industry_cluster_proportions[industry].values
        bar = ax.bar(x_positions, values, bottom=bottom, label=industry, 
                     color=colors[idx], edgecolor='white', linewidth=0.5)
        bars.append(bar)
        bottom += values
    
    # Customize the plot
    ax.set_xlabel('Cluster', fontsize=14, fontweight='bold')
    ax.set_ylabel('Proportion (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Industry Distribution by {cluster_column.replace("_", " ")}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'Cluster {i+1}' for i in industry_cluster_proportions.index], 
                       fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=12)
    
    # Add legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
             frameon=True, title='Industry', title_fontsize=12)
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig, ax


def plot_all_clustering_methods(clustering_df):
    """
    Creates subplots showing industry distribution for all clustering methods.
    
    Parameters:
    -----------
    clustering_df : pd.DataFrame
        DataFrame containing clustering results with columns for different clustering methods
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    # Determine which clustering methods are available
    available_methods = []
    possible_methods = ['Leiden_Cluster', 'Louvain_Cluster', 'Marsili_Giada_Cluster', 'Industry_Cluster']
    for method in possible_methods:
        if method in clustering_df.columns:
            available_methods.append(method)
    
    n_methods = len(available_methods)
    fig = plt.figure(figsize=(6 * n_methods, 6))
    
    cluster_methods = available_methods
    
    for idx, cluster_method in enumerate(cluster_methods, 1):
        ax = fig.add_subplot(1, n_methods, idx)
        
        # Map tickers to industries
        industries = clustering_df.index.map(lambda ticker: industry_mapping.get(ticker, 'Other'))
        
        # Create a temporary dataframe with cluster and industry
        temp_df = pd.DataFrame({
            'Cluster': clustering_df[cluster_method],
            'Industry': industries
        })
        
        # Calculate counts for each industry in each cluster
        industry_cluster_counts = temp_df.groupby(['Cluster', 'Industry']).size().unstack(fill_value=0)
        
        # Calculate proportions (percentage)
        industry_cluster_proportions = industry_cluster_counts.div(industry_cluster_counts.sum(axis=1), axis=0) * 100
        
        # Sort clusters
        industry_cluster_proportions = industry_cluster_proportions.sort_index()
        
        # Define colors for each industry
        industries_list = industry_cluster_proportions.columns.tolist()
        colors = plt.cm.tab20(np.linspace(0, 1, len(industries_list)))
        
        # Create stacked bar chart
        x_positions = np.arange(len(industry_cluster_proportions.index))
        bottom = np.zeros(len(industry_cluster_proportions.index))
        
        for i, industry in enumerate(industries_list):
            values = industry_cluster_proportions[industry].values
            ax.bar(x_positions, values, bottom=bottom, label=industry, 
                  color=colors[i], edgecolor='white', linewidth=0.5)
            bottom += values
        
        # Customize the subplot
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
        if idx == 1:
            ax.set_ylabel('Proportion (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{cluster_method.replace("_", " ")}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'C{i+1}' for i in industry_cluster_proportions.index], 
                          fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Add a single legend for all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, frameon=True, title='Industry', title_fontsize=12)
    
    plt.suptitle('Industry Distribution Across Different Clustering Methods', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_cluster_correlation_graph(returns_df, clustering_df, cluster_column='Louvain_Cluster', 
                                   min_edge_weight=5.0, figsize=(12, 10)):
    """
    Creates a network graph showing correlation strength within and between clusters.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame containing stock returns with dates as index and ticker symbols as columns
    clustering_df : pd.DataFrame
        DataFrame containing clustering results with cluster assignments
    cluster_column : str
        Name of the column containing cluster assignments (e.g., 'Leiden_Cluster', 'Louvain_Cluster')
    min_edge_weight : float
        Minimum edge weight to display (filters weak connections)
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Get cluster assignments
    clusters = clustering_df[cluster_column].values
    tickers = clustering_df.index.tolist()
    unique_clusters = sorted(clustering_df[cluster_column].unique())
    
    # Compute correlation matrix
    corr_matrix = returns_df.corr()
    
    # Calculate average correlation within and between clusters
    cluster_corr = {}
    cluster_sizes = {}
    
    for i in unique_clusters:
        cluster_sizes[i] = np.sum(clusters == i)
        
        # Within-cluster correlation
        mask_i = clusters == i
        within_corr = corr_matrix.values[np.ix_(mask_i, mask_i)]
        # Exclude diagonal (self-correlation)
        np.fill_diagonal(within_corr, np.nan)
        avg_within = np.nanmean(np.abs(within_corr))
        cluster_corr[(i, i)] = avg_within
        
        # Between-cluster correlation
        for j in unique_clusters:
            if j > i:  # Only compute once for each pair
                mask_j = clusters == j
                between_corr = corr_matrix.values[np.ix_(mask_i, mask_j)]
                avg_between = np.nanmean(np.abs(between_corr))
                cluster_corr[(i, j)] = avg_between
                cluster_corr[(j, i)] = avg_between  # Symmetric
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (clusters)
    for cluster_id in unique_clusters:
        G.add_node(cluster_id, size=cluster_sizes[cluster_id])
    
    # Add edges (correlations between clusters)
    for (i, j), weight in cluster_corr.items():
        if i <= j and weight >= min_edge_weight / 100:  # Convert to correlation scale
            G.add_edge(i, j, weight=weight * 100)  # Store as percentage
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position nodes in a circular layout
    pos = nx.circular_layout(G)
    
    # Define node colors - generate colors dynamically for any number of clusters
    num_clusters = len(unique_clusters)
    if num_clusters <= 5:
        # Use specific colors for small number of clusters
        color_list = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#1f77b4', '#1f77b4']
        node_colors = color_list[:num_clusters]
    else:
        # Use colormap for larger number of clusters
        node_colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
    
    # Draw nodes with size proportional to cluster size
    node_sizes = [cluster_sizes[node] * 200 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          ax=ax, alpha=0.9, edgecolors='black', linewidths=2)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_weight='bold', 
                           font_color='white', ax=ax)
    
    # Draw edges with width proportional to correlation strength
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize edge widths
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 0
    edge_widths = [2 + 6 * (w - min_weight) / (max_weight - min_weight + 0.001) for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax, edge_color='gray')
    
    # Draw edge labels (correlation values)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, 
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                          alpha=0.8, edgecolor='none'), ax=ax)
    
    # Customize plot
    ax.set_title(f'{cluster_column.replace("_", " ")} - Correlation Graph Between Clusters', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend explaining the visualization
    legend_text = (
        "Node size: Number of stocks in cluster\n"
        "Node labels: Cluster ID\n"
        "Edge width: Correlation strength\n"
        "Edge labels: Average correlation (%)"
    )
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, ax


def plot_all_cluster_correlation_graphs(returns_df, clustering_df):
    """
    Creates correlation graphs for all clustering methods.
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame containing stock returns
    clustering_df : pd.DataFrame
        DataFrame containing clustering results
    
    Returns:
    --------
    fig : matplotlib figure object
    """
    # Determine which clustering methods are available
    available_methods = []
    possible_methods = ['Louvain_Cluster', 'Leiden_Cluster', 'Marsili_Giada_Cluster', 'Industry_Cluster']
    for method in possible_methods:
        if method in clustering_df.columns:
            available_methods.append(method)
    
    n_methods = len(available_methods)
    fig = plt.figure(figsize=(6 * n_methods, 6))
    
    cluster_methods = available_methods
    
    for idx, cluster_method in enumerate(cluster_methods, 1):
        ax = fig.add_subplot(1, n_methods, idx)
        
        # Get cluster assignments
        clusters = clustering_df[cluster_method].values
        unique_clusters = sorted(clustering_df[cluster_method].unique())
        
        # Compute correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate average correlation within and between clusters
        cluster_corr = {}
        cluster_sizes = {}
        
        for i in unique_clusters:
            cluster_sizes[i] = np.sum(clusters == i)
            
            # Within-cluster correlation
            mask_i = clusters == i
            within_corr = corr_matrix.values[np.ix_(mask_i, mask_i)]
            np.fill_diagonal(within_corr, np.nan)
            avg_within = np.nanmean(np.abs(within_corr))
            cluster_corr[(i, i)] = avg_within
            
            # Between-cluster correlation
            for j in unique_clusters:
                if j > i:
                    mask_j = clusters == j
                    between_corr = corr_matrix.values[np.ix_(mask_i, mask_j)]
                    avg_between = np.nanmean(np.abs(between_corr))
                    cluster_corr[(i, j)] = avg_between
                    cluster_corr[(j, i)] = avg_between
        
        # Create network graph
        G = nx.Graph()
        
        for cluster_id in unique_clusters:
            G.add_node(cluster_id, size=cluster_sizes[cluster_id])
        
        for (i, j), weight in cluster_corr.items():
            if i <= j:
                G.add_edge(i, j, weight=weight * 100)
        
        # Position and draw
        pos = nx.circular_layout(G)
        
        # Node colors - generate dynamically
        num_clusters_method = len(unique_clusters)
        if num_clusters_method <= 5:
            color_list = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#1f77b4', '#1f77b4']
            node_colors = [color_list[i] if i < len(color_list) else '#8c564b' for i in range(num_clusters_method)]
        else:
            node_colors = plt.cm.tab20(np.linspace(0, 1, num_clusters_method))
        
        node_sizes = [cluster_sizes[node] * 150 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              ax=ax, alpha=0.9, edgecolors='black', linewidths=1.5)
        
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', 
                               font_color='white', ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        if weights:
            max_weight = max(weights)
            min_weight = min(weights)
            edge_widths = [1 + 4 * (w - min_weight) / (max_weight - min_weight + 0.001) for w in weights]
        else:
            edge_widths = [1] * len(edges)
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, ax=ax, edge_color='gray')
        
        edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title(f'{cluster_method.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Correlation Graphs Between Clusters', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig
