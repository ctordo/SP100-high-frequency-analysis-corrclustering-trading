from numpy import linalg as LA
import numpy as np
import pandas as pd
import math
import networkx as nx
from community import community_louvain 


def compute_C_minus_C0(lambdas,v,lambda_plus):
    N=len(lambdas)
    C_clean=np.zeros((N, N))
    
    v_m=np.matrix(v)

    # largest eigenvalue index
    idx_max = np.argmax(lambdas)
    C_sector = np.zeros((N, N), dtype=float)

    for i in range(N):
        if (lambdas[i] > lambda_plus) and (i != idx_max):
            C_sector += lambdas[i] * np.outer(v[:, i], v[:, i])

    C_clean = np.abs(C_sector)

    return C_clean    
 
def LouvainCorrelationClustering(R):   # R is a matrix of return
    N=R.shape[1]
    T=R.shape[0]

    q=N*1./T
    lambda_plus=(1.+np.sqrt(q))**2

    C=R.corr()
    lambdas, v = LA.eigh(C)
    
    C_s=compute_C_minus_C0(lambdas,v,lambda_plus)
    C_s=np.abs(C_s)
    
    mygraph= nx.from_numpy_array(np.abs(C_s))
    partition = community_louvain.best_partition(mygraph)

    DF=pd.DataFrame.from_dict(partition,orient="index")
    DF=DF.set_index(R.columns)
    return DF


def format_clusters(clusters_df):
    """Return a Series: ticker -> cluster_id"""
    if isinstance(clusters_df, pd.DataFrame):
        if clusters_df.shape[1] != 1:
            raise ValueError("Expected clusters to have exactly 1 column.")
        labels = clusters_df.iloc[:, 0]
    else:
        labels = clusters_df
    labels = labels.astype(int)
    labels.index = labels.index.astype(str)
    labels.name = "cluster"
    return labels


def cluster_summary_table(labels):
    """Tidy table: ticker, cluster, cluster_size (sorted)."""
    summary = pd.DataFrame({"cluster": labels.values}, index=labels.index)
    return summary