from numpy import linalg as LA
import numpy as np
import pandas as pd
from scipy import sparse
from sknetwork.clustering import Leiden

def compute_C_minus_C0(lambdas, v, lambda_plus):
    idx_max = np.argmax(lambdas)
    C_sector = np.zeros((len(lambdas), len(lambdas)), dtype=float)

    for i in range(len(lambdas)):
        if (lambdas[i] > lambda_plus) and (i != idx_max):
            C_sector += lambdas[i] * np.outer(v[:, i], v[:, i])

    C_sector = np.abs(C_sector)
    return C_sector

def LeidenCorrelationClustering(R):
    N, T = R.shape[1], R.shape[0]
    q = N / T
    lambda_plus = (1. + np.sqrt(q)) ** 2

    C = R.corr()
    lambdas, v = LA.eigh(C)

    C_s = compute_C_minus_C0(lambdas, v, lambda_plus)

    # remove self-loops
    np.fill_diagonal(C_s, 0.0)

    adjacency = sparse.csr_matrix(C_s)

    leiden = Leiden()
    partition = leiden.fit_predict(adjacency)

    DF = pd.DataFrame({"cluster": partition}, index=R.columns)
    return DF
