from covariances import run_covariance_denoise
from covariances import CovarianceDenoiser

class PairsSingleWindow() : 
    def __init__(self,ret_window, cov_window, tickers, nb_pairs) : 
        self.ret_window = ret_window
        self.cov_window = cov_window
        self.tickers = tickers 
        self.nb_pairs = nb_pairs