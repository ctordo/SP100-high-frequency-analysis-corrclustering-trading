"""
Covariance Matrix Denoising using Marcenko-Pastur Algorithm

This module implements Random Matrix Theory (RMT) based denoising for correlation/covariance
matrices, following Laloux et al. (1999) and Bouchaud & Potters (2003).

Key features:
- Marcenko-Pastur theoretical distribution for eigenvalue filtering
- Constant Residual Eigenvalue (CRE) method for denoising
- EWMA online covariance updates for computational efficiency
- Integration with polars-based panel data

References:
- Laloux, L., et al. (1999). "Noise dressing of financial correlation matrices"
- Bouchaud, J.P. & Potters, M. (2003). "Theory of Financial Risk and Derivative Pricing"
- RiskMetrics (1996). "Technical Document"

Pipeline : 
- Load data
- Filter to common timestamp
- Calculate log returns 
- Select optimal window (for an optimal q)
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from pathlib import Path


class CovarianceDenoiser:
    """
    Covariance matrix denoiser using Random Matrix Theory
    """
    
    def __init__(self, input_path, decay_ewma: float = 0.94):
        """
        Initialize the covariance denoiser
        
        Args:
            decay_ewma: EWMA decay factor for updates (default: 0.94 from RiskMetrics)
        """
        self.decay_ewma = decay_ewma
        self.input_path = input_path
        self.cov_matrix = None
        self.denoised_cov = None
        self.eigenvalues = None
        self.n_signals = None
   

    @staticmethod
    def marcenko_pastur_pdf(var: float, q: float, pts: int = 1000) -> pd.Series:
        """
        Theoretical Marcenko-Pastur probability density function
        
        Args:
            var: Variance (typically set to 1 for correlation matrices)
            q: Ratio N/T where N=number of assets, T=time periods
            pts: Number of points for the PDF
            
        Returns:
            pd.Series with eigenvalue as index and PDF values
        """
        # Theoretical bounds for random eigenvalues
        # Using q = N/T convention, so sqrt(q) = sqrt(N/T)
        e_min = var * (1 - np.sqrt(q))**2
        e_max = var * (1 + np.sqrt(q))**2
        
        # Generate eigenvalue range
        e_val = np.linspace(e_min, e_max, pts)
        
        # MP density formula
        # Standard form: pdf(λ) = (1/(2πσ²q_standard)) * sqrt((e_max - λ)(λ - e_min)) / λ
        # where q_standard = T/N = 1/q (in our convention)
        # So we need to use 1/q instead of q in the formula
        pdf = (1 / (2 * np.pi * var * q * e_val)) * np.sqrt((e_max - e_val) * (e_val - e_min))
        pdf = np.nan_to_num(pdf)  # Handle numerical issues at boundaries  
        
        return pd.Series(pdf, index=e_val), e_max
    
    def fit_empirical_kde(self, eigenvalues: np.ndarray, bandwidth: float = 0.01) -> Tuple:
        """
        Fit KDE to empirical eigenvalue distribution
        
        Args:
            eigenvalues: Array of eigenvalues
            bandwidth: KDE bandwidth
            
        Returns:
            Tuple of (x_values, pdf_values)
        """
        from sklearn.neighbors import KernelDensity
        
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(eigenvalues.reshape(-1, 1))
        
        x_range = np.linspace(eigenvalues.min(), eigenvalues.max(), 1000)
        log_pdf = kde.score_samples(x_range.reshape(-1, 1))
        pdf = np.exp(log_pdf)
        
        return x_range, pdf
    
    def denoise_correlation(self, corr: np.ndarray, q: float, alpha: float = 0) -> Tuple[np.ndarray, Dict]:
        """
        Denoise a correlation matrix using Constant Residual Eigenvalue Method
        First decompose the correlation matrix into eigenvalues/eigenvectors
        Then identifies which eigenvalues are noise (i.e eigenvalues below threshold)
        Replace noise eigenvalues with their average
        Reconstruct the covariance matrix
        Args:
            corr: Correlation matrix (N x N)
            q: Ratio T/N (observations/variables)
            alpha: Shrinkage parameter (0 = no shrinkage)
            
        Returns:
            Tuple of (denoised_correlation_matrix, info_dict)
        """
        N = corr.shape[0]
        
        # 1. Eigen-decomposition
        e_val, e_vec = np.linalg.eigh(corr)
        
        # Sort in descending order
        indices = e_val.argsort()[::-1]
        e_val, e_vec = e_val[indices], e_vec[:, indices]
        
        # 2. Theoretical MP threshold
        var = 1 - alpha  # For correlation matrix, var ≈ 1
        _, e_max = self.marcenko_pastur_pdf(var, q)
        
        # 3. Count signal eigenvalues (above noise threshold)
        n_signals = np.sum(e_val > e_max)
        
        # 4. Constant Residual Eigenvalue (CRE): Replace noise with average
        e_val_denoised = np.copy(e_val)
        if n_signals < N:  # Only denoise if there are noise eigenvalues
            avg_noise = np.mean(e_val[n_signals:])
            e_val_denoised[n_signals:] = avg_noise
        
        # 5. Reconstruct correlation matrix
        corr_denoised = e_vec @ np.diag(e_val_denoised) @ e_vec.T
        
        # 6. Rescale diagonal to ensure valid correlation matrix
        diag = np.diag(corr_denoised)
        corr_denoised = corr_denoised / np.sqrt(np.outer(diag, diag))
        
        # Store information
        info = {
            'n_signals': n_signals,
            'n_noise': N - n_signals,
            'e_max_theory': e_max,
            'eigenvalues_original': e_val,
            'eigenvalues_denoised': e_val_denoised,
            'eigenvectors': e_vec,  # Store eigenvectors for factor analysis
            'variance_explained_signals': np.sum(e_val[:n_signals]) / np.sum(e_val),
            'q_ratio': q
        }
        
        self.eigenvalues = e_val
        self.eigenvectors = e_vec
        self.n_signals = n_signals
        
        return corr_denoised, info
    
    def denoise_covariance(self, cov: np.ndarray, q: float, alpha: float = 0) -> Tuple[np.ndarray, Dict]:
        """
        Denoise a covariance matrix by converting to correlation, denoising, and converting back
        
        Args:
            cov: Covariance matrix (N x N)
            q: Ratio N/T (variables/observations)
            alpha: Shrinkage parameter
            
        Returns:
            Tuple of (denoised_covariance_matrix, info_dict)
        """
        # Extract volatilities
        vols = np.sqrt(np.diag(cov))
        
        # Convert to correlation
        corr = cov / np.outer(vols, vols)
        
        # Denoise correlation
        corr_denoised, info = self.denoise_correlation(corr, q, alpha)
        
        # Convert back to covariance
        cov_denoised = np.diag(vols) @ corr_denoised @ np.diag(vols)
        
        self.cov_matrix = cov
        self.denoised_cov = cov_denoised
        
        return cov_denoised, info
    
    def update_ewma_cov(self, prev_cov: np.ndarray, new_return: np.ndarray, 
                        mean_return: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Online EWMA update for covariance matrix
        
        Important: Uses demeaned returns for proper covariance estimation
        
        Args:
            prev_cov: Previous covariance matrix (N x N)
            new_return: New return vector (N,) or (N, 1)
            mean_return: Mean return vector for demeaning (if None, assumes zero mean)
            
        Returns:
            Updated covariance matrix
        """
        # Ensure column vector
        if new_return.ndim == 1:
            new_return = new_return.reshape(-1, 1)
        
        # Demean returns
        if mean_return is not None:
            mean_return = mean_return.reshape(-1, 1)
            new_return = new_return - mean_return
        
        # EWMA update: Cov_t = λ * Cov_{t-1} + (1-λ) * r_t * r_t'
        updated_cov = (self.decay_ewma * prev_cov + 
                      (1 - self.decay_ewma) * (new_return @ new_return.T))
        
        return updated_cov
    
    def select_optimal_window(self, n_assets: int, target_q: float = 0.2, 
                             max_window_days: int = 20) -> int:
        """
        Select optimal rolling window size based on q ratio constraint
        
        Using q = N/T convention (number of assets / time periods).
        For good MP theory: q should be small (q <= 0.5, ideally q <= 0.2)
        This means T should be much larger than N (T >= 2N, ideally T >= 5N)
        
        Args:
            n_assets: Number of assets (N)
            target_q: Target q ratio N/T (default: 0.2, meaning T=5N)
            max_window_days: Maximum window size in trading days (default: 20)
            
        Returns:
            Optimal number of time periods (T)
        """
        # Assuming 390 minutes per trading day (6.5 hours)
        mins_per_day = 390
        
        # Calculate T needed for target q ratio: q = N/T => T = N/q
        target_T = int(np.ceil(n_assets / target_q))
        
        # Maximum T based on day constraint
        max_T = max_window_days * mins_per_day
        
        # Choose T (prioritize target_q, but cap at max days)
        optimal_T = min(target_T, max_T)
        
        # Ensure T is at least N (needed for valid covariance)
        optimal_T = max(optimal_T, n_assets)
        
        # Calculate actual q
        actual_q = n_assets / optimal_T
        
        print(f"Window selection for N={n_assets} assets:")
        print(f"  Target q=N/T: {target_q:.2f}")
        print(f"  Required T for q={target_q}: {target_T} mins ({target_T/mins_per_day:.1f} days)")
        print(f"  Selected T: {optimal_T} mins ({optimal_T/mins_per_day:.1f} days)")
        print(f"  Actual q ratio (N/T): {actual_q:.3f}")
        
        if actual_q > 0.5:
            print(f"  ⚠️  Warning: q={actual_q:.3f} > 0.5, MP theory may not apply well")
            print(f"     Consider increasing window or reducing number of assets")
        elif actual_q <= 0.2:
            print(f"  ✓ Excellent: q={actual_q:.3f} <= 0.2, good MP theory conditions")
        
        return optimal_T
    
    def visualize_eigenvalues(self, info: Dict, save_path: Optional[str] = None):
        """
        Visualize eigenvalue distribution vs Marcenko-Pastur theory
        
        Args:
            info: Info dict from denoise_correlation/denoise_covariance
            save_path: Optional path to save the figure
        """
        e_val = info['eigenvalues_original']
        e_val_denoised = info['eigenvalues_denoised']
        q = info['q_ratio']
        n_signals = info['n_signals']
        e_max = info['e_max_theory']
        
        # Get MP theoretical PDF
        mp_pdf, _ = self.marcenko_pastur_pdf(var=1.0, q=q, pts=1000)
        
        # Get empirical KDE
        x_emp, pdf_emp = self.fit_empirical_kde(e_val)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Eigenvalue distribution
        ax = axes[0]
        ax.hist(e_val, bins=50, density=True, alpha=0.6, label='Empirical', color='skyblue')
        ax.plot(mp_pdf.index, mp_pdf.values, 'r-', linewidth=2, label='MP Theory')
        ax.plot(x_emp, pdf_emp, 'b--', linewidth=2, label='Empirical KDE')
        ax.axvline(e_max, color='green', linestyle='--', linewidth=2, label=f'MP threshold (λ_max={e_max:.3f})')
        ax.set_xlabel('Eigenvalue')
        ax.set_ylabel('Density')
        ax.set_title(f'Eigenvalue Distribution (q={q:.2f}, N_signals={n_signals})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right plot: Eigenvalue spectrum (scree plot)
        ax = axes[1]
        ax.plot(range(1, len(e_val)+1), e_val, 'o-', label='Original', markersize=4)
        ax.plot(range(1, len(e_val_denoised)+1), e_val_denoised, 's-', 
                label='Denoised', markersize=4, alpha=0.7)
        ax.axhline(e_max, color='green', linestyle='--', linewidth=2, label='MP threshold')
        ax.axvline(n_signals, color='red', linestyle=':', linewidth=2, label=f'Signal cutoff ({n_signals})')
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Eigenvalue Spectrum (Scree Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.close()  # Close instead of show() to avoid blocking in scripts
        
        return fig
    
    def get_factor_loadings(self, info: Dict, ticker_list: list, n_factors: Optional[int] = None) -> pd.DataFrame:
        """
        Extract ticker loadings on each signal factor (eigenvector)
        
        Args:
            info: Info dict from denoise_correlation/denoise_covariance containing eigenvectors
            ticker_list: List of ticker names corresponding to matrix rows/columns
            n_factors: Number of top factors to extract (default: all signal factors)
            
        Returns:
            DataFrame with tickers as rows and factors as columns
            Shape: (N_tickers, n_factors)
        """
        eigenvectors = info['eigenvectors']
        n_signals = info['n_signals']
        
        # Default: extract all signal factors
        if n_factors is None:
            n_factors = n_signals
        else:
            n_factors = min(n_factors, n_signals)
        
        # Extract signal eigenvectors (first n_signals columns)
        signal_vectors = eigenvectors[:, :n_factors]
        
        # Create DataFrame with ticker names
        factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
        loadings_df = pd.DataFrame(
            signal_vectors,
            index=ticker_list,
            columns=factor_names
        )
        
        return loadings_df
    
    def analyze_factor_composition(self, info: Dict, ticker_list: list, 
                                   n_top: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Analyze which tickers contribute most to each signal factor
        
        Args:
            info: Info dict from denoise_correlation/denoise_covariance
            ticker_list: List of ticker names
            n_top: Number of top contributors to show per factor
            
        Returns:
            Dictionary mapping factor name to DataFrame of top contributors
        """
        loadings_df = self.get_factor_loadings(info, ticker_list)
        n_signals = info['n_signals']
        eigenvalues = info['eigenvalues_original']
        
        factor_compositions = {}
        
        for i in range(n_signals):
            factor_name = f'Factor_{i+1}'
            
            # Get loadings for this factor
            factor_loadings = loadings_df[factor_name].copy()
            
            # Sort by absolute value to get strongest contributors
            top_loadings = factor_loadings.abs().sort_values(ascending=False).head(n_top)
            
            # Create summary DataFrame
            summary = pd.DataFrame({
                'ticker': top_loadings.index,
                'loading': factor_loadings.loc[top_loadings.index].values,
                'abs_loading': top_loadings.values
            })
            
            # Add variance info
            var_explained = eigenvalues[i] / eigenvalues.sum() * 100
            
            factor_compositions[factor_name] = {
                'loadings': summary,
                'eigenvalue': eigenvalues[i],
                'variance_explained_pct': var_explained
            }
        
        return factor_compositions
    
    def print_factor_analysis(self, info: Dict, ticker_list: list, n_top: int = 10):
        """
        Pretty print factor analysis showing top contributors to each factor
        
        Args:
            info: Info dict from denoise_correlation/denoise_covariance
            ticker_list: List of ticker names
            n_top: Number of top contributors to show per factor
        """
        compositions = self.analyze_factor_composition(info, ticker_list, n_top)
        n_signals = info['n_signals']
        
        print("\n" + "="*80)
        print(f"FACTOR ANALYSIS: {n_signals} Signal Factors Identified")
        print("="*80)
        
        for i in range(n_signals):
            factor_name = f'Factor_{i+1}'
            comp = compositions[factor_name]
            
            print(f"\n{factor_name} (λ={comp['eigenvalue']:.3f}, explains {comp['variance_explained_pct']:.2f}% variance)")
            print("-" * 70)
            print(f"{'Rank':<6} {'Ticker':<10} {'Loading':<12} {'|Loading|':<12}")
            print("-" * 70)
            
            for rank, row in enumerate(comp['loadings'].itertuples(), 1):
                print(f"{rank:<6} {row.ticker:<10} {row.loading:>11.4f} {row.abs_loading:>11.4f}")
        
        print("\n" + "="*80)


def compute_returns_from_panel(panel_df: pl.DataFrame, price_col: str = 'mid-price',
                               return_type: str = 'log') -> pl.DataFrame:
    """
    Compute returns from panel data
    
    Args:
        panel_df: Panel DataFrame with columns ['timestamp', 'ticker', price_col, ...]
        price_col: Column name for price (default: 'mid-price')
        return_type: 'log' or 'simple'
        
    Returns:
        DataFrame with returns added as 'return' column
    """
    if return_type == 'log':
        returns_df = panel_df.sort(['ticker', 'timestamp']).with_columns([
            (pl.col(price_col).log() - pl.col(price_col).log().shift(1))
            .over('ticker')
            .alias('return')
        ])
    else:  # simple returns
        returns_df = panel_df.sort(['ticker', 'timestamp']).with_columns([
            ((pl.col(price_col) - pl.col(price_col).shift(1)) / pl.col(price_col).shift(1))
            .over('ticker')
            .alias('return')
        ])
    
    # Drop NaN returns (first observation for each ticker)
    returns_df = returns_df.drop_nulls(subset=['return'])
    print("Df returns : ")
    print(returns_df)
    return returns_df


def pivot_returns_to_matrix(returns_df: pl.DataFrame) -> Tuple[np.ndarray, list, pd.DatetimeIndex]:
    """
    Pivot long-format returns to wide matrix format
    
    Args:
        returns_df: Long-format DataFrame with ['timestamp', 'ticker', 'return']
        
    Returns:
        Tuple of (returns_matrix, ticker_list, datetime_index)
    """
    # Pivot to wide format
    returns_wide = returns_df.pivot(
        values='return',
        index='timestamp',
        columns='ticker'
    )
    
    # Convert to pandas for easier manipulation
    returns_pd = returns_wide.to_pandas()
    returns_pd = returns_pd.set_index('timestamp')
    
    # Extract components
    returns_matrix = returns_pd.values
    ticker_list = returns_pd.columns.tolist()
    datetime_index = returns_pd.index
    
    return returns_matrix, ticker_list, datetime_index


def run_covariance_denoise(input_path): 
    #Load panel data : 
    panel_data = pl.read_parquet(input_path)

    #Compute returns : 
    returns = compute_returns_from_panel(panel_data, price_col='mid-price',return_type='log')

    #Pivot returns matrix : 
    returns_matrix, tickers, timestamps = pivot_returns_to_matrix(returns)

    #Check for NaN values : 
    nan_pct = np.isnan(returns_matrix).sum() / returns_matrix.size * 100
    print(f"  NaN percentage: {nan_pct:.2f}%")

    #Initialize denoiser : 
    N = returns_matrix.shape[1]  #Number of assets 
    T_total = returns_matrix.shape[0]  #Total time periods

    denoiser = CovarianceDenoiser(input_path=input_path)

    #Select window such that q = N/T = 1/5
    T_window = denoiser.select_optimal_window(
        n_assets=N,
        target_q=0.2, # q = N/T = 1/5 = 0.2
        max_window_days=10)
    
    #Compute initial covariance based on first window 
    window_returns = returns_matrix[:T_window]

    #Handle NaN by forward fill (prevent look ahead bias)
    if np.isnan(window_returns).any():
        # Convert to DataFrame for easier NaN handling
        window_df = pd.DataFrame(window_returns, columns=tickers)
        # Forward fill NaN values (propagate last valid observation forward)
        window = window_df.ffill()

        # If still NaN (entire column is NaN), fill with 0
        window_df = window_df.fillna(0.0)
        window_returns = window_df.values
        remaining_nans = np.isnan(window_returns).sum()
        print(f"    Forward/backward fill complete. Remaining NaN: {remaining_nans}")
    
    #Compute raw covariance 
    cov_raw = np.cov(window_returns, rowvar=False)
    q = N / T_window  
    print(f"  q ratio (N/T): {q:.3f}")

    #Denoise covaiance matrix : 
    cov_denoised, info = denoiser.denoise_covariance(cov_raw, q=q)
    print(f"    Signal eigenvalues: {info['n_signals']}")
    print(f"    Noise eigenvalues: {info['n_noise']}")
    print(f"    Variance explained by signals: {info['variance_explained_signals']:.2%}")
    print(f"    MP threshold (λ_max): {info['e_max_theory']:.4f}")

    #Visualize Eigenvalue distrib : 
    denoiser.visualize_eigenvalues(info, save_path='eigenvalue_analysis.png')

    #Compute condition number to check stability 
    cond_raw = np.linalg.cond(cov_raw)
    cond_denoised = np.linalg.cond(cov_denoised)
    print(f"    Condition number (raw): {cond_raw:.2e}")
    print(f"    Condition number (denoised): {cond_denoised:.2e}")
    print(f"    Improvement: {cond_raw/cond_denoised:.1f}x")
    
    #Factor loadings analysis
    # Print detailed factor analysis
    denoiser.print_factor_analysis(info, tickers, n_top=10)
    
    # Get full loadings matrix for further analysis
    loadings_df = denoiser.get_factor_loadings(info, tickers)
    print(f"\nFull loadings matrix shape: {loadings_df.shape}")
    print(f"(Saved to 'factor_loadings.csv' for further analysis)")
    loadings_df.to_csv('factor_loadings.csv')
    
    # Save matrices
    print("\nSaving covariance matrices...")
    np.save('covariance_raw.npy', cov_raw)
    np.save('covariance_denoised.npy', cov_denoised)
    print("  Saved: covariance_raw.npy, covariance_denoised.npy")
    
    return window_returns, cov_denoised, info, tickers










