from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np


def barplot_distrib_clusters(df_clusters, n_clusters):
    """
    Plot the distribution of assets across clusters as a bar chart.

    Parameters
    ----------
    df_clusters : pandas.DataFrame
    n_clusters : int
    """
    plt.figure(figsize=(4, 3))
    plt.bar(df_clusters["cluster"].value_counts().index, df_clusters["cluster"].value_counts().values, width=0.6, alpha=0.7)
    plt.xticks(range(n_clusters+1)[1:])
    plt.ylabel("Cluster")
    plt.ylabel("Nb assets")
    plt.grid(axis="y")
    plt.show()

def draw_positions_table(np_positions, TICKERS):
    """
    Display a heatmap of asset positions over time.

    Parameters
    ----------
    np_positions : numpy.ndarray
        2D array of shape (time, assets) representing position values.
        Zero values are masked in the visualization.
    TICKERS : list of str
        List of asset tickers corresponding to the columns of
        ``np_positions`` (excluding the final summary column).
    """
    # Mask zeros so they are transparent
    masked = np.ma.masked_equal(np_positions, 0)

    vminmax = max(abs(masked.min()), abs(masked.max()))
    bounds = np.arange(-vminmax - 0.5, vminmax + 1.5)
    cmap = plt.get_cmap("coolwarm", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(15, 9))
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation='vertical',
        shrink=0.7,
        label="Value",
        ticks=np.arange(-int(vminmax/5)*5-5, int(vminmax/5)*5+5, 5),
        pad=0.1  # space between plot and colorbar
    )

    # Plot formatting
    ax.set_xlabel("Assets")
    ax.set_ylabel("Time ticks (minute)")
    ax.set_title("Positions")
    ax.set_xticks(range(np_positions.shape[1]))
    ax.set_xticklabels(TICKERS + ["Sum (check0)"], rotation=90, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cumulative_pnl(pnl_curve):
    """
    Plot the cumulative profit and loss (PnL) over time.

    Parameters
    ----------
    pnl_curve : pandas.Series
        Time-indexed Series containing cumulative PnL values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(pnl_curve.index, pnl_curve.values, linewidth=2, color = 'blue')
    plt.axhline(y=0, color='black', linestyle='--', alpha = 0.75)
    plt.title('Cumulative PnL over time')
    plt.xlabel('Time')
    plt.xticks([])
    plt.ylabel('Cumulative PnL')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()