from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np


def barplot_distrib_clusters(df_clusters, n_clusters):
    # Plot the distribution in clusters
    plt.figure(figsize=(4, 3))
    plt.bar(df_clusters["cluster"].value_counts().index, df_clusters["cluster"].value_counts().values, width=0.6, alpha=0.7)
    plt.xticks(range(n_clusters+1)[1:])
    plt.ylabel("Cluster")
    plt.ylabel("Nb assets")
    plt.grid(axis="y")
    plt.show()

    return df_clusters

def draw_positions_table(np_positions, TICKERS):
    """
    Create a heatmap representing positions in assets over time.
    """
    # Mask zeros so they are transparent
    masked = np.ma.masked_equal(np_positions, 0)

    vmin = min(masked.min(), -masked.max())
    vmax = max(masked.max(), -masked.min())
    bounds = np.arange(vmin - 0.5, vmax + 1.5)
    cmap = plt.get_cmap("coolwarm", len(bounds) - 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    # Horizontal colorbar on top
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation='horizontal',
        shrink=0.7,
        label="Value",
        ticks=np.arange(vmin, vmax + 1),
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
    # Plot PnL time series
	plt.figure(figsize=(12, 6))
	plt.plot(pnl_curve.index, pnl_curve.values, linewidth=2, color = 'red')
	plt.axhline(y=0, color='black')
	plt.title('Cumulative PnL over time')
	plt.xlabel('Time')
	plt.xticks([])
	plt.ylabel('Cumulative PnL')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()