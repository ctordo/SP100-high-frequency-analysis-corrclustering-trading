import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def panel_to_table(df_panel1min, attribute:str, aggfunc:str, disp:bool=False):
    df_pivot = (
        df_panel1min
        .pivot_table(
            index="timestamp",
            columns="ticker",
            values=attribute,
            aggfunc=aggfunc   # can be "last", "mean", "first", etc.
        )
    )
    # Set index to datetime type
    df_pivot.index = pd.to_datetime(df_pivot.index)
    
    if(disp):
        print(f"Succesfully returning Table (index:timestamps, columns:tickers):")
        display(df_pivot)

    return df_pivot

def plot_daily_nan_proportion_heatmap(df_wide, title, list_tickers, gradient_count, gradient_prop):
    """
    Plots a heatmap showing the number of NaNs per asset per day.
    Plots a heatmap showing the proportion of NaNs per asset per day.
    
    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide dataframe with timestamps and asset columns.
        Timestamp must be the index.
    """

    df = df_wide.copy()

    # Ensure timestamp is datetime index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)

    # ==== Count NaNs per week per asset ====
    weekly_nan_counts = (
        df
        .isna()
        .resample("D")
        .sum()
    )
    # ==== Compute weekly NaN proportion ====
    weekly_nan_proportion = (
        df
        .isna()
        .resample("D")
        .mean()   # mean of boolean = proportion
    )

    # Plot heatmaps
    fig, ax = plt.subplots(1, 2, figsize = (24, 16), sharey=True)
    fig.suptitle(title)

    hm1 = sns.heatmap(
        weekly_nan_counts,
        ax=ax[0],
        cmap=gradient_count,
        cbar= False,
        linewidths=0.2
    )
    ax[0].set_xlabel("Asset")
    ax[0].set_xticks(np.arange(len(list_tickers)))
    ax[0].set_xticklabels(list_tickers, fontsize=8, rotation=90)
    ax[0].set_ylabel("Day")
    ax[0].set_title("Daily Number of Missing Values per Asset")

    # Create horizontal colorbar above
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("top", size="3%", pad=0.3)
    cbar1 = fig.colorbar(hm1.collections[0], cax=cax1, orientation="horizontal")
    cbar1.set_label("Number of NaNs")
    cax1.xaxis.set_label_position("top")
    cax1.xaxis.tick_top()
    
    hm2 = sns.heatmap(
        weekly_nan_proportion,
        ax=ax[1],
        cmap=gradient_prop,
        cbar = False,
        vmin=0,
        #vmax=1,
        fmt=".2f",
        annot_kws={"size": 6, "color": "black"},
        linewidths=0.2
    )
    ax[1].set_xlabel("Asset")
    ax[1].set_xticks(np.arange(len(list_tickers)))
    ax[1].set_xticklabels(list_tickers, fontsize=8, rotation=90)
    ax[1].set_ylabel("Day")
    ax[1].set_title("Daily Proportion of Missing Values per Asset")

    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("top", size="3%", pad=0.3)
    cbar2 = fig.colorbar(hm2.collections[0], cax=cax2, orientation="horizontal")
    cbar2.set_label("Proportion of NaNs")
    cax2.xaxis.set_label_position("top")
    cax2.xaxis.tick_top()

    plt.tight_layout()
    plt.show()