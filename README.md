# Financial Big Data 2025
**EPFL - Financial Big Data Course Project**

High-frequency trading data analysis pipeline and implementation of a pairs trading strategy for S&P 100 stocks (2004-2008).

---

## Project Overview

This project implements a complete pipeline for analyzing high-frequency BBO (Best Bid and Offer) data and developing pairs trading strategies through graph-based clustering. The pipeline consists of four main components:

1. **Data Preprocessing** - Clean and aggregate raw tick data using Polars
2. **Data Formatting** - Prepare data structures for clustering algorithms  
3. **Clustering** - Identify trading pairs using Leiden/Louvain community detection
4. **Pairs Trading** - Implement and backtest statistical arbitrage strategies

---

## Quick Start

### 1. Data Setup

Download the required data from Google Drive:
```
https://drive.google.com/drive/folders/1xProHPN1YtKKkLh8917-R50KtgXmy_rO
```

**Important**: Place the downloaded `FBD_local_data/` folder in your computer's root directory:
```
/Users/<your-username>/FBD_local_data/
```

The data folder should contain:
- `Data_parquet/` - Raw BBO data for S&P 100 assets
- Additional outputs will be generated here during processing

### 2. Run the Pipeline

Execute the complete workflow by running:
```bash
jupyter notebook master_notebook.ipynb
```

This master notebook illustrates the entire pipeline process ; from raw data to trading signals.

---

## Pipeline Components

### 1. Preprocessing 

**Purpose**: Transform raw high-frequency data into clean, analysis-ready format.

**Key Steps**:
- Load raw parquet files (23.4 GB of tick data)
- Filter by date range (default: Sep-Dec 2008)
- Remove duplicates via VWAP aggregation
- Filter invalid observations (NaN, zero volume)
- Identify and exclude assets with incomplete coverage
- Resample to 1-minute intervals using VWAP
- Create unified panel data (timestamp × ticker format)

**Main Files**:
- `main_preprocessing.py` - Preprocessing execution script
- `utils/preprocessing_utils/preprocessing_utils.py` - Core preprocessing functions


**Output**: Clean panel data with 79 assets at 1-minute frequency

### 2. Formatting

**Purpose**: Prepare data structures for graph-based clustering algorithms.

**Main Files**:
- `main_formatting.py` - Formatting execution script
- `utils/formatting_utils/formatting_utils.py` - Data transformation utilities

### 3. Clustering 

**Purpose**: Identify pairs of co-moving stocks using community detection on correlation graphs.

**Methods**:
- **Leiden Clustering** - High-quality community detection
- **Louvain Clustering** - Fast modularity optimization
- **Marsili-Giada Clustering** - Correlation-based filtering method

**Main Files**:
- `main_clustering.py` - Clustering execution script
- `utils/clustering_utils/Leiden_clustering.py` - Leiden algorithm implementation
- `utils/clustering_utils/Louvain_clustering.py` - Louvain algorithm implementation
- `utils/clustering_utils/Marsili_Giada_clustering.py` - Marsili-Giada implementation
- `utils/clustering_utils/Utils.py` - Helper functions for graph construction
- `utils/clustering_utils/plots.py` - Visualization utilities

**Output**: Asset pairs with high statistical co-movement

### 4. Pairs Trading

**Purpose**: Implement statistical arbitrage strategies on identified pairs.

**Main Files**:
- `main_trading.py` - Trading execution script
- `utils/trading_utils/trading_utils.py` - Trading strategy logic
- `utils/trading_utils/trading_visuals.py` - Performance visualization and analysis

---

## Technical Details

### Data Characteristics
- **Assets**: S&P 100 constituents
- **Period**: January 2004 - December 2008 (full coverage for 79 assets)
- **Frequency**: Original tick data (~microsecond) → Resampled to 1-minute
- **Size**: ~23.4 GB raw data
- **Format**: Parquet files (one per asset)

### Data Quality Checks
- Timestamp coverage validation (2004-01-02 to 2008-12-31)
- Missing data analysis per asset per year
- Automatic exclusion of 7 assets with incomplete coverage (AA, MA, MS, NOV, PM, V, DVN)

### Key Features
- Bid/ask prices and volumes
- Spread (ask - bid)
- Mid-price
- Volume imbalance (bid-ask pressure)
- High-frequency returns

### Technologies
- **Data Processing**: Polars (high-performance DataFrames)
- **Clustering**: NetworkX, python-igraph
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Python 3.14, Jupyter notebooks

---

## Repository Structure

```
FBD2025/
├── README.md
├── master_notebook.ipynb          # Main pipeline execution
├── main_preprocessing.py          # Preprocessing entry point
├── main_formatting.py             # Formatting entry point
├── main_clustering.py             # Clustering entry point
├── main_trading.py                # Trading strategy entry point
└── utils/
    ├── preprocessing_utils/
    │   ├── preprocessing_utils.py
    │   └── datapreprocessing_pandas.py
    ├── formatting_utils/
    │   └── formatting_utils.py
    ├── clustering_utils/
    │   ├── Leiden_clustering.py
    │   ├── Louvain_clustering.py
    │   ├── Marsili_Giada_clustering.py
    │   ├── Utils.py
    │   └── plots.py
    └── trading_utils/
        ├── trading_utils.py
        └── trading_visuals.py
```

---

## Requirements

Install dependencies:
```bash
conda create -n finbigdata python=3.14
conda activate finbigdata
pip install polars numpy pandas matplotlib seaborn scipy networkx jupyter
```

---

## Authors
Tordo Cyprien,
Dard Timothé, 
Pécaut Marius. 

EPFL - Financial Big Data Course, 2025
