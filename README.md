# High-frequency S&P100 Analysis

## Overview

This project was conducted as part of our participiation in the EPFL course Financial Big Data (FIN-252), under the supervision of Prof. Damien Challet. It aims to demonstrate the application of big data techniques to high-frequency financial markets by implementing a complete pipeline from raw limit order book data to an executable pairs trading strategy. We process roughly **40 GB of nanosecond-level S&P100 data** and transform it into minute-frequency panels (addressing computational efficiency, missing values, and estimation noise). We then perform and compare **various time-series correlation clustering techniques** (Leiden, Louvain...). We finally use our findings to build a functionnal **pairs trading strategy**, carefully avoiding any look-ahead bias or data leakage.

We have meticulously documented our entire analysis and results, along with interpretations and improvement suggestions, in the report `FBD2025_report.pdf`. Guidelines to replicate our results are stated at the end of this ReadMe file.

## Pipeline Components

Our work flow comprises four phases:

### 1. Data Preprocessing 

**Purpose**: Transform raw high-frequency data into clean, analysis-ready format.

**Key Steps**:
- Load raw parquet files (~40 GB of tick data)
- Filter by date range (default: Sep-Dec 2008)
- Remove duplicates via VWAP aggregation
- Identify and exclude assets with incomplete coverage
- Resample to 1-minute intervals using VWAP
- Create unified panel data (timestamp × ticker format)

**Main Files**:
- `main_preprocessing.py` - Preprocessing execution script
- `utils/preprocessing_utils/preprocessing_utils.py` - Core preprocessing functions

**Output**: Clean panel data with 79 assets at 1-minute frequency

### 2. Formatting

**Purpose**: Handle missing values and prepare data structures for graph-based clustering algorithms.

**Key Steps**:
- Format stock prices
- Missing value treatment and NaN proportion visualizations
- Compute stock returns as percentage changes

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
- `utils/clustering_utils/Utils.py` - Helper functions 
- `utils/clustering_utils/plots.py` - Visualization utilities
- `utils/clustering_utils/clustering_analysis_report.ipynb` - Clustering analysis notebook

**Output**: Asset pairs with high statistical co-movement

### 4. Pairs Trading

**Purpose**: Implement statistical arbitrage strategies on periodically identified pairs and evaluate financial performance.

**Main Files**:
- `main_trading.py` - Trading execution script
- `utils/trading_utils/trading_utils.py` - Trading strategy logic
- `utils/trading_utils/trading_visuals.py` - Performance visualization and analysis


## Replication

To reproduce our entire analysis and results, please closely follow the below guidelines.

#### Data Setup

- Download our code folder `FBD2025`. Place it in some high root (possibly the Desktop).
- Download the required intial data from Google Drive: https://drive.google.com/drive/folders/1xProHPN1YtKKkLh8917-R50KtgXmy_rO. This should download a folder named `Data_parquet`, containing 85 parquet files (1 per asset).
- Create a folder named `FBD_local_data` and place it in the same root as the `FBD2025` (again, possibly the Desktop). `FBD2025`and `FBD_local_data` must be at the same hierarchy level.
- Place the downloaded `Data_parquet` folder inside `FBD_local_data`.
- You are ready to go! Additional outputs will be generated in `FBD_local_data` during execution.

To summarize, here is what your set-up should look like:
```
Desktop/                    # (or any high root)
├── FBD2025/                # Folder with code files
│   ├── main.py
│   ├── utils.py              
│   └── ...                 
│
└── FBD_local_data/         # Folder with initial parquet data 
    └── Data_parquet/
        ├── AA.N.parquet
        ├── ...
        └── XRX.N.parquet
```

#### Requirements
```bash
conda create -n finbigdata python=3.14
conda activate finbigdata
pip install polars numpy pandas matplotlib seaborn scipy networkx jupyter scikit-network community
```

#### Execute the complete analysis 

```bash
jupyter notebook master_notebook.ipynb
```

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
    │   ├── plots.py
    |   └── clustering_analysis_report.ipynb
    └── trading_utils/
        ├── trading_utils.py
        └── trading_visuals.py
```




## Technical Details

### Data Characteristics
- **Assets**: S&P100 constituents
- **Period**: January 2004 - December 2008 (full coverage for 79 assets)
- **Frequency**: Original tick data (~microsecond) → Resampled to 1-minute
- **Size**: ~41.48 GB raw data
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

## Authors

Dard Timothé, 
Pécaut Marius,
Tordo Cyprien.

EPFL - Financial Big Data Course, 2025
