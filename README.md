# Poker Player Profiling: Heuristics vs. K-Means Clustering

A machine learning project that profiles poker players by comparing traditional heuristic-based categorization with data-driven K-Means clustering techniques.

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Running the Analysis](#running-the-analysis)
- [Features & Methodology](#features--methodology)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Team](#team)
- [References](#references)

## About the Project

### Problem Statement

Poker is a popular decision-making game where understanding player behavior provides strategic advantages. Players are traditionally categorized into types based on statistics like VPIP (Voluntarily Put Money in Pot) and PFR (Pre-Flop Raise):

- **TAG** (Tight-Aggressive): Plays few hands but plays them aggressively
- **LAG** (Loose-Aggressive): Plays many hands aggressively
- **The Fish** (Loose-Passive): Plays many hands passively
- **The Rock** (Tight-Passive): Plays few hands passively

### Objective

This project compares two approaches to player profiling:

1. **Heuristic Method**: Traditional rule-based categorization using VPIP/PFR thresholds
2. **K-Means Clustering**: Data-driven approach that discovers player clusters based on behavioral features

We evaluate both approaches using the **Silhouette Score** metric to determine which produces better-separated, more coherent player groupings.

## Dataset

### Source

[IRC Poker Database](https://poker.cs.ualberta.ca/) - University of Alberta

- Over 10 million hands of Texas Hold'em poker
- Played on Internet Relay Chat (IRC) servers
- Spans from 1995-1999

### Data Format

Raw hand history logs containing:

- Unique hand identifiers
- Number of players per hand
- Betting actions across rounds (pre-flop, flop, turn, river)
- Player participation information

### Data Processing

The raw hand histories are preprocessed to extract:

- **Player-level statistics** aggregated across multiple hands
- **Behavioral features** including VPIP and PFR
- **Structured CSV formats** for clustering analysis

## Project Structure

```
poker-player-profiling/
├── README.md                           # This file
├── data/
│   ├── raw/                           # Original IRC poker hand histories
│   │   ├── holdem/                    # Texas Hold'em games
│   │   ├── 7stud/                     # 7-Card Stud games
│   │   ├── omaha/                     # Omaha games
│   │   └── [other game types]/
│   ├── intermediate/                  # Partially processed data
│   │   ├── raw_data_sampled.csv       # Sample of raw data for testing
│   │   ├── holdem_human_parse_summary.csv
│   │   └── kmeans_model_selection_all_human_holdem.csv
│   └── processed/                     # Final processed datasets
│       ├── player_stats_all_human_holdem.csv      # Player statistics
│       ├── player_stats_sampled.csv               # Sampled player stats
│       ├── player_clusters_all_human_holdem.csv   # K-Means cluster assignments
│       ├── player_cluster_centroids_all_human_holdem.csv
│       └── holdem_human_channel_summary.csv
├── notebooks/
│   ├── 01-exploratory-data-analysis.ipynb              # EDA on sampled data
│   ├── 02-kmeans-clustering-sampled.ipynb              # K-Means on sampled data
│   ├── 03-preprocess-human-holdem.ipynb                # Full data preprocessing
│   └── 04-kmeans-all-human-holdem.ipynb                # K-Means on full dataset
├── figures/                            # Generated visualizations
├── reports/
│   ├── 01-proposal/                   # Initial project proposal
│   └── 02-methodology/                # Detailed methodology
└── .git/                              # Version control
```

## Requirements

### Software

- **Python** 3.7+
- **Jupyter Notebook** or **JupyterLab**

### Python Libraries

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning (K-Means, Silhouette Score)
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `jupyter` - Interactive notebooks

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/lakshyajain/poker-player-profiling.git
cd poker-player-profiling
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n poker-profiling python=3.9
conda activate poker-profiling
```

### Step 3: Install Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or create a requirements file and install from it:

```bash
pip install -r requirements.txt
```

_Note: If a requirements.txt doesn't exist in the repository, you can create one by running:_

```bash
pip freeze > requirements.txt
```

### Step 4: Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

## Running the Analysis

The analysis pipeline consists of four main notebooks, designed to be run sequentially:

### **Notebook 1: Exploratory Data Analysis**

**File**: `notebooks/01-exploratory-data-analysis.ipynb`

Performs initial exploration on a sampled subset of the data:

- Load and inspect the sampled player statistics
- Visualize distributions of VPIP and PFR
- Identify outliers and data quality issues
- Generate preliminary histograms and scatter plots

**When to run**: Start here to understand the data structure and features.

### **Notebook 2: K-Means Clustering on Sampled Data**

**File**: `notebooks/02-kmeans-clustering-sampled.ipynb`

Tests K-Means clustering on the sampled dataset:

- Perform model selection (determine optimal number of clusters)
- Train K-Means models with varying k values (typically 2-10)
- Calculate Silhouette Scores for each k
- Generate elbow plot to visualize optimal cluster count
- Visualize clusters in 2D space (VPIP vs. PFR)

**When to run**: Use this for quick prototyping and parameter tuning before running on full data.

### **Notebook 3: Full Data Preprocessing**

**File**: `notebooks/03-preprocess-human-holdem.ipynb`

Preprocesses the entire Texas Hold'em dataset:

- Parse raw hand history files
- Extract player statistics across all hands
- Aggregate behavioral features (VPIP, PFR, etc.)
- Handle missing data and clean inconsistencies
- Export processed player statistics to CSV

**Expected runtime**: Several hours (processes millions of hands)

**When to run**: After confirming the pipeline works on sampled data. This generates the complete dataset for final analysis.

### **Notebook 4: K-Means Clustering on Full Data**

**File**: `notebooks/04-kmeans-all-human-holdem.ipynb`

Performs clustering on the entire processed dataset:

- Load all player statistics
- Train K-Means with optimal k value (from Notebook 2)
- Compare K-Means clusters with heuristic categories
- Calculate Silhouette Scores for both approaches
- Generate comprehensive visualizations and analysis
- Export cluster assignments and centroids

**When to run**: Final step - runs after all preprocessing is complete.

### Recommended Workflow

```
Start here
    ↓
01-EDA (sampled data) ← Quick understanding
    ↓
02-KMeans-Sampled ← Parameter tuning
    ↓
03-Preprocess (full data) ← Takes several hours
    ↓
04-KMeans-FullData ← Final results & comparison
```

## Features & Methodology

### Feature Engineering

**Primary Features** (used in both heuristic and clustering approaches):

- **VPIP** (Voluntarily Put Money in Pot): Percentage of hands where player commits money pre-flop
- **PFR** (Pre-Flop Raise): Percentage of hands where player raises pre-flop

These features capture two key dimensions of player behavior:

- **Tightness**: How selective the player is (low VPIP = tight)
- **Aggressiveness**: How often the player raises (high PFR = aggressive)

### Heuristic Categories

Traditional rule-based grouping using VPIP/PFR thresholds:

| Category | VPIP Range | PFR Range |
| -------- | ---------- | --------- |
| Rock     | Low        | Low       |
| Fish     | High       | Low       |
| LAG      | High       | High      |
| TAG      | Low        | High      |

### K-Means Clustering

Unsupervised machine learning approach:

1. Normalize feature vectors (VPIP, PFR)
2. Train K-Means for multiple k values (2-10)
3. Calculate Silhouette Score for each k
4. Select k with highest Silhouette Score
5. Analyze resulting clusters and compare with heuristic categories

### Evaluation Metric

**Silhouette Score**: Ranges from -1 to 1

- Values near 1: Well-separated, compact clusters
- Values near 0: Overlapping clusters
- Negative values: Samples in wrong clusters

## Results

### Output Files

**Processed Data** (`data/processed/`):

- `player_stats_all_human_holdem.csv` - Aggregated player statistics
- `player_clusters_all_human_holdem.csv` - K-Means cluster assignments
- `player_cluster_centroids_all_human_holdem.csv` - Cluster center coordinates

**Model Selection** (`data/intermediate/`):

- `kmeans_model_selection_all_human_holdem.csv` - Silhouette scores for different k values

**Visualizations** (`figures/`):

- Cluster scatter plots (VPIP vs. PFR)
- Elbow plots for k selection
- Silhouette score comparisons
- Heuristic vs. K-Means comparison plots

## Ethical Considerations

### Potential Misuse

Player profiling tools can be used to build cheating bots. **This project is strictly for research and strategy analysis purposes, not for real-time use in live games or any illegal activities.**

### Data Bias

- Dataset originates from 1995-1999 IRC poker games
- Player behavior may differ significantly from modern online/live poker
- Results reflect historical playing styles, not current trends
- Findings should not be overgeneralized to contemporary poker populations

### Responsible Use

- Use insights only for personal strategy development
- Do not use for unauthorized surveillance or competitive advantage
- Respect player privacy and terms of service agreements
- Consider the historical context when interpreting results

## Team

**Project Members**:

- Lakshya J.
- Aryan S.
- Shreya A.
- Claire S.

**Course**: Algorithm Implementation (Option B)

## References

1. Brown, N., & Sandholm, T. (2017). Libratus: The Superhuman AI for No-Limit Poker. In _IJCAI_, pp. 5226-5228.

2. Moravčík, M., et al. (2017). DeepStack: Expert-level artificial intelligence in heads-up no-limit poker. _Science_, 356(6337), 508-513.

3. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In _Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability_, Vol. 1, pp. 281-297.

4. University of Alberta. IRC Poker Database. [https://poker.cs.ualberta.ca/](https://poker.cs.ualberta.ca/)

---

**Last Updated**: May 2026

For questions or issues, please open an issue on GitHub or contact the team members.
