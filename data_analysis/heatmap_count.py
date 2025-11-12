"""
Player Metrics Distribution (Count) Heatmap Generator

This script processes individual player data files to calculate performance
metrics (delta, alpha) over a 365-day rolling window.

The core function of this script is to generate a 2D histogram (heatmap)
that visualizes the *distribution* or *count* of (delta, alpha) pairs.
Each bin's color and annotation represent the number of data points
that fall into it.

This script uses the same algebraic metric calculation as
'generate_performance_heatmap.py' but plots the *count* instead of
the weighted average.

ASSUMPTIONS:
- Input Excel files use English column names ('Date', 'Performance Index').
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib as mpl


# ===============================
# Function Definitions
# ===============================

def calculate_window_metrics(subset):
    """
    Calculates delta (a) and alpha (b) using a direct algebraic
    solution based on the min/max performance in the subset.

    Model:
        alpha = max_p - min_p
        delta = min_p / (1 - max_p + min_p)  [which is min_p / (1 - alpha)]

    Returns delta, alpha, and the max performance.
    """
    # Assumes English column 'Performance Index'
    min_p = subset['Performance Index'].min()
    max_p = subset['Performance Index'].max()

    alpha = max_p - min_p
    denominator = 1.0 - max_p + min_p  # Equivalent to (1 - alpha)

    # Safety check for division by zero
    if np.isclose(denominator, 0):
        delta = np.nan
    else:
        delta = min_p / denominator

    # Check for invalid values
    if not (0 <= delta <= 1 and 0 <= alpha <= 1):
        delta, alpha = np.nan, np.nan

    return delta, alpha, max_p


def process_player_windows(file_path):
    """
    Processes a single player's file, calculating metrics
    for each 365-day sliding window.
    """
    try:
        df = pd.read_excel(file_path)
        # Assumes English column 'Date'
        df['date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('date').reset_index(drop=True)

        results = []
        if df.empty:
            return pd.DataFrame(results)

        last_date = df['date'].iloc[-1]

        for i in range(len(df)):
            start_date = df['date'].iloc[i]
            # 365-day window (inclusive of start_date)
            end_date = start_date + pd.Timedelta(days=364)

            if end_date > last_date:
                break  # Window extends past the last match

            subset = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            # Require at least 3 matches in the window
            if len(subset) < 3:
                continue

            delta, alpha, max_perf = calculate_window_metrics(subset)
            results.append({
                'player': os.path.basename(file_path).split('.')[0],
                'start_date': start_date,
                'delta': delta,
                'alpha': alpha,
                'max_performance': max_perf
            })

        return pd.DataFrame(results)
    except Exception as e:
        print(f"  Warning: Could not process {file_path}. Error: {e}")
        return pd.DataFrame()


def plot_count_heatmap(all_data, output_path):
    """
    Plots a highly-styled heatmap of the *count* of data points.
    - 0-value bins are not colored (masked).
    - Bins are forced to be square.
    - Axis labels and colorbar labels are removed.
    """
    plt.figure(figsize=(12, 8))

    # --- 1. Define Bins ---
    # Bins for delta (a). Final bin [0.95, 1.05] captures 1.0
    delta_bins = np.append(np.arange(0.3, 0.96, 0.05), 1.05)
    # Bins for alpha (b)
    alpha_bins = np.arange(0, 0.70, 0.05)

    # --- 2. Calculate 2D Histogram (Counts) ---
    counts, xedges, yedges = np.histogram2d(
        all_data['delta'],
        all_data['alpha'],
        bins=[delta_bins, alpha_bins]
    )

    df_heatmap = pd.DataFrame(
        counts.T,
        index=np.round(yedges[:-1], 2),
        columns=np.round(xedges[:-1], 2)  # Columns are [0.3, ..., 0.95]
    )
    df_heatmap = df_heatmap.iloc[::-1]  # Reverse rows to put 0 at bottom

    # --- 3. Prepare Annotation Matrix ---
    # Create a string matrix for annotations, hiding '0'
    annot_matrix = df_heatmap.fillna(0).astype(int).astype(str)
    annot_matrix[df_heatmap == 0] = ""  # Hide 0 values

    # Create the plot matrix, replacing 0 with NaN so it can be masked
    df_plot = df_heatmap.replace(0, np.nan)

    # --- 4. Plotting ---
    with mpl.rc_context({'font.size': 16}):
        ax = sns.heatmap(
            df_plot,
            cmap='GnBu',
            cbar_kws={'label': None},  # Remove colorbar label
            linewidths=0.5,
            edgecolor='lightgray',
            annot=annot_matrix,  # Use the custom string matrix
            fmt='',  # Annotation format is already string
            annot_kws={"size": 16},
            square=True,  # Force square bins
            mask=df_plot.isna()  # Hide NaN (originally 0) cells
        )

    # --- 5. Customize Axes Ticks (as requested) ---
    ax.set_xticks(np.arange(len(xedges)))

    # Custom x-axis tick labels, correcting '1.05' to '1.0'
    xticklabels = [f"{v:.2f}".rstrip('0').rstrip('.') for v in xedges]
    if xticklabels[-1] == '1.05':
        xticklabels[-1] = '1.0'
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')

    ax.set_yticks(np.arange(len(yedges)))
    ax.set_yticklabels([f"{v:.2f}".rstrip('0').rstrip('.') for v in yedges[::-1]])

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, format='eps', dpi=600, bbox_inches='tight')
    plt.close()  # Close figure to free memory


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":

    # --- Configuration ---
    DATA_DIR = r"YOUR_DATA_FOLDER_PATH"
    OUTPUT_DIR = r"./analysis_results"
    OUTPUT_FILENAME = "heatmap_distribution_count.eps"

    # --- Setup ---
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Input directory not found at {DATA_DIR}")
        print("Please check the 'DATA_DIR' variable.")
    else:
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
                print(f"Created output directory: {OUTPUT_DIR}")
            except Exception as e:
                print(f"Warning: Could not create output directory {OUTPUT_DIR}. {e}")

        all_data = pd.DataFrame()
        file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(('.xlsx', '.xls'))]

        print(f"Processing {len(file_list)} player files from {DATA_DIR}...")

        # Iterate with a progress bar
        for file in tqdm(file_list):
            file_path = os.path.join(DATA_DIR, file)
            player_data = process_player_windows(file_path)
            all_data = pd.concat([all_data, player_data], ignore_index=True)

        if all_data.empty:
            print("No valid data was processed. Exiting.")
        else:
            # --- Data Filtering ---
            # Drop any rows where metrics calculation failed
            valid_data = all_data.dropna(subset=['delta', 'alpha'])

            # Optional: Filter to specific ranges
            # valid_data = valid_data[
            #     (valid_data['delta'] >= 0.2) & (valid_data['delta'] <= 1) &
            #     (valid_data['alpha'] >= 0) & (valid_data['alpha'] <= 0.6)
            # ]

            print(f"\nTotal valid data points: {len(valid_data)}")

            # --- Plotting ---
            output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
            print(f"Generating count heatmap... saving to {output_path}")
            plot_count_heatmap(valid_data, output_path)
            print("Heatmap generation complete.")