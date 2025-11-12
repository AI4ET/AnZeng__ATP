"""
Performance Metrics Heatmap Generator

This script processes individual player data files to calculate performance
metrics (delta, alpha) over a 365-day rolling window.

It then aggregates all player data to generate a 2D heatmap. In this heatmap:
- The X-axis is the 'delta' (a) parameter.
- The Y-axis is the 'alpha' (b) parameter.
- The color/value of each bin represents the weighted average of the
  'Max Performance' within that bin.

The script assumes input Excel files already use English column names
(e.g., 'Date', 'Performance Index').
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ===============================
# Function Definitions
# ===============================

def calculate_window_metrics(subset):
    """
    Calculates delta (a) and alpha (b) using a direct algebraic
    solution based on the min/max performance in the subset.

    This uses a different model than the fsolve-based scripts.
    Model:
        alpha = max_p - min_p
        delta = min_p / (1 - alpha)

    Returns delta, alpha, and the max performance (as the weight).
    """
    min_p = subset['Performance Index'].min()
    max_p = subset['Performance Index'].max()

    alpha = max_p - min_p
    denominator = 1.0 - alpha

    # Safety check for division by zero
    if np.isclose(denominator, 0):
        delta = np.nan
    else:
        delta = min_p / denominator

    # Check for invalid values
    if not (0 <= delta <= 1 and 0 <= alpha <= 1):
        delta = np.nan

    # Return metrics and the max performance (used for weighting the heatmap)
    return delta, alpha, max_p


def process_player(file_path):
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

            # Get all matches within the 1-year window
            subset = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            # Require at least 3 matches in the window to be valid
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


def plot_heatmap(all_data, output_path):
    """
    Plots a heatmap where each bin's value is the weighted average
    of 'max_performance'. Bins with no data are shown as white.
    """
    plt.figure(figsize=(12, 10))

    # --- 1. Define Bins ---
    # Bins for delta (a)
    delta_bins = np.append(np.arange(0.3, 0.96, 0.05), 1.05)
    # Bins for alpha (b)
    alpha_bins = np.arange(0, 0.70, 0.05)

    # --- 2. Build Weighted Average Matrix ---
    # Calculate the sum of weights ('max_performance') in each bin
    heatmap, xedges, yedges = np.histogram2d(
        all_data['delta'],
        all_data['alpha'],
        bins=[delta_bins, alpha_bins],
        weights=all_data['max_performance']
    )

    # Calculate the count of items in each bin
    counts, _, _ = np.histogram2d(
        all_data['delta'],
        all_data['alpha'],
        bins=[delta_bins, alpha_bins]
    )

    # Calculate the weighted average: sum(weights) / count
    # Suppress divide-by-zero warnings, as 'where' handles it
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_avg = np.divide(heatmap, counts, where=counts != 0)

    # --- 3. Convert to DataFrame for Plotting ---
    df_heatmap = pd.DataFrame(
        heatmap_avg.T,  # Transpose to get delta on x-axis
        index=np.round(yedges[:-1], 2),
        columns=np.round(xedges[:-1], 2)
    )
    df_heatmap = df_heatmap.iloc[::-1]  # Reverse y-axis to put 0 at bottom

    # === Set empty or zero-value bins to NaN to display as white ===
    df_heatmap[df_heatmap < 1e-9] = np.nan

    # --- 4. Set colormap ---
    # Use a copy to avoid modifying the original
    cmap = sns.color_palette("YlGnBu", as_cmap=True).copy()
    cmap.set_bad(color='white')  # Make all NaN bins white

    # --- 5. Draw the heatmap ---
    ax = sns.heatmap(
        df_heatmap,
        cmap=cmap,
        cbar=True,  # Explicitly show colorbar
        vmin=0, vmax=1,
        linewidths=0.2,
        annot=True,  # Show numerical values
        fmt=".2f",  # Format values to 2 decimal places
        annot_kws={'size': 10, 'ha': 'center', 'va': 'center'}
    )

    # --- 6. Customize Axes and Labels ---
    xtick_labels = np.round(delta_bins, 2).astype(str)
    xtick_labels[xtick_labels == '1.05'] = '1.0'  # Fix the last label

    ax.set_xticks(np.arange(len(delta_bins)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

    ax.set_yticks(np.arange(len(alpha_bins)))
    ax.set_yticklabels(np.round(alpha_bins[::-1], 2))  # Match reversed index

    ax.tick_params(labelsize=14)
    ax.set_aspect('equal')  # Make bins square

    # Standardized axis labels
    ax.set_xlabel(r"$\delta$ (delta)", fontsize=16)
    ax.set_ylabel(r"$\alpha$ (alpha)", fontsize=16)
    ax.set_title("Heatmap of Max Performance (Weighted Average)", fontsize=18, pad=20)

    # Label the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Weighted Mean of Max Performance', rotation=270, labelpad=20, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, format='eps', dpi=600, bbox_inches='tight')
    plt.close()  # Close figure to free memory


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":

    # --- Configuration ---
    # Use relative paths for open-source compatibility
    DATA_DIR = r"YOUR_DATA_FOLDER_PATH"
    OUTPUT_DIR = r"./analysis_results"
    OUTPUT_FILENAME = "heatmap_max_performance_avg.eps"

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
            player_data = process_player(file_path)
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
            print(f"Generating heatmap... saving to {output_path}")
            plot_heatmap(valid_data, output_path)
            print("Heatmap generation complete.")