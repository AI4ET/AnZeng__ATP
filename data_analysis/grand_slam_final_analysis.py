import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from datetime import timedelta

'''
This program is designed to generate boxplots and analyze the distribution of
various parameters (delta, alpha) for winners and losers across all Grand Slam finals.
'''


def calculate_ab(df, target_date, window_days=182):
    """
    Calculates the 'a' (delta) and 'b' (alpha) parameters for a player
    based on their performance within a specified time window around a target date.
    """
    if df.empty:
        return np.nan, np.nan

    start_date = target_date - timedelta(days=window_days)
    end_date = target_date + timedelta(days=window_days)

    # Ensure the 'Date' column is of datetime type (already done in main)
    subset = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # If there are fewer than 3 matches in the window, calculation is invalid, return NaN
    if len(subset) < 3:
        return np.nan, np.nan

    try:
        min_p = subset['Performance Index'].min()
        max_p = subset['Performance Index'].max()

        # If max and min are equal, cannot solve
        if np.isclose(min_p, max_p):
            return np.nan, np.nan

        # Define the system of equations to solve
        def equations(vars):
            a, b = vars
            eq1 = a * (1 - b) - min_p
            eq2 = a + (1 - a) * b - max_p
            return [eq1, eq2]

        # Solve the equations
        a_initial = max(min_p + 1e-3, 0.1)  # Set a reasonable initial value
        a, b = fsolve(equations, (a_initial, 0.55))

        # Check the validity of the solution (changed a >= 1 to a > 1)
        if a <= min_p or b < 0 or a > 1 or b > 1:
            return np.nan, np.nan

        return round(a, 4), round(b, 4)
    except Exception:
        return np.nan, np.nan


def analyze_grand_slam_finals(data_directory):
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.xlsx')]
    if not all_files:
        print(f"Error: No .xlsx files found in directory '{data_directory}'.")
        return

    df_list = [pd.read_excel(file) for file in all_files]
    df_all_matches = pd.concat(df_list, ignore_index=True)

    print(
        f"Successfully loaded and consolidated {len(all_files)} files, with a total of {len(df_all_matches)} match records.")

    # Data Preprocessing
    required_cols = ['Date', 'Match Level', 'Ending Round', 'Result', 'Player Name', 'Opponent Name',
                     'Performance Index']
    missing_cols = [col for col in required_cols if col not in df_all_matches.columns]
    if missing_cols:
        print(f"Error: Missing required English columns: {missing_cols}")
        print("Please ensure your data files have the correct English headers.")
        return

    df_all_matches['Date'] = pd.to_datetime(df_all_matches['Date'])

    # Filter for all Grand Slam final winning records
    df_gs_finals_winners = df_all_matches[
        (df_all_matches['Match Level'] == 'GRAND SLAM') &
        (df_all_matches['Ending Round'] == 'G') &
        (df_all_matches['Result'] == 'W')
        ].copy()

    if df_gs_finals_winners.empty:
        print("Error: No Grand Slam final winning records found in the data.")
        return

    print(f"Found {len(df_gs_finals_winners)} Grand Slam finals.")

    #  Calculate a, b values for winners and losers in each final
    print("Calculating a, b metrics for finalists...")
    results = []
    for _, final_match in df_gs_finals_winners.iterrows():
        winner_name = final_match['Player Name']
        loser_name = final_match['Opponent Name']
        target_date = final_match['Date']

        # Get the complete match history for the winner and loser
        df_winner = df_all_matches[df_all_matches['Player Name'] == winner_name]
        df_loser = df_all_matches[df_all_matches['Player Name'] == loser_name]

        # Calculate metrics
        a_winner, b_winner = calculate_ab(df_winner, target_date)
        a_loser, b_loser = calculate_ab(df_loser, target_date)

        if not np.isnan(a_winner):
            results.append({'player_type': 'Winner', 'delta_value': a_winner, 'alpha_value': b_winner})
        if not np.isnan(a_loser):
            results.append({'player_type': 'Loser', 'delta_value': a_loser, 'alpha_value': b_loser})

    if not results:
        print("\nCalculation complete, but no valid metric values could be calculated.")
        print(
            "This is often because players have fewer than 3 match records within the 6-month window (before/after) the final date.")
        print("Please check if your dataset covers a sufficient time range.")
        return

    df_results = pd.DataFrame(results)
    print(f"Successfully calculated {len(df_results)} sets of valid metrics.")

    # Plotting boxplots using Seaborn (modified)
    print("Generating plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    # Define plot order
    plot_order = ['Winner', 'Loser']
    # Plot boxplot for delta (formerly 'a') values
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='player_type', y='delta_value', data=df_results, palette=['#66c2a5', '#fc8d62'], width=0.4,
                order=plot_order)
    plt.xlabel("Player Status in Grand Slam Final", fontsize=12)
    plt.ylabel(r"$\delta$", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("delta_values_boxplot.eps", format='eps', dpi=600)
    print("Plot 'delta_values_boxplot.eps' saved successfully.")
    # Plot boxplot for alpha (formerly 'b') values
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='player_type', y='alpha_value', data=df_results, palette=['#66c2a5', '#fc8d62'], width=0.4,
                order=plot_order)
    plt.xlabel("Player Status in Grand Slam Final", fontsize=12)
    plt.ylabel(r"$\alpha$", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig("alpha_values_boxplot.eps", format='eps', dpi=600)
    print("Plot 'alpha_values_boxplot.eps' saved successfully.")

if __name__ == '__main__':
    # PLEASE UPDATE THIS PATH with the actual folder containing your .xlsx files
    DATA_FOLDER = r"YOUR_DATA_FOLDER_PATH"

    # --- Execute Analysis ---
    if DATA_FOLDER == "YOUR_DATA_FOLDER_PATH" or not os.path.isdir(DATA_FOLDER):
        print("Error: Please update the 'DATA_FOLDER' variable with a valid directory path.")
    else:
        analyze_grand_slam_finals(DATA_FOLDER)