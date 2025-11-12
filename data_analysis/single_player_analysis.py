"""
Player Performance Visualization Script

This script contains a function to generate a detailed visualization
of a single player's performance over time from their match history.

The main function `plot_player_performance` generates a plot that includes:
1.  Scatter points for the 'Performance Index' of each match.
2.  Vertical lines indicating the *maximum* performance within a given
    grouping period (either month or quarter).
3.  A horizontal dashed line indicating the player's overall *mean* performance.

This script is designed to be run from a terminal or as part of a larger
analysis pipeline, saving the resulting plot directly to a file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Use 'Agg' backend to save figures to disk without requiring a display
matplotlib.use("Agg")


def plot_player_performance(input_excel_path, output_image_path, grouping_period='m'):
    """
    Generates and saves a scatter plot of performance index over time,
    grouped by quarter or month, with vertical max lines.

    ASSUMPTIONS:
    - Input Excel file has a 'Date' column (datetime-compatible).
    - Input Excel file has a 'Performance Index' column (numeric).

    :param input_excel_path: Path to the player's .xlsx data file.
    :param output_image_path: Path to save the output image (e.g., "player.eps").
    :param grouping_period: 'q' for quarterly grouping, 'm' for monthly grouping.
    """
    try:
        # Read and sort data by date
        df = pd.read_excel(input_excel_path)
        # --- Assumes English column 'Date' ---
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')

        # ---- Handle Grouping ----
        if grouping_period == 'q':  # By Quarter
            df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
            grouped = df.groupby('Quarter')

            # Generate a complete timeline of quarters (as strings)
            min_date = df['Date'].min().to_period('Q')
            max_date = df['Date'].max().to_period('Q')
            quarters = pd.period_range(start=min_date, end=max_date, freq='Q').astype(str)
            x_positions = list(range(len(quarters)))  # For x-axis range

        elif grouping_period == 'm':  # By Month
            # Use mid-month as the representative timestamp for grouping
            df['month_date'] = df['Date'].dt.to_period('M').dt.to_timestamp() + pd.DateOffset(days=14)
            grouped = df.groupby('month_date')
            quarters = sorted(grouped.groups.keys())  # 'quarters' here is a list of Timestamps
            x_positions = quarters

        else:
            raise ValueError("grouping_period must be 'q' (quarter) or 'm' (month)")

        # ---- Plotting Setup ----
        plt.rcParams['figure.dpi'] = 800
        plt.figure(figsize=(12, 6))

        # --- Assumes English column 'Performance Index' ---
        y_min = float(df['Performance Index'].min())
        y_max = float(df['Performance Index'].max())
        span = y_max - y_min

        if span == 0:
            # Handle case where all values are the same
            margin = max(1.0, 0.1 * abs(y_max) if y_max != 0 else 1.0)
        else:
            margin = 0.05 * span

        axis_ymin = y_min - margin
        axis_ymax = y_max + margin
        plt.ylim(axis_ymin, axis_ymax)

        # Color settings
        line_color = '#A9A9A9'  # Gray color for lines
        vertical_alpha = 1.0
        horiz_alpha = 1.0

        # 🔹 Plot vertical lines first (low zorder)
        for i, q in enumerate(quarters):
            if q in grouped.groups:
                group = grouped.get_group(q)
                # --- Assumes English column 'Performance Index' ---
                max_integral = float(group['Performance Index'].max())

                # Calculate normalized height (0-1) based on actual axis limits
                denom = (axis_ymax - axis_ymin)
                if denom == 0:
                    ymax_normalized = 1.0
                else:
                    ymax_normalized = (max_integral - axis_ymin) / denom

                ymax_normalized = max(0.0, min(1.0, ymax_normalized))

                # X-value: index for quarter, Timestamp for month
                x_val = i if grouping_period == 'q' else q

                plt.axvline(
                    x=x_val,
                    ymin=0.0, ymax=ymax_normalized,
                    color=line_color,
                    linestyle='-',
                    linewidth=1,
                    alpha=vertical_alpha,
                    zorder=1
                )

        # 🔹 Plot scatter points (higher zorder)
        for i, q in enumerate(quarters):
            if q in grouped.groups:
                group = grouped.get_group(q)
                x_val = i if grouping_period == 'q' else q
                plt.scatter(
                    [x_val] * len(group),
                    group['Performance Index'],  # --- Assumes 'Performance Index' ---
                    color='#b0262b',
                    edgecolor='#646463',
                    alpha=1,
                    s=50,
                    label='Game' if i == 0 else "",
                    zorder=2
                )

        # Mean performance line (dashed horizontal)
        # --- Assumes English column 'Performance Index' ---
        mean_performance = df['Performance Index'].mean()
        plt.axhline(
            y=mean_performance,
            color=line_color,
            linestyle='--',
            linewidth=1.5,
            alpha=horiz_alpha,
            zorder=1
        )

        # ---- Axis Labels ----
        plt.xlabel('Year', fontsize=20)
        plt.ylabel(r"$c_{it}$", fontsize=24)  # Using $c_{it}$ as per your code
        plt.grid(False)

        # ---- X-axis Ticks ----
        if grouping_period == 'q':
            # Show one tick per year (using the start of the quarter string)
            year_labels = [q[:4] for q in quarters]
            year_indices, last_year = [], None
            for i, year in enumerate(year_labels):
                if year != last_year:
                    year_indices.append(i)
                    last_year = year

            plt.xticks(year_indices, [year_labels[i] for i in year_indices],
                       rotation=45, fontsize=18)
            # Adjust x-axis limits for discrete indices
            plt.xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)

        elif grouping_period == 'm':
            # Select ticks by year (one every two years)
            min_year, max_year = quarters[0].year, quarters[-1].year
            tick_dates, tick_labels = [], []
            for year in range(min_year, max_year + 1):
                if (year + 1 - min_year) % 2 == 0:
                    tick_dates.append(pd.Timestamp(str(year) + '-01-15'))
                    tick_labels.append(str(year))

            plt.xticks(tick_dates, tick_labels, rotation=45, fontsize=18)

            # Adjust x-axis limits for time data
            time_range_days = (max(x_positions) - min(x_positions)).days
            space_days = max(1, int(time_range_days * 0.02))
            plt.xlim(min(x_positions) - pd.Timedelta(days=space_days),
                     max(x_positions) + pd.Timedelta(days=space_days))

        # ---- Y-axis Ticks Fontsize ----
        plt.yticks(fontsize=20)

        # ---- Legend & Spines ----
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_image_path, format='eps', dpi=600, bbox_inches='tight')
        plt.close()  # Release memory

    except FileNotFoundError:
        print(f"Error: File not found at {input_excel_path}. Please check the path.")
    except KeyError as e:
        print(f"Error: Missing required column {e}. Script expects 'Date' and 'Performance Index'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Example Call ---
    # This block demonstrates how to use the function in a project.
    # Replace paths with your project's structure (e.g., using relative paths).

    # --- Configuration ---
    # The player's name or ID, used to build file paths
    PLAYER_NAME = "Stan Wawrinka"
    DATA_DIR = r"DATASET_PATH"
    OUTPUT_DIR = r"OUT_PATh"

    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        except Exception as e:
            print(f"Warning: Could not create output directory {OUTPUT_DIR}. {e}")

    # --- File Paths ---
    input_file = os.path.join(DATA_DIR, f"{PLAYER_NAME}.xlsx")
    output_file = os.path.join(OUTPUT_DIR, f"{PLAYER_NAME}_performance.eps")

    # --- Execute ---
    print(f"Generating plot for {PLAYER_NAME}...")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")

    # Check if input file exists before running
    if os.path.exists(input_file):
        plot_player_performance(
            input_excel_path=input_file,
            output_image_path=output_file,
            grouping_period='m'  # Use 'm' for monthly, 'q' for quarterly
        )
        print("Plot generation complete.")
    else:
        print(f"Error: Input file not found at {input_file}")
        print("Please check your DATA_DIR and PLAYER_NAME variables.")