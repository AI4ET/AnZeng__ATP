import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import make_interp_spline
from matplotlib.lines import Line2D
import re

plt.rcParams["axes.unicode_minus"] = False  # Ensure minus sign displays correctly


def calculate_ab(subset):
    min_p = subset['Performance Index'].min()
    max_p = subset['Performance Index'].max()
    def equations(vars):
        a, b = vars
        return [
            a * (1 - b) - min_p,
            a + (1 - a) * b - max_p
        ]

    try:
        a_sol, b_sol = fsolve(equations, (0.5, 0.1), xtol=1e-5)
        if not all(np.isfinite([a_sol, b_sol])) or a_sol < 0 or b_sol < 0 or a_sol > 1 or b_sol > 1:
            return np.nan, np.nan
        return a_sol, b_sol
    except Exception:
        return np.nan, np.nan


def process_players_by_halfyear_group(folder_path, output_prefix, window_size=3):
    records = []

    for file in os.listdir(folder_path):
        if not file.endswith('.xlsx'):
            continue

        try:
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            df['date'] = pd.to_datetime(df['Date']).dt.date
            df = df.sort_values('date').reset_index(drop=True)

            if len(df) < window_size:
                continue

            player_id = os.path.splitext(file)[0]
            career_start = df['date'].min()
            career_end = df['date'].max()
            career_span_years = (career_end - career_start).days / 365.0

            for i in range(len(df) - window_size + 1):
                window = df.iloc[i:i + window_size]
                window_start = window['date'].min()
                window_end = window['date'].max()
                mid_point = window_start + (window_end - window_start) / 2
                years_since_debut = (mid_point - career_start).days / 365.0

                a, b = calculate_ab(window)
                if not np.isnan(a) and not np.isnan(b):
                    records.append({
                        'year': years_since_debut,
                        'delta': a,
                        'alpha': b,
                        'career_span': career_span_years,
                        'player_id': player_id
                    })

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    data = pd.DataFrame(records).dropna()
    if data.empty:
        print("No valid data to process after calculation.")
        return

    # Define career span groups
    career_bins = [0, 10, 15, 20, np.inf]
    career_labels = ['0-10 years', '10-15 years', '15-20 years', '20+ years']  # Made labels clearer
    data['career_group'] = pd.cut(data['career_span'], bins=career_bins, labels=career_labels, right=False)

    colors = ['#F49568', '#77DCDD', '#82C61E', '#AA66EB']
    markers = ['o', 's', 'D', '^']
    sp_color = '#8B0000'  # maroon
    ep_color = '#006400'  # dark green

    print("Generating 2x2 subplot figure...")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.text(0.5, 0.04, r"Average $\delta$", ha='center', va='center', fontsize=16)
    fig.text(0.06, 0.5, r"Average $\alpha$", ha='center', va='center', rotation='vertical', fontsize=16)

    for idx, group in enumerate(career_labels):
        ax = axes[idx]
        subset = data[data['career_group'] == group]

        ax.set_title(f"Career Span: {group}", fontsize=14)

        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            # Bin by half-year
            bin_width = 0.5
            max_year = subset['year'].max()
            bins = np.arange(0, max_year + bin_width, bin_width)
            # Use 'right=True' to include the lowest values
            subset['year_bin'] = pd.cut(subset['year'], bins=bins, include_lowest=True, right=True)

            grouped = subset.groupby('year_bin').agg(
                mean_year=('year', 'mean'),
                mean_delta=('delta', 'mean'),  # Use new name
                mean_alpha=('alpha', 'mean')  # Use new name
            ).reset_index().sort_values('mean_year')

            x_points = grouped['mean_delta'].values
            y_points = grouped['mean_alpha'].values

            # Scatter points
            ax.scatter(x_points, y_points, color=colors[idx], marker=markers[idx], s=40, alpha=0.8)

            # Smoothing
            if len(x_points) >= 6:
                try:
                    spline = make_interp_spline(range(len(x_points)), np.column_stack((x_points, y_points)), k=3)
                    t_smooth = np.linspace(0, len(x_points) - 1, 300)
                    smooth_points = spline(t_smooth)
                    x_smooth, y_smooth = smooth_points[:, 0], smooth_points[:, 1]
                    ax.plot(x_smooth, y_smooth, color=colors[idx], linewidth=2, linestyle='-')
                except Exception:
                    ax.plot(x_points, y_points, color=colors[idx], linewidth=2, linestyle='--')
            elif len(x_points) > 1:
                ax.plot(x_points, y_points, color=colors[idx], linewidth=2, linestyle='--')

            # Mark Start Point (SP) and End Point (EP)
            if len(x_points) > 0:
                ax.scatter(x_points[0], y_points[0], color=sp_color, s=150, zorder=5,
                           edgecolor='white', linewidth=1.5, marker='>')
                ax.scatter(x_points[-1], y_points[-1], color=ep_color, s=250, zorder=5,
                           edgecolor='white', linewidth=1.5, marker='*')

            # --- MODIFICATION 2: Standardized Legend ---
            legend_elements = [
                Line2D([0], [1], color=colors[idx], marker=markers[idx], linestyle='-',
                       linewidth=2, markersize=8, label=group),
                Line2D([0], [0], marker='>', color='w', markerfacecolor=sp_color, markersize=10,
                       linestyle='', markeredgecolor='white', markeredgewidth=1.5,
                       label='SP'),  # Simplified label
                Line2D([0], [0], marker='*', color='w', markerfacecolor=ep_color, markersize=12,
                       linestyle='', markeredgecolor='white', markeredgewidth=1.5,
                       label='EP')  # Simplified label
            ]
            ax.legend(handles=legend_elements, fontsize=10, frameon=True, fancybox=True, shadow=False, loc='best')

        # Fixed axis limits and grid
        ax.set_xlim(0.4, 0.9)
        ax.set_ylim(0, 0.3)
        ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    fig_filename = f"{output_prefix}_career_groups_subplot.eps"
    plt.savefig(fig_filename, format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure
    print(f"Subplot figure saved to: {fig_filename}")

    print("Generating combined group plot...")
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    legend_elements_all = []

    for idx, group in enumerate(career_labels):
        subset = data[data['career_group'] == group]
        if subset.empty:
            continue

        bin_width = 0.5
        max_year = subset['year'].max()
        bins = np.arange(0, max_year + bin_width, bin_width)
        subset['year_bin'] = pd.cut(subset['year'], bins=bins, include_lowest=True, right=True)

        grouped = subset.groupby('year_bin').agg(
            mean_year=('year', 'mean'),
            mean_delta=('delta', 'mean'),
            mean_alpha=('alpha', 'mean')
        ).reset_index().sort_values('mean_year')

        x_points = grouped['mean_delta'].values
        y_points = grouped['mean_alpha'].values

        plt.scatter(x_points, y_points, color=colors[idx], marker=markers[idx], s=40, alpha=0.8)

        if len(x_points) >= 6:
            try:
                spline = make_interp_spline(range(len(x_points)), np.column_stack((x_points, y_points)), k=3)
                t_smooth = np.linspace(0, len(x_points) - 1, 300)
                smooth_points = spline(t_smooth)
                x_smooth, y_smooth = smooth_points[:, 0], smooth_points[:, 1]
                plt.plot(x_smooth, y_smooth, color=colors[idx], linewidth=2, linestyle='-')
            except Exception:
                plt.plot(x_points, y_points, color=colors[idx], linewidth=2, linestyle='--')
        elif len(x_points) > 1:
            plt.plot(x_points, y_points, color=colors[idx], linewidth=2, linestyle='--')

        if len(x_points) > 0:
            plt.scatter(x_points[0], y_points[0], color=sp_color, s=150, zorder=5,
                        edgecolor='white', linewidth=1.5, marker='>')
            plt.scatter(x_points[-1], y_points[-1], color=ep_color, s=250, zorder=5,
                        edgecolor='white', linewidth=1.5, marker='*')

        legend_elements_all.append(
            Line2D([0], [1], color=colors[idx], marker=markers[idx], linestyle='-',
                   linewidth=2, markersize=8, label=group)
        )

    # Add SP and EP to legend
    legend_elements_all.extend([
        Line2D([0], [0], marker='>', color='w', markerfacecolor=sp_color, markersize=12,
               linestyle='', markeredgecolor='white', markeredgewidth=1.5,
               label='SP'),  # Simplified
        Line2D([0], [0], marker='*', color='w', markerfacecolor=ep_color, markersize=18,
               linestyle='', markeredgecolor='white', markeredgewidth=1.5,
               label='EP')  # Simplified
    ])

    ax.set_xlim(0.4, 0.9)
    ax.set_ylim(0, 0.3)
    plt.xlabel(r"Average $\delta$", fontsize=12)
    plt.ylabel(r"Average $\alpha$", fontsize=12)
    plt.title("Career Trajectories by Career Span", fontsize=14)
    plt.legend(handles=legend_elements_all, fontsize=10, frameon=True, fancybox=True,
               shadow=False, loc='best', title="Career Span Group", title_fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    fig_filename_combined = f"{output_prefix}_combined.eps"
    plt.savefig(fig_filename_combined, format='eps', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to: {fig_filename_combined}")
    print("Generating overall trajectory plot...")
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    all_data = data.copy()

    bin_width = 0.5
    max_year = all_data['year'].max()
    bins = np.arange(0, max_year + bin_width, bin_width)
    all_data['year_bin'] = pd.cut(all_data['year'], bins=bins, include_lowest=True, right=True)

    grouped_all = all_data.groupby('year_bin').agg(
        mean_year=('year', 'mean'),
        mean_delta=('delta', 'mean'),
        mean_alpha=('alpha', 'mean')
    ).reset_index().sort_values('mean_year')

    x_points = grouped_all['mean_delta'].values
    y_points = grouped_all['mean_alpha'].values

    overall_color = '#4A6FA5'
    plt.scatter(x_points, y_points, color=overall_color, marker='o', s=40, alpha=0.8)

    if len(x_points) >= 6:
        try:
            spline = make_interp_spline(range(len(x_points)), np.column_stack((x_points, y_points)), k=3)
            t_smooth = np.linspace(0, len(x_points) - 1, 300)
            smooth_points = spline(t_smooth)
            x_smooth, y_smooth = smooth_points[:, 0], smooth_points[:, 1]
            plt.plot(x_smooth, y_smooth, color=overall_color, linewidth=2, linestyle='-')
        except Exception:
            plt.plot(x_points, y_points, color=overall_color, linewidth=2, linestyle='--')
    elif len(x_points) > 1:
        plt.plot(x_points, y_points, color=overall_color, linewidth=2, linestyle='--')

    if len(x_points) > 0:
        plt.scatter(x_points[0], y_points[0], color=sp_color, s=150, zorder=5,
                    edgecolor='white', linewidth=1.5, marker='>')
        plt.scatter(x_points[-1], y_points[-1], color=ep_color, s=250, zorder=5,
                    edgecolor='white', linewidth=1.5, marker='*')

    legend_elements_overall = [
        Line2D([0], [1], color=overall_color, marker='o', linestyle='-',
               linewidth=2, markersize=8, label='All Players'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor=sp_color, markersize=12,
               linestyle='', markeredgecolor='white', markeredgewidth=1.5,
               label='SP'),  # Simplified
        Line2D([0], [0], marker='*', color='w', markerfacecolor=ep_color, markersize=18,
               linestyle='', markeredgecolor='white', markeredgewidth=1.5,
               label='EP')  # Simplified
    ]
    plt.legend(handles=legend_elements_overall, fontsize=10, frameon=True, fancybox=True, shadow=False, loc='best')

    ax.set_xlim(0.4, 0.9)
    ax.set_ylim(0, 0.3)
    # --- MODIFICATION 2: Standardized Axes/Title ---
    plt.xlabel(r"Average $\delta$", fontsize=12)
    plt.ylabel(r"Average $\alpha$", fontsize=12)
    plt.title("Overall Career Trajectory (All Players)", fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    fig_filename_overall = f"{output_prefix}_overall.eps"
    plt.savefig(fig_filename_overall, format='eps', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Overall plot saved to: {fig_filename_overall}")

    print("\n--- All plotting tasks complete. ---")

if __name__ == '__main__':

    INPUT_DATA_DIRECTORY = r"E:\大学学学学\科研\XY课题组\X002-tennis\Code\Get&Process Data\WholeDataset_Processed"
    OUTPUT_FILE_PREFIX = r"./analysis_results/career_trajectory"
        # Execute
    process_players_by_halfyear_group(
            folder_path=INPUT_DATA_DIRECTORY,
            output_prefix=OUTPUT_FILE_PREFIX,
            window_size=3
        )