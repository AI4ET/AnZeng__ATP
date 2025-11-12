import pandas as pd
import math
import os
import re

"""
Tennis Match Data Preprocessing Script

This script reads raw tennis match data from Excel files, processes them,
and calculates a custom 'Total Points' and 'Performance Index' for each
player's run in a tournament.

It identifies the final match played by a player in each tournament
(grouping by Date and Tournament) and then applies a complex scoring logic
based on the player's final round, match result, and individual set scores.

This script assumes input Excel files already use English headers.
"""

# --- Points Tables and Constants ---
POINTS = {
    "ATP1000_G": 1000, "ATP1000_F": 650, "ATP1000_SF": 400,
    "ATP1000_QF": 200, "ATP1000_R16": 100, "ATP1000_R32": 50,
    "ATP1000_R64": 30, "ATP1000_R56": 30,
    "ATP1000_R96": 10, "ATP1000_R128": 10,
    "ATP1000_Q": 20, "ATP1000_Q3": 0, "ATP1000_Q2": 10, "ATP1000_Q1": 0,
    "GRAND SLAM_G": 2000, "GRAND SLAM_F": 1300, "GRAND SLAM_SF": 800,
    "GRAND SLAM_QF": 400, "GRAND SLAM_R16": 200, "GRAND SLAM_R32": 100,
    "GRAND SLAM_R64": 50, "GRAND SLAM_R128": 10,
    "GRAND SLAM_Q": 30, "GRAND SLAM_Q3": 16, "GRAND SLAM_Q2": 8, "GRAND SLAM_Q1": 0
}

BASE_POINTS_MAP = {
    "ATP1000_F": POINTS["ATP1000_SF"], "ATP1000_SF": POINTS["ATP1000_QF"],
    "ATP1000_QF": POINTS["ATP1000_R16"], "ATP1000_R16": POINTS["ATP1000_R32"],
    "ATP1000_R32": POINTS["ATP1000_R64"], "ATP1000_R64": POINTS["ATP1000_R128"],
    "ATP1000_R56": POINTS["ATP1000_R96"],
    "GRAND SLAM_F": POINTS["GRAND SLAM_SF"], "GRAND SLAM_SF": POINTS["GRAND SLAM_QF"],
    "GRAND SLAM_QF": POINTS["GRAND SLAM_R16"], "GRAND SLAM_R16": POINTS["GRAND SLAM_R32"],
    "GRAND SLAM_R32": POINTS["GRAND SLAM_R64"], "GRAND SLAM_R64": POINTS["GRAND SLAM_R128"],
    "GRAND SLAM_R128": 0
}

CHAMPION_POINTS = {
    "ATP1000": POINTS["ATP1000_G"],
    "GRAND SLAM": POINTS["GRAND SLAM_G"]
}

# Assumes input Excel files use these English headers for set scores
SET_COLS = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5']

# Defines the next round after a win. 'G' stands for Champion (Gagnant).
ROUND_PROGRESSION = {
    "R128": "R64",
    "R96": "R64",
    "R64": "R32",
    "R56": "R32",
    "R32": "R16",
    "R16": "QF",
    "QF": "SF",
    "SF": "F",
    "F": "G"
}


# --- Helper Functions ---

def parse_score(score_str):
    """Parses a set score string 'p:o' into player games and opponent games."""
    if pd.isna(score_str):
        return None, None
    match = re.match(r'(\d+):(\d+)', str(score_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def calculate_scores(final_match):
    """
    Calculates the total points and performance index for a player's
    final match in a tournament.
    """
    # Assumes English columns: 'Match Level', 'Round', 'Result'
    level = final_match['Match Level']
    round_name = final_match['Round']
    result = final_match['Result']

    # Special case for the champion
    if round_name == 'F' and result == 'W':
        score = CHAMPION_POINTS.get(level, 0)
        return score, 1.0

    # --- Core Logic ---
    # 1. Base points (y1) are the full points for the current round
    current_round_key = f"{level}_{round_name}"
    y1 = POINTS.get(current_round_key, 0)

    # 2. Points upper bound (y2) is the next round's points
    next_round_name = ROUND_PROGRESSION.get(round_name)
    y2 = y1  # Default to current points
    if next_round_name:
        if next_round_name == 'G':  # If next round is winning the final
            y2 = CHAMPION_POINTS.get(level, y1)
        else:
            next_round_key = f"{level}_{next_round_name}"
            y2 = POINTS.get(next_round_key, y1)

    # 3. 'diff' is the potential points gain
    diff = y2 - y1

    bonus = 0
    # Assumes English columns 'Set 1', 'Set 2', etc.
    played_sets = sum(1 for col in SET_COLS if pd.notna(final_match[col]))

    # Calculate bonus points based on set scores
    if diff > 0 and played_sets > 0:
        per_set = diff / played_sets
        for col in SET_COLS:
            pg, og = parse_score(final_match[col])
            if pg is not None and og is not None:
                max_games = max(pg, og)
                if max_games > 0:
                    # Even a loss with a close score gets partial bonus
                    bonus += per_set if pg > og else (pg / max_games) * per_set

    # 4. Total score = base points for the round + performance bonus
    total_score = y1 + bonus

    champ_points = CHAMPION_POINTS.get(level, 1)
    perf_idx = 0
    if total_score > 0 and champ_points > 1:
        try:
            perf_idx = math.log(total_score) / math.log(champ_points)
        except (ValueError, ZeroDivisionError):
            perf_idx = 0

    return total_score, perf_idx


# --- Core Processing Function ---
def process_tennis_excel(input_path: str, output_path: str) -> None:
    """
    Reads a single raw Excel file, processes its matches, and saves
    the summarized tournament results to a new file.
    """
    try:
        df = pd.read_excel(input_path)
    except Exception as e:
        print(f"Error: Could not read file {os.path.basename(input_path)}. Reason: {e}")
        return

    df.columns = df.columns.str.strip()

    # Assumes English columns: 'Match Level', 'Date', 'Tournament', 'Round', 'Result'
    required_cols = ['Match Level', 'Date', 'Tournament', 'Round', 'Result']
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping: {os.path.basename(input_path)} is missing required columns.")
        return

    df_filtered = df[df['Match Level'].isin(['ATP1000', 'GRAND SLAM'])].copy()
    if df_filtered.empty:
        return

    df_filtered.loc[:, "Date"] = pd.to_datetime(df_filtered["Date"])
    result_rows = []

    # Group by tournament (identified by Date and Name)
    for _, group in df_filtered.groupby(["Date", "Tournament"], sort=False):
        group = group.reset_index(drop=True)  # Sorts by match order (implicit)

        # 'group.loc[0]' is the *last* match played in the tournament
        # Check if the last match was a qualifier
        if group.loc[0, "Round"] in ["Q1", "Q2", "Q3"]:
            continue

        last_match = group.loc[0].copy()
        # 'group.loc[len(group) - 1]' is the *first* match played
        # Assumes output column 'Starting Round'
        last_match["Starting Round"] = group.loc[len(group) - 1, "Round"]
        result_rows.append(last_match)

    if not result_rows:
        return

    result_df = pd.DataFrame(result_rows)

    # --- Scoring Logic ---
    # 1. Create a temporary 'Scoring_Round'
    #    If Won, scoring round = next round. If Lost, scoring round = current round.
    def get_scoring_round(row):
        last_played_round = row['Round']
        result = row['Result']
        if result == 'W' and last_played_round != 'F':
            return ROUND_PROGRESSION.get(last_played_round, last_played_round)
        return last_played_round

    # 2. Create the final 'Ending Round' for display
    #    Same as scoring round, but shows 'G' for a final win.
    def get_termination_round(row):
        last_played_round = row['Round']
        result = row['Result']
        if result == 'W':
            return ROUND_PROGRESSION.get(last_played_round, last_played_round)
        return last_played_round

    result_df['Scoring_Round_Temp'] = result_df.apply(get_scoring_round, axis=1)
    result_df['Ending Round'] = result_df.apply(get_termination_round, axis=1)

    # 3. Use the temp 'Scoring_Round_Temp' to calculate scores
    calc_df = result_df.copy()
    # Temporarily overwrite 'Round' for the calculation function
    calc_df['Round'] = calc_df['Scoring_Round_Temp']

    scores = calc_df.apply(calculate_scores, axis=1, result_type='expand')
    # Assumes output columns 'Total Points' and 'Performance Index'
    result_df[['Total Points', 'Performance Index']] = scores

    # 4. Clean up and reorder columns
    result_df = result_df.drop(columns=['Scoring_Round_Temp'])
    result_df['Date'] = result_df['Date'].dt.strftime('%Y-%m-%d')
    all_cols = result_df.columns.tolist()

    # Assumes English columns 'Round', 'Starting Round', 'Ending Round', 'Date of Birth'
    cols_to_move = ['Round', 'Starting Round', 'Ending Round', 'Date of Birth']
    base_cols = [col for col in all_cols if col not in cols_to_move]

    try:
        time_index = base_cols.index('Date')
        base_cols.insert(time_index + 1, 'Starting Round')
        base_cols.insert(time_index + 2, 'Ending Round')
    except ValueError:
        base_cols.extend(['Starting Round', 'Ending Round'])

    # Assumes English columns 'Player Name' and 'Date of Birth'
    if 'Date of Birth' in all_cols:
        try:
            name_index = base_cols.index('Player Name')
            base_cols.insert(name_index + 1, 'Date of Birth')
        except ValueError:
            base_cols.insert(0, 'Date of Birth')

    final_df = result_df[base_cols]

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_excel(output_path, index=False)
        print(f"  -> Processed and saved to: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"  -> Error: Failed to save file {output_path}. Reason: {e}")


# --- Batch Processing Function ---
def batch_process_directory(input_root_dir: str, output_root_dir: str) -> None:
    """
    Processes all .xlsx files in the input directory, handling
    duplicates by keeping the one with the larger file size.
    """
    processed_files = {}
    print(f"--- Starting Batch Process ---")
    print(f"Input Directory: {input_root_dir}")
    print(f"Output Directory: {output_root_dir}")

    for dirpath, _, filenames in os.walk(input_root_dir):
        for filename in filenames:
            if not filename.endswith('.xlsx'):
                continue

            input_path = os.path.join(dirpath, filename)
            # Remove leading numbers (e.g., '1. Roger Federer.xlsx' -> 'Roger Federer.xlsx')
            new_filename = re.sub(r'^\d+\s+', '', filename)
            output_path = os.path.join(output_root_dir, new_filename)

            if output_path in processed_files:
                existing_source_path = processed_files[output_path]
                current_file_size = os.path.getsize(input_path)
                existing_file_size = os.path.getsize(existing_source_path)

                if current_file_size <= existing_file_size:
                    print(f"\nSkipping (Smaller File): {filename}")
                    print(
                        f"  - Reason: A file generated from a larger source '{os.path.basename(existing_source_path)}' already exists.")
                    continue
                else:
                    print(f"\nReplacing (Larger File): {filename}")
                    print(
                        f"  - Reason: This file is larger than '{os.path.basename(existing_source_path)}', overwriting.")

            processed_files[output_path] = input_path

    print("\n--- Starting File Processing and Generation ---")
    if not processed_files:
        print("No new Excel files to process.")
        return

    for output_path, input_path in processed_files.items():
        print(f"Processing: {os.path.basename(input_path)}")
        process_tennis_excel(input_path, output_path)

    print("\n--- Batch Process Complete ---")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # Please update these relative paths for your project structure

    # Directory containing raw, unprocessed .xlsx files
    INPUT_DATA_ROOT = r"./data/raw_match_data"

    # Directory to save the processed .xlsx files
    OUTPUT_DATA_ROOT = r"./data/processed_player_data"

    # --- Run ---
    if not os.path.isdir(INPUT_DATA_ROOT):
        print(f"Error: Input directory not found: {INPUT_DATA_ROOT}")
        print("Please update the 'INPUT_DATA_ROOT' variable.")
    else:
        batch_process_directory(INPUT_DATA_ROOT, OUTPUT_DATA_ROOT)