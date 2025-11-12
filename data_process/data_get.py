"""
ATP Player Match History Scraper

This script uses DrissionPage to scrape player data from the official
ATP Tour website. It reads a master list of players, then iterates
through each player to scrape their birthdate and full match history.

Each player's data is saved as a separate .xlsx file with standardized
English headers, suitable for use in downstream analysis scripts.
"""

import openpyxl
from DrissionPage import Chromium, ChromiumOptions
import pandas as pd
import os
import time

# --- Global Constants ---

# Mapping of ATP event codes to standardized names
EVENT_TYPE_MAP = {
    '250': 'ATP250', 'CH': 'Challenger Tour', '500': 'ATP500',
    '1000': 'ATP1000', 'FU': 'ITF', 'WS': 'ATP Tour',
    'CS': 'ATP Tour', 'DC': 'ITF', 'GS': 'GRAND SLAM',
    'WC': 'ATP FINALS', 'UC': 'UNITED CUP', 'OL': 'ITF',
}

# Standardized English headers for the output Excel file
# These match the headers required by the processing scripts.
OUTPUT_HEADERS = [
    'Player Name', 'Player Ranking', 'Match Level', 'Tournament', 'Surface Type', 'Date',
    'Round', 'Result', 'Opponent Name', 'Opponent Ranking',
    'Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Date of Birth'
]


def read_master_list(file_path):
    """
    Reads the master Excel file.
    Assumes file has [Rank, PlayerCode, PlayerName] in the first 3 columns.

    :param file_path: Path to the master .xlsx file.
    :return: A list of [rank, code, name] lists, or None if file not found.
    """
    try:
        df = pd.read_excel(file_path)
        player_list = []
        # Read by position to avoid dependency on header names
        for _, row in df.iterrows():
            player_list.append([row[0], row[1], row[2]])
        print(f"✅ Successfully read master list. Total players: {len(player_list)}.")
        return player_list
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading master list: {e}")
        return None


def scrape_player_data(tab, player_name, player_code, output_directory):
    """
    Scrapes a single player's birthdate and full match history,
    then saves it to an Excel file.

    :param tab: The DrissionPage tab object to use.
    :param player_name: Player's full name (for saving).
    :param player_code: Player's ATP code (for API URL).
    :param output_directory: The directory to save the resulting .xlsx file.
    """

    # --- 1. Get Player Birthdate ---
    birthdate = ''
    try:
        birthdate_url = f'https://www.atptour.com/en/-/www/players/hero/{player_code}?v=1'
        tab.get(birthdate_url)
        hero_data = tab.json
        birthdate_raw = hero_data.get('BirthDate', '')
        if birthdate_raw:
            birthdate = birthdate_raw[:10]  # Get 'YYYY-MM-DD'
    except Exception as e:
        print(f"Warning: Failed to retrieve birthdate for {player_name}. Error: {e}")

    # --- 2. Get Match Data ---
    match_api_url = f'https://www.atptour.com/en/-/www/activity/sgl/{player_code}/?v=1'
    try:
        print(f"Fetching match data for {player_name}...")
        # Use retry and interval for robustness
        tab.get(match_api_url, retry=5, interval=3)
        match_data = tab.json
    except Exception as e:
        print(f"Error: Failed to get match data for {player_name} from {match_api_url}. Error: {e}")
        return

    # --- 3. Write to Excel ---
    work = openpyxl.Workbook()
    sheet = work.active
    sheet.append(OUTPUT_HEADERS)

    data_list = match_data.get('Activity', [])
    if not data_list:
        print(f"No activity data found for {player_name}.")

    for activity in data_list:
        for tournament in activity.get('Tournaments', []):
            event_date = tournament.get('EventDate', '')[:10]
            event_type_code = tournament.get('EventType', '')
            event_type = EVENT_TYPE_MAP.get(event_type_code, event_type_code)  # Use code if not in map
            surface = tournament.get('Surface', '')
            event_name = tournament.get('EventName', '')
            # This is the player's rank *at the time of the tournament*
            player_rank_at_event = tournament.get('PlayerRank', '')

            for match in tournament.get('Matches', []):
                try:
                    round_info = match['Round']['ShortName']
                    win_loss = match.get('WinLoss', '')
                    opponent_name = f"{match.get('OpponentFirstName', '')} {match.get('OpponentLastName', '')}".strip()
                    opponent_rank = match.get('OpponentRank', '')

                    sets = []
                    for i in range(1, 6):
                        p = match.get(f'Set{i}Player', -1)
                        o = match.get(f'Set{i}Opponent', -1)
                        if p != -1 and o != -1:
                            # CRITICAL FIX: Save as 'p:o' to be compatible
                            # with the downstream parse_score() function.
                            sets.append(f"{p}:{o}")
                        else:
                            sets.append(None)  # Use None for empty cells

                    sheet.append([
                        player_name, player_rank_at_event, event_type, event_name, surface, event_date,
                        round_info, win_loss, opponent_name, opponent_rank,
                        *sets,
                        birthdate
                    ])
                except Exception as e:
                    print(f"Error processing a single match for {player_name}: {e}")
                    continue

    # --- 4. Save File ---
    # Ensure file name is safe for file systems
    safe_filename = "".join(c for c in player_name if c.isalnum() or c in (' ', '-')).rstrip()
    output_path = os.path.join(output_directory, f'{safe_filename}.xlsx')

    try:
        work.save(output_path)
        print(f"File saved: {output_path}")
    except Exception as e:
        print(f"Error saving file for {player_name}: {e}")


def main():
    """Main execution function."""

    # --- Configuration ---
    # !! IMPORTANT: Update these paths for your project structure !!

    # Optional: Path to your Chrome executable.
    # If Chrome/Chromium is in your system PATH, you can comment this out.
    BROWSER_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

    # Path to the master list of players
    MASTER_LIST_FILE = r"./data/master_player_list.xlsx"

    # Directory where individual player .xlsx files will be saved
    OUTPUT_DIR = r"./data/scraped_player_data"

    # --- End Configuration ---

    # Setup browser options
    co = ChromiumOptions()
    if os.path.exists(BROWSER_PATH):
        co.set_browser_path(BROWSER_PATH)
    else:
        print(f"Browser path not found at '{BROWSER_PATH}'. \n    Using default system Chrome/Chromium.")

    # Read master list
    name_list = read_master_list(MASTER_LIST_FILE)
    if name_list is None:
        exit("Cannot continue, master list not found or is empty.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    browser = Chromium(co)

    # --- Main Processing Loop ---
    for i, item in enumerate(name_list):
        # item[0] = Rank, item[1] = Player Code, item[2] = Player Name
        player_rank_master = item[0]
        player_code = item[1]
        player_name = item[2]

        tab = browser.new_tab()
        print(f"\nProcessing {i + 1}/{len(name_list)}: {player_name} (Code: {player_code})")

        try:
            scrape_player_data(tab, player_name, player_code, OUTPUT_DIR)
            # Add a small delay to avoid rate-limiting
            time.sleep(1)
        except Exception as e:
            print(f"Critical error processing {player_name}: {e}")
        finally:
            tab.close()
            print(f"Completed: {player_name}\n")

    browser.quit()
    print("All player data scraping complete!")


if __name__ == '__main__':
    main()