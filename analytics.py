import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from random import random
from statistics import mean

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import special

from tracker import SEASON_OPENER, get_tracking_period

load_dotenv()
LEAGUE_ID = os.getenv("LEAGUE_ID")

if not LEAGUE_ID:
    print("ERROR: LEAGUE_ID environment variable is not set!")
    sys.exit(1)

GAMMA = 0.88
REGULAR_SEASON_LAST_WEEK = 21
LAST_WEEK = REGULAR_SEASON_LAST_WEEK
NUM_SIMS = 10000
REGRESSION_FACTOR = 1
CURR_WIN_PROB_MATRIX = None
SLOPE = 1.12
PF_RANKING_WEIGHT = 0.65
MAX_PF_RANKING_WEIGHT = 0.35
session = requests.Session()
session.headers["Connection"] = "close"
SEASON_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), str(SEASON_OPENER.year)
)
WEEKLY_DATA_FILE = os.path.join(SEASON_DATA_DIR, "weekly_data.json")
CACHE_FILE = os.path.join(SEASON_DATA_DIR, "matchups_cache.json")

def ensure_season_data_files():
    os.makedirs(SEASON_DATA_DIR, exist_ok=True)
    for path in (WEEKLY_DATA_FILE, CACHE_FILE):
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as data_file:
                json.dump({}, data_file, indent=2)

def clamp_week(week, last_week=LAST_WEEK):
    return min(int(week), last_week)

def get_current_week(now=None):
    tracking_period = get_tracking_period(now)
    tracking_week = tracking_period[0] if tracking_period is not None else 1
    return min(max(1, tracking_week), LAST_WEEK + 1)

def format_record(wins, completed_weeks):
    return f"{wins}-{completed_weeks - wins}"

def get_seed_map(wins):
    ordered_teams = sorted(wins, key=lambda team: (-wins[team], str(team)))
    return {team: seed for seed, team in enumerate(ordered_teams, start=1)}

def get_seed_delta_summary(actual_seeds, median_seeds):
    seed_deltas = {
        team: actual_seeds[team] - median_seeds[team]
        for team in actual_seeds
    }
    improved = sum(delta > 0 for delta in seed_deltas.values())
    average_delta = mean(seed_deltas.values()) if seed_deltas else 0
    return improved, average_delta

def fetch(url, retries=10):
    i = 0
    while i < retries:
        try:
            return session.get(url, timeout=10)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError):
            time.sleep(0.5 * (i + 1))
            i += 1
    raise RuntimeError(f"Failed after {retries} retries")

def get_roster_map(league_id):
    try:
        users = fetch(f"https://api.sleeper.app/v1/league/{league_id}/users").json()   
        rosters = fetch(f"https://api.sleeper.app/v1/league/{league_id}/rosters").json()

        user_map = {user['user_id']: {
            'display_name': user.get('display_name', 'Unknown'),
            'team_name': user.get('metadata', {}).get('team_name', 'Unnamed Team')
        } for user in users}

        roster_map = {}
        scoring_map = {}
        actual_win_map = {}
        for roster in rosters:
            owner_id = roster.get('owner_id')
            roster_id = str(roster.get('roster_id'))
            if owner_id in user_map:
                roster_map[roster_id] = user_map[owner_id]['display_name']
                settings = roster['settings']
                scoring_map[roster_id] = (
                    settings['fpts'] + settings['fpts_decimal'] / 100,
                    settings['fpts_against'] + settings['fpts_against_decimal'] / 100,
                    settings['ppts'] + settings['ppts_decimal'] / 100,
                )
                actual_win_map[roster_id] = roster['metadata']['record'].count("W")
        return roster_map,scoring_map,actual_win_map
    except requests.RequestException as e:
        print(f"Error fetching roster map: {e}")
        return {}

def fetch_matchups(league_id, week):
    ensure_season_data_files()

    # Build a unique key for this request
    key = f"{league_id}-{week}"

    # Load cache if it exists
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}
    else:
        cache = {}

    # If we already have this entry, return it
    if key in cache:
        return cache[key]

    # Otherwise fetch from API
    try:
        url = f"https://api.sleeper.app/v1/league/{league_id}/matchups/{week}"
        data = fetch(url).json()
    except requests.RequestException as e:
        print(f"Error fetching matchups: {e}")
        return []

    # Store in cache
    cache[key] = data
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    return data

def initialize_team_history(team_history, completed_weeks, roster_ids):
    fields = ("pf", "pa", "mpf")
    history_updated = False

    for week in range(1, completed_weeks + 1):
        week_data = team_history.setdefault(str(week), {})
        for roster_id in roster_ids:
            stats = week_data.setdefault(roster_id, {})
            for field in fields:
                if field not in stats:
                    stats[field] = 0
                    history_updated = True

    if history_updated:
        save_team_history(WEEKLY_DATA_FILE, team_history)


def load_team_history(path):
    with open(path, "r") as f:
        return json.load(f)

def get_weekly_arrays(team_history, completed_weeks):
    completed_weeks = int(completed_weeks)
    weekly_z = defaultdict(list)
    ewtd = defaultdict(list)

    for week in range(1, completed_weeks + 1):
        team_data = team_history.get(str(week), {})
        if not team_data:
            continue

        team_ids = list(team_data.keys())
        scores = np.array([stats['pf'] for stats in team_data.values()], dtype=float)
        
        std = scores.std()
        if np.isclose(std, 0):
            z = np.zeros_like(scores, dtype=float)
        else:
            z = (scores - scores.mean()) / std
        base_ranks = scores.argsort().argsort()

        _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)

        for score_idx, counts in enumerate(counts):
            if counts > 1:
                tied_positions = np.where(inverse == score_idx)[0]
                avg_rank = base_ranks[tied_positions].mean()
                base_ranks[tied_positions] = avg_rank

        ranks = base_ranks

        for team_id, z_val, rank in zip(team_ids, z, ranks):
            weekly_z[team_id].append(z_val)
            ewtd[team_id].append(rank / 15)

    return weekly_z, ewtd

def get_weekly_field_z_scores(team_history, field, completed_weeks):
    field_z_scores = defaultdict(list)

    for week in range(1, int(completed_weeks) + 1):
        team_data = team_history.get(str(week), {})
        if not team_data:
            continue

        team_ids = list(team_data)
        scores = np.array([stats[field] for stats in team_data.values()], dtype=float)
        std = scores.std()
        if np.isclose(std, 0):
            z_scores = np.zeros_like(scores, dtype=float)
        else:
            z_scores = (scores - scores.mean()) / std

        for team_id, z_score in zip(team_ids, z_scores):
            field_z_scores[team_id].append(float(z_score))

    return field_z_scores

def save_team_history(path, team_history):
    with open(path, "w") as f:
        json.dump(team_history, f, indent=2)

def plot_rating_versus_luck(rosters, luck, ratings):
    plt.figure(figsize=(10, 6))
    plt.scatter(ratings.values(), luck.values(), s=120)

    # Label each point with the team name
    for r, l, name in zip(ratings.values(), luck.values(), rosters.values()):
        plt.text(r + 0.02, l + 0.02, name, fontsize=10)

    plt.axhline(0, color='gray', linewidth=1)
    plt.axvline(0, color='gray', linewidth=1)

    plt.xlabel("Team Rating")
    plt.ylabel("Luck (Actual Wins - Expected Wins)")
    plt.title("Luck vs Rating")

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def simulate_season(teams_list, schedule, actual_wins,scores,playoffMode=False):
    expected_wins_acc = np.zeros(len(teams_list), dtype=int)
    seed_counts = np.zeros((len(teams_list),len(teams_list)), dtype=int)
    championship_counts = np.zeros(len(teams_list), dtype=int)
    finals_counts = np.zeros(len(teams_list), dtype=int)
    # not currently being used
    semis_counts = np.zeros(len(teams_list), dtype=int)
    toilet_counts = np.zeros(len(teams_list), dtype=int)
    finals_matchups = defaultdict(int)
    toilet_matchups = defaultdict(int)
    tiebreak_arr = np.array([scores[team][0] for team in teams_list])

    for _ in range(NUM_SIMS):

        standings = np.array([actual_wins[team] for team in teams_list], dtype=int)
        
        for i, j in schedule:
            p = CURR_WIN_PROB_MATRIX[i, j]
            winner = i if random() < p else j
            standings[winner] += 1

        expected_wins_acc += standings
        order_idx = np.lexsort((tiebreak_arr,standings))
        order_idx = order_idx[::-1]

        # Record seeds
        for seed, team_idx in enumerate(order_idx):
            seed_counts[team_idx,seed] += 1

        simulate_playoffs(championship_counts, finals_counts, semis_counts, order_idx[:8], finals_matchups, playoffMode)
        simulate_toilet_bowl(toilet_counts, order_idx[8:], toilet_matchups, playoffMode)

    return seed_counts, expected_wins_acc, championship_counts, toilet_counts, finals_matchups, toilet_matchups

def simulate_toilet_bowl(toilet_counts, toilet_teams, toilet_matchups, manual=True):
    wp = CURR_WIN_PROB_MATRIX
    r = random
    tt = toilet_teams

    if not manual:
        toilet_loser_1 = tt[-1] if r() < wp[tt[0],tt[-1]] else tt[0]
        toilet_loser_2 = tt[-2] if r() < wp[tt[1],tt[-2]] else tt[1]
        toilet_loser_3 = tt[-3] if r() < wp[tt[2],tt[-3]] else tt[2]
        toilet_loser_4 = tt[-4] if r() < wp[tt[3],tt[-4]] else tt[3]
    else:
        toilet_loser_1 = tt[0]
        toilet_loser_2 = tt[1]
        toilet_loser_3 = tt[-3]
        toilet_loser_4 = tt[-4]

    toilet_semi_1 = toilet_loser_4 if r() < wp[toilet_loser_1,toilet_loser_4] else toilet_loser_1
    toilet_semi_2 = toilet_loser_3 if r() < wp[toilet_loser_2,toilet_loser_3] else toilet_loser_2

    toilet_matchups[frozenset((toilet_semi_1,toilet_semi_2))] += 1
    toilet_king = toilet_semi_2 if r() < wp[toilet_semi_1,toilet_semi_2] else toilet_semi_1
    toilet_counts[toilet_king] += 1

def simulate_playoffs(championship_counts, finals_counts, semis_counts, playoff_teams, finals_matchups, manual=True):
    wp = CURR_WIN_PROB_MATRIX
    r = random
    pt = playoff_teams

    if not manual:
        quarter_winner_1 = pt[0] if r() < wp[pt[0],pt[-1]] else pt[-1]
        quarter_winner_2 = pt[1] if r() < wp[pt[1],pt[-2]] else pt[-2]
        quarter_winner_3 = pt[2] if r() < wp[pt[2],pt[-3]] else pt[-3]
        quarter_winner_4 = pt[3] if r() < wp[pt[3],pt[-4]] else pt[-4]
    else:
        quarter_winner_1 = pt[0]
        quarter_winner_2 = pt[1]
        quarter_winner_3 = pt[2]
        quarter_winner_4 = pt[3]

    semis_counts[quarter_winner_1] += 1
    semis_counts[quarter_winner_2] += 1
    semis_counts[quarter_winner_3] += 1
    semis_counts[quarter_winner_4] += 1

    semi_winner_1 = quarter_winner_1 if r() < wp[quarter_winner_1,quarter_winner_4] else quarter_winner_4
    finals_counts[semi_winner_1] += 1
    semi_winner_2 = quarter_winner_2 if r() < wp[quarter_winner_2,quarter_winner_3] else quarter_winner_3
    finals_counts[semi_winner_2] += 1

    finals_matchups[frozenset((semi_winner_1,semi_winner_2))] += 1
    champion = semi_winner_1 if r() < wp[semi_winner_1,semi_winner_2] else semi_winner_2
    championship_counts[champion] += 1

def get_remaining_schedule(current_week,team_index,week_num=LAST_WEEK):
    remaining_schedule = []
    matchup_map = defaultdict(list)

    for i in range(current_week,week_num+1):
        for team in fetch_matchups(LEAGUE_ID, i):
            matchup_map[team['matchup_id']].append(team_index[str(team['roster_id'])])
        remaining_schedule.extend(matchup_map.values())
        matchup_map.clear()

    return remaining_schedule

def main():

    global CURR_WIN_PROB_MATRIX

    parser = argparse.ArgumentParser(description="League Analytics")

    parser.add_argument("--update-week", action="store_true",
                        help="Update weekly_data.json with the latest week")

    parser.add_argument("--print-power-rankings", action="store_true",
                        help="Print the current power rankings")

    parser.add_argument("--plot-luck", action="store_true",
                        help="Plot rating vs luck")

    parser.add_argument("--simulate-season", action="store_true",
                        help="Run full season simulation")

    parser.add_argument("--playoff-mode", action="store_true",
                        help="Include playoff outcome simulation")
    
    parser.add_argument("--current-week-probs", action="store_true",
                        help="Get the probabilities for this week's matchups")
    
    # parser.add_argument("--backtesting", action="store_true",
    #                     help="Perform backtesting")
    
    parser.add_argument("--momentum", action="store_true",
                        help="Get momentum using last 3 weeks")
    
    parser.add_argument("--parity", action="store_true",
                        help="Get league parity")
    
    parser.add_argument("--strength-of-schedule", action="store_true",
                        help="Get average rating of remaining opponents")
    
    parser.add_argument("--versus-median", action="store_true",
                        help="Get team records if vs. median")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr) # Print the help message to standard error
        sys.exit(1) # Exit with an error code

    # Load shared data

    current_week = get_current_week()
    completed_weeks = current_week - 1
    rosters,scores,actual_wins = get_roster_map(LEAGUE_ID)
    ensure_season_data_files()
    team_history = load_team_history(WEEKLY_DATA_FILE)
    initialize_team_history(team_history, completed_weeks, rosters)

    # --- COMMAND ROUTING ---

    if args.update_week:
        update_week_data(str(completed_weeks), scores, team_history)
        print(f"Updated data for week {completed_weeks}")

    teams_list = sorted(rosters.keys(), key=int)
    team_index = {team: i for i, team in enumerate(teams_list)}

    # if args.backtesting:
        
    #     bt_correct = bt_total = total_underdog_games = upsets = 0
    #     bt_brier = 0.0

    #     buckets = {
    #         (0.50, 0.55): [],
    #         (0.55, 0.60): [],
    #         (0.60, 0.65): [],
    #         (0.65, 0.70): [],
    #         (0.70, 0.75): [],
    #         (0.75, 1.01): []
    #     }

    #     wpm_map = {}
    #     weekly_z = defaultdict(list)

    #     for week in range(1, current_week):
            
    #         rating_map = get_weekly_arrays_for_week(team_history, weekly_z, week)
    #         wpm = get_wpm(teams_list, rating_map)
    #         wpm_map[week] = wpm

    #         if week >= 2:
    #             matchup_map = defaultdict(list)
    #             for team in fetch_matchups(LEAGUE_ID, week):
    #                 matchup_map[team['matchup_id']].append(team_index[str(team['roster_id'])])
    #             games = list(matchup_map.values())          # predict Week N+1

    #             wpm = wpm_map[week - 1]  # ratings BEFORE the game
    #             week_data = team_history[str(week)]
                            
    #             for i, j in games:
    #                 # predicted probability at the time
    #                 p = wpm[i,j]

    #                 pfA = week_data[teams_list[i]]['pf']
    #                 pfB = week_data[teams_list[j]]['pf']

    #                 # actual outcome
    #                 actual = 1 if pfA > pfB else 0

    #                 # accuracy
    #                 pred_winner = i if p >= 0.5 else j
    #                 actual_winner = i if actual == 1 else j
    #                 if pred_winner == actual_winner:
    #                     bt_correct += 1
    #                 bt_total += 1

    #                 if p < 0.5:
    #                     total_underdog_games += 1
    #                     if actual_winner == i:
    #                         upsets += 1

    #                 # Brier contribution
    #                 bt_brier += (p - actual)**2

    #                 # Bucket
    #                 for (low, high) in buckets:
    #                     if low <= p < high:
    #                         buckets[(low, high)].append((p,actual))
    #                         break
    #                     elif low <= 1 - p < high:
    #                         buckets[(low, high)].append((1-p,1-actual))
    #                         break

    #     # final metrics
    #     bt_accuracy = bt_correct / bt_total
    #     bt_brier_score = bt_brier / bt_total

    #     print("Backtest accuracy:", bt_accuracy)
    #     print("Backtest Brier:", bt_brier_score)
    #     print("Calibration buckets:")
    #     for (low, high), outcomes in buckets.items():
    #         if outcomes:
    #            avg_pred = sum(q for q, _ in outcomes) / len(outcomes)
    #            avg_actual = sum(a for _, a in outcomes) / len(outcomes)
    #            print(f"{low:.2f}-{high:.2f}: pred={avg_pred:.3f}, actual={avg_actual:.3f}, n={len(outcomes)}")

    #     if total_underdog_games > 0:
    #         upset_frequency = upsets / total_underdog_games
    #     else:
    #         upset_frequency = 0

    #     print("Upset frequency:", upset_frequency)
        
    weekly_z, ewtd = get_weekly_arrays(team_history, completed_weeks)

    if args.versus_median:

        completed_weeks = clamp_week(completed_weeks, REGULAR_SEASON_LAST_WEEK)

        median_wins = {
            team: sum(1 for win in ewtd[team] if win > 0.5) + actual_wins[team]
            for team in actual_wins
        }
        total_pf = {
            team: sum(
                float(team_history.get(str(week), {}).get(team, {}).get("pf", 0))
                for week in range(1, completed_weeks + 1)
            )
            for team in median_wins
        }

        median_wins_sorted = sorted(
            median_wins,
            key=lambda team: (-median_wins[team], -total_pf[team], str(team)),
        )
        team_width = max(len(rosters[team]) for team in median_wins_sorted)
        pf_width = max(len(f"{total_pf[team]:.2f}") for team in median_wins_sorted)
        record_width = max(
            len(format_record(median_wins[team], completed_weeks * 2))
            for team in median_wins_sorted
        )
        print("Versus median record:")
        print(
            f"{'Team':<{team_width}} | {'Orig':>{record_width}} -> "
            f"{'Median':<{record_width}} | {'Total PF':>{pf_width}}"
        )
        for team in median_wins_sorted:
            median_record = format_record(median_wins[team], completed_weeks * 2)
            actual_record = format_record(actual_wins[team], completed_weeks)
            print(
                f"{rosters[team]:<{team_width}} | "
                f"{actual_record:>{record_width}} -> {median_record:<{record_width}} | "
                f"{total_pf[team]:>{pf_width}.2f}"
            )
        
    max_pf_weekly_z = get_weekly_field_z_scores(
        team_history, "mpf", completed_weeks
    )

    if args.momentum:
        momentum_scores = get_momentum_scores(weekly_z, max_pf_weekly_z)
        momentum = {
            rosters[team]: score
            for team, score in momentum_scores.items()
        }
        
        momentum_sorted = sorted(momentum.items(),
                                 key=lambda x:x[1],
                                 reverse=True)

        print(f"\033[4mHottest Teams\033[0m")
        for team, m in momentum_sorted[:3]:
            print(f"{team}: {m}")        

        print(f"\033[4mColdest Teams\033[0m")
        for team, m in momentum_sorted[-3:][::-1]:
            print(f"{team}: {m}")

    power_ranking_map, curr_team_rating_map = get_rating_maps(
        completed_weeks, weekly_z, max_pf_weekly_z
    )

    if args.strength_of_schedule:
        remaining_schedule = get_remaining_schedule(current_week, team_index)
        opp_ratings = defaultdict(list)

        for i, j in remaining_schedule:
            opp_ratings[rosters[teams_list[i]]].append(curr_team_rating_map[teams_list[j]])
            opp_ratings[rosters[teams_list[j]]].append(curr_team_rating_map[teams_list[i]])

        avg_opp_rating = {team : mean(opp_ratings) for team, opp_ratings in opp_ratings.items()}
        avg_opp_rating_sorted = sorted(avg_opp_rating.items(),key=lambda x:x[1],reverse=True)

        lines = []
        for idx, (team, sos) in enumerate(avg_opp_rating_sorted, start=1):
            lines.append(f"{idx:>2}. {team:<22} {sos:>6.3f}")

        print(f"\033[4mRemaining Strength of Schedule - Week {current_week}\033[0m")
        output = "\n".join(lines)
        print(output)

    if args.parity:
        parity_index = np.std(list(curr_team_rating_map.values()))
        print(f"Parity: {parity_index}")

    if args.print_power_rankings:
        prev_week = completed_weeks - 1
        prev_weekly_z = {k: v[:-1] for k, v in weekly_z.items()}
        prev_max_pf_weekly_z = {
            k: v[:-1] for k, v in max_pf_weekly_z.items()
        }
        
        prev_power_ranking_map, _ = get_rating_maps(
            prev_week, prev_weekly_z, prev_max_pf_weekly_z
        )

        print_power_rankings(power_ranking_map,rosters,current_week,prev_power_ranking_map)

    if args.plot_luck:
        luck = {
            key: actual_wins[key] - sum(ewtd[str(key)])
            for key in actual_wins
        }
        plot_rating_versus_luck(rosters, luck, power_ranking_map)
    
    CURR_WIN_PROB_MATRIX = get_wpm(teams_list,curr_team_rating_map)

    if args.simulate_season:
        remaining_schedule = get_remaining_schedule(current_week, team_index)
        seed_counts, expected_wins_acc, championship_odds, toilet_odds,finals_matchups,toilet_matchups = simulate_season(teams_list, remaining_schedule, actual_wins, scores, args.playoff_mode)
        champ_probs = championship_odds / NUM_SIMS
        champ_sorted_idx = np.argsort(champ_probs)[::-1]
        print("Championship Odds:")
        for idx in champ_sorted_idx[:8]:
            team_id = teams_list[idx]
            team_name = rosters[team_id]
            print(f"{team_name}: {champ_probs[idx]:.2%}")

        toilet_probs = toilet_odds / NUM_SIMS
        toilet_sorted_idx = np.argsort(toilet_probs)[::-1]
        print("\nToilet Odds:")
        for idx in toilet_sorted_idx[:8]:
            team_id = teams_list[idx]
            team_name = rosters[team_id]
            print(f"{team_name}: {toilet_probs[idx]:.2%}")

        # print("\nLikeliest Championship Matchups:")
        # for pair, count in sorted(finals_matchups.items(), key=lambda x: x[1], reverse=True):
        #     a, b = tuple(pair)
        #     print(f"{rosters[teams_list[a]]} vs {rosters[teams_list[b]]}: {(count/NUM_SIMS):.2%}")

        # print("\nLikeliest Toilet King Matchups:")
        # for pair, count in sorted(toilet_matchups.items(), key=lambda x: x[1], reverse=True):
        #     a, b = tuple(pair)
        #     print(f"{rosters[teams_list[a]]} vs {rosters[teams_list[b]]}: {(count/NUM_SIMS):.2%}")

    if args.current_week_probs:
        remaining_schedule = get_remaining_schedule(current_week,team_index,current_week)
        print_week_win_probs(remaining_schedule,rosters,current_week,teams_list)

def get_rating_maps(completed_weeks, weekly_pf_z, weekly_max_pf_z):
    power_ranking_map = {}
    curr_team_rating_map = {}

    for team, pf_scores in weekly_pf_z.items():
        max_pf_scores = weekly_max_pf_z.get(team, [])
        weeks_to_score = min(
            int(completed_weeks), len(pf_scores), len(max_pf_scores)
        )
        if weeks_to_score:
            pf_rating = mean(pf_scores[:weeks_to_score])
            max_pf_rating = mean(max_pf_scores[:weeks_to_score])
            rating = (
                PF_RANKING_WEIGHT * pf_rating
                + MAX_PF_RANKING_WEIGHT * max_pf_rating
            )
        else:
            rating = 0.0

        power_ranking_map[team] = rating
        curr_team_rating_map[team] = rating

    return power_ranking_map,curr_team_rating_map

def get_momentum_scores(weekly_pf_z, weekly_max_pf_z):
    momentum_scores = {}

    for team, pf_scores in weekly_pf_z.items():
        max_pf_scores = weekly_max_pf_z.get(team, [])
        weeks_to_score = min(3, len(pf_scores), len(max_pf_scores))
        if weeks_to_score:
            weekly_scores = [
                PF_RANKING_WEIGHT * pf_score
                + MAX_PF_RANKING_WEIGHT * max_pf_score
                for pf_score, max_pf_score in zip(
                    pf_scores[-weeks_to_score:],
                    max_pf_scores[-weeks_to_score:],
                )
            ]
            momentum_scores[team] = mean(weekly_scores)
        else:
            momentum_scores[team] = 0.0

    return momentum_scores

def get_wpm(teams_list, rating_map, platt_model=None):
    ratings = np.array([rating_map[team] for team in teams_list])
    diff = ratings[:, None] - ratings[None, :]
    z = diff / math.sqrt(2)
    base_wpm = 0.5 * (1 + special.erf(SLOPE*z / math.sqrt(2)))

    if platt_model is None:
        return base_wpm

    calibrated = np.zeros_like(base_wpm)

    flat = base_wpm.flatten()

    flat = np.asarray(flat).reshape(-1, 1)
    cal_flat = platt_model.predict_proba(flat)[:, 1]
    calibrated = cal_flat.reshape(base_wpm.shape)

    np.fill_diagonal(calibrated, 0.5)
    return calibrated

def get_weekly_arrays_for_week(team_history, weekly_z, week):
    week_data = team_history[str(week)]

            # compute z-scores for THIS week only
    scores = np.array([stats['pf'] for stats in week_data.values()], dtype=int)
    mu = scores.mean()
    sigma = scores.std() or 1.0
    z = (scores - mu) / sigma

            # append to weekly_z
    for team_id, z_val in zip(week_data.keys(), z):
        weekly_z[team_id].append(z_val)

            # compute rating snapshot for this week
    n = week
    weights = get_recency_weights(n)

    rating_map = {}
    for team, z_list in weekly_z.items():
        rating_map[team] = get_rating(n, weights, z_list)
    return rating_map

def get_rating(n, weights, z_list):
    weighted = sum(v * w for v, w in zip(z_list, weights))
    regressed = weighted * n / (n + REGRESSION_FACTOR)
    return regressed

def get_recency_weights(n):
    raw_weights = [GAMMA ** (n - 1 - i) for i in range(n)]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]
    return weights

def expected_seed(prob_array):
    seeds = np.arange(1, len(prob_array) + 1)
    return np.sum(prob_array * seeds)

def plot_seed_heatmap(seed_map,current_week):
    teams = sorted(seed_map.keys(), key=lambda team: expected_seed(seed_map[team]))
    data = np.array([seed_map[t] for t in teams])

    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect="auto", cmap="Blues")

    plt.colorbar(label="Probability")
    plt.xticks(range(16), range(1, 17))
    plt.yticks(range(16), teams)

    plt.title(f"Seed Distribution Heatmap Before Week {current_week}")
    plt.xlabel("Seed")
    plt.ylabel("Team")
    plt.tight_layout()
    plt.show()

def print_power_rankings(power_ranking_map, rosters, week, prev_power_ranking_map=None):
    print(f"\033[4mWeek {week} Power Rankings\033[0m")

    # Sort current week
    current_sorted = sorted(
        power_ranking_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Sort previous week if available
    if prev_power_ranking_map:
        prev_sorted = sorted(
            prev_power_ranking_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        prev_pos = {team_id: idx for idx, (team_id, _) in enumerate(prev_sorted, start=1)}
        prev_val = {team_id: val for team_id, val in prev_sorted}
    else:
        prev_pos = {}
        prev_val = {}

    value_deltas = {}
    for idx, (team_id, value) in enumerate(current_sorted, start=1):
        name = rosters[team_id]

        # compute delta
        if team_id in prev_pos:
            delta_pos = prev_pos[team_id] - idx
            pos_str = delta_emoji(delta_pos)
        else:
            pos_str = ""

        if team_id in prev_val:
            delta_val = value - prev_val[team_id]
            value_deltas[team_id] = delta_val
            delta_val_str = f"({delta_val:+.3f})"
        else:
            delta_val_str = ""

        print(f"{idx:>2}. {name:<22} {pos_str:<2}  {value:>6.3f} {delta_val_str}")

    if value_deltas:
        # Sort by delta descending for risers
        risers = sorted(value_deltas.items(), key=lambda x: x[1], reverse=True)[:3]
        # Sort by delta ascending for fallers
        fallers = sorted(value_deltas.items(), key=lambda x: x[1])[:3]

        print("\nTop Risers:")
        for team_id, delta in risers:
            name = rosters[team_id]
            print(f"  {name:<22} +{delta:.3f}")

        print("\nTop Fallers:")
        for team_id, delta in fallers:
            name = rosters[team_id]
            print(f"  {name:<22} {delta:.3f}")

def delta_emoji(delta):
    if delta > 0:
        return f"🟢 ↑{delta}"
    elif delta < 0:
        return f"🔴 ↓{-delta}"
    else:
        return "➖   "

def print_week_win_probs(matchups, rosters, week, team_list):
    print(f"\033[4mWin Probabilities - Week {week}\033[0m\n")

    for i, j in matchups:
        pA = CURR_WIN_PROB_MATRIX[i,j]
        pB = 1 - pA

        nameA = rosters[team_list[i]]
        nameB = rosters[team_list[j]]

        print(f"{nameA} vs {nameB}")
        print(f"    {matchup_line(pA, pB, nameA, nameB)}\n")

def matchup_line(pA, pB, nameA, nameB):
    if abs(pA - pB) < 0.005:
        # true coin flip
        return f"Pick’em ({pA*100:.1f}% vs {pB*100:.1f}%)"
    elif pA > pB:
        return f"{nameA} favored ({pA*100:.1f}% vs {pB*100:.1f}%)"
    else:
        return f"{nameB} favored ({pB*100:.1f}% vs {pA*100:.1f}%)"

def update_week_data(completed_weeks, scores, team_history):
    week_key = str(completed_weeks)
    previous_history = {
        week: data for week, data in team_history.items() if week != week_key
    }
    new_data = {}
    fields = ("pf", "pa", "mpf")

    for team, totals in scores.items():
        new_data[team] = {}
        for field, total in zip(fields, totals):
            previous_total = sum(
                float(week_data.get(team, {}).get(field, 0))
                for week_data in previous_history.values()
            )
            new_data[team][field] = total - previous_total

    team_history[week_key] = new_data
    save_team_history(WEEKLY_DATA_FILE, team_history)

if __name__ == "__main__":
    main()
