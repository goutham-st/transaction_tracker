from collections import defaultdict
import json
import math
from random import random
import numpy as np
import requests
from dotenv import load_dotenv
import os
import sys
from tracker import START_DATE, get_week_by_cutoff
from statistics import pstdev
import matplotlib.pyplot as plt
import argparse
from scipy import special

load_dotenv()
LEAGUE_ID = os.getenv("LEAGUE_ID")

if not LEAGUE_ID:
    print("ERROR: LEAGUE_ID environment variable is not set!")
    sys.exit(1)

GAMMA = 0.88
ALPHA = 0.75
LAST_WEEK = 21
NUM_SIMS = 10000
REGRESSION_FACTOR = 4
TEAM_RATING_MAP = None
WIN_PROB_MATRIX = None

def get_roster_map(league_id):
    try:
        users_response = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users", timeout=10)
        users_response.raise_for_status()
        users = users_response.json()
        
        rosters_response = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters", timeout=10)
        rosters_response.raise_for_status()
        rosters = rosters_response.json()

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
                scoring_map[roster_id] = (settings['fpts'],settings['fpts_against'],settings['ppts'])
                actual_win_map[roster_id] = roster['metadata']['record'].count("W")
        return roster_map,scoring_map,actual_win_map
    except requests.RequestException as e:
        print(f"Error fetching roster map: {e}")
        return {}

def fetch_matchups(league_id, week):
    url = f"https://api.sleeper.app/v1/league/{league_id}/matchups/{week}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching transactions: {e}")
        return []

def get_win_prob(A, B):    
    # Z-score of the difference
    z = (TEAM_RATING_MAP[str(A)] - TEAM_RATING_MAP[str(B)]) / math.sqrt(2)
    # Standard normal CDF
    win_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return win_prob

def load_team_history(path):
    with open(path, "r") as f:
        return json.load(f)

def get_weekly_arrays(team_history, completed_weeks):
    completed_weeks = int(completed_weeks)
    weekly_z = defaultdict(list)
    ewtd = defaultdict(float)

    # for team_data in team_history.values():
    #     weekly_scores = {team_id : stats['pf'] for team_id,stats in team_data.items()}
    #     mu_league = mean(weekly_scores.values())
    #     sigma_league = pstdev(weekly_scores.values())

    #     for team_id,score in weekly_scores.items():
    #         weekly_z[team_id].append((score-mu_league)/sigma_league)

    #         sorted_weekly_scores = sorted(weekly_scores.values())
    #         index = sorted_weekly_scores.index(score)            

    #         if sorted_weekly_scores.count(score) > 1:
    #             indexes = [index for index, value in enumerate(sorted_weekly_scores) if value == score]
    #             index = mean(indexes)
                
    #         ewtd[team_id] += index / 15

    for team_data in team_history.values():
        team_ids = list(team_data.keys())
        scores = np.array([stats['pf'] for stats in team_data.values()],dtype=int)
        
        z = (scores - scores.mean()) / scores.std()
        base_ranks = scores.argsort().argsort()

        unique_vals, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)

        for score_idx, counts in enumerate(counts):
            if counts > 1:
                tied_positions = np.where(inverse == score_idx)[0]
                avg_rank = base_ranks[tied_positions].mean()
                base_ranks[tied_positions] = avg_rank

        ranks = base_ranks

        for team_id, z_val, rank in zip(team_ids, z, ranks):
            weekly_z[team_id].append(z_val)
            ewtd[team_id] += rank / 15

    ewtd_normalized = {team : (ewtd_t - completed_weeks / 2) / (pstdev(ewtd.values())) 
                       for team, ewtd_t in ewtd.items()}
    
    raw_weights = [GAMMA ** (completed_weeks - 1 - i) for i in range(completed_weeks)]
    # normalize weights so they sum to 1
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    power_ranking_map = {}
    team_rating_map = {}

    for team,z_list in weekly_z.items():
        team_rating = sum(v * w for v, w in zip(z_list, weights)) * completed_weeks / (completed_weeks + REGRESSION_FACTOR)
        team_rating_map[team] = team_rating
        power_ranking_map[team] = ALPHA*team_rating + (1-ALPHA)*ewtd_normalized[team]

    return power_ranking_map, team_rating_map, ewtd

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

def mathematically_clinched(team, current_wins, remaining_games):

    # Compute max possible wins for every team
    max_wins = {
        t: current_wins[t] + remaining_games
        for t in current_wins.keys()
    }

    # Sort teams by current wins (descending)
    sorted_teams = sorted(current_wins.keys(), key=lambda t: current_wins[t], reverse=True)

    # Identify the 9th-place team (first team OUT of playoffs)
    # If fewer than 9 teams exist, adjust accordingly
    if len(sorted_teams) > 8:
        ninth_place_team = sorted_teams[8]
        ninth_place_max = max_wins[ninth_place_team]
    else:
        # If fewer than 9 teams, nobody can be "out"
        return False

    # Team clinches if its MIN possible wins > MAX possible wins of 9th place
    if current_wins[team] > ninth_place_max:
        return True

    return False

def mathematically_eliminated(team, current_wins, remaining_games):
    max_wins_team = current_wins[team] + remaining_games

    # Sort by current wins (desc)
    sorted_teams = sorted(current_wins.keys(), key=lambda t: current_wins[t], reverse=True)

    # 8th place team (last team IN)
    if len(sorted_teams) < 8:
        return False  # trivial case, not your league

    eighth_place_team = sorted_teams[7]
    eighth_place_wins = current_wins[eighth_place_team]

    # Eliminated if even max possible wins < current wins of 8th place
    return max_wins_team < eighth_place_wins

def simulate_season(teams_list, schedule, actual_wins,scores):
    expected_wins_acc = np.zeros(len(teams_list), dtype=int)
    seed_counts = np.zeros((len(teams_list),len(teams_list)), dtype=int)
    playoff_counts = np.zeros(len(teams_list), dtype=int)
    championship_counts = np.zeros(len(teams_list), dtype=int)
    finals_counts = np.zeros(len(teams_list), dtype=int)
    semis_counts = np.zeros(len(teams_list), dtype=int)
    toilet_counts = np.zeros(len(teams_list), dtype=int)
    tiebreak_arr = np.array([scores[team][0] for team in teams_list])

    for _ in range(NUM_SIMS):

        standings = np.array([actual_wins[team] for team in teams_list], dtype=int)
        
        for i, j in schedule:
            p = WIN_PROB_MATRIX[i, j]
            winner = i if random() < p else j
            standings[winner] += 1

        expected_wins_acc += standings
        order_idx = np.lexsort((tiebreak_arr,standings))
        order_idx = order_idx[::-1]

        # Record playoff appearances and seeds
        for seed, team_idx in enumerate(order_idx):
            seed_counts[team_idx,seed] += 1
            if seed < 7:
                playoff_counts[team_idx] += 1

        simulate_playoffs(championship_counts, finals_counts, semis_counts, order_idx[:8])
        simulate_toilet_bowl(toilet_counts, order_idx[8:])

    return playoff_counts, seed_counts, expected_wins_acc, championship_counts, toilet_counts

def simulate_toilet_bowl(toilet_counts, toilet_teams):

    wp = WIN_PROB_MATRIX
    r = random
    tt = toilet_teams

    toilet_loser_1 = tt[-1] if r() < wp[tt[0],tt[-1]] else tt[0]
    toilet_loser_2 = tt[-2] if r() < wp[tt[1],tt[-2]] else tt[1]
    toilet_loser_3 = tt[-3] if r() < wp[tt[2],tt[-3]] else tt[2]
    toilet_loser_4 = tt[-4] if r() < wp[tt[3],tt[-4]] else tt[3]

    toilet_semi_1 = toilet_loser_4 if r() < wp[toilet_loser_1,toilet_loser_4] else toilet_loser_1
    toilet_semi_2 = toilet_loser_3 if r() < wp[toilet_loser_2,toilet_loser_3] else toilet_loser_2

    toilet_king = toilet_semi_2 if r() < wp[toilet_semi_1,toilet_semi_2] else toilet_semi_1
    toilet_counts[toilet_king] += 1

def simulate_playoffs(championship_counts, finals_counts, semis_counts, playoff_teams):

    wp = WIN_PROB_MATRIX
    r = random
    pt = playoff_teams

    quarter_winner_1 = pt[0] if r() < wp[pt[0],pt[-1]] else pt[-1]
    semis_counts[quarter_winner_1] += 1
    quarter_winner_2 =  pt[1] if r() < wp[pt[1],pt[-2]] else pt[-2]
    semis_counts[quarter_winner_2] += 1
    quarter_winner_3 = pt[2] if r() < wp[pt[2],pt[-3]] else pt[-3]
    semis_counts[quarter_winner_3] += 1
    quarter_winner_4 = pt[3] if r() < wp[pt[3],pt[-4]] else pt[-4]
    semis_counts[quarter_winner_4] += 1

    semi_winner_1 = quarter_winner_1 if r() < wp[quarter_winner_1,quarter_winner_4] else quarter_winner_4
    finals_counts[semi_winner_1] += 1
    semi_winner_2 = quarter_winner_2 if r() < wp[quarter_winner_2,quarter_winner_3] else quarter_winner_3
    finals_counts[semi_winner_2] += 1

    champion = semi_winner_1 if r() < wp[semi_winner_1,semi_winner_2] else semi_winner_2
    championship_counts[champion] += 1

# def simulate_regular_season(teams, schedule, actual_wins):
#     # 1. Copy current standings
#     standings = {team: actual_wins[team] for team in teams}

#     # 2. Simulate remaining games
#     for (A, B) in schedule:
#         standings[get_winner(A,B)] += 1

#     return standings

# def get_winner(A, B):
#     return A if random() < WIN_PROB_MATRIX[A,B] else B

# def get_loser(A, B):
#     return B if random() < WIN_PROB_MATRIX[A,B] else A

def get_remaining_schedule(current_week,team_index,week_num=LAST_WEEK):
    remaining_schedule = []
    matchup_map = defaultdict(list)

    for i in range(current_week,week_num+1):
        for team in fetch_matchups(LEAGUE_ID, i):
            matchup_map[team['matchup_id']].append(team_index[str(team['roster_id'])])
        remaining_schedule.extend([matchup for matchup in matchup_map.values()])
        matchup_map.clear()

    return remaining_schedule

def playoff_odds_report(completed_weeks, rosters, actual_wins, playoff_appearances):
    for key,value in playoff_appearances.items():
        if value == NUM_SIMS and not mathematically_clinched(key,actual_wins,LAST_WEEK-completed_weeks):
            value = NUM_SIMS-1
        elif value == 0 and not mathematically_eliminated(key,actual_wins,LAST_WEEK-completed_weeks):
            value = 1

        print(f"{rosters[key]}: {100*value/NUM_SIMS}%")

def main():

    global TEAM_RATING_MAP
    global WIN_PROB_MATRIX

    parser = argparse.ArgumentParser(description="League Analytics")

    parser.add_argument("--update-week", action="store_true",
                        help="Update weekly_data.json with the latest week")

    parser.add_argument("--print-power-rankings", action="store_true",
                        help="Print the current power rankings")

    parser.add_argument("--plot-luck", action="store_true",
                        help="Plot rating vs luck")

    parser.add_argument("--simulate-season", action="store_true",
                        help="Run full season simulation")
    
    parser.add_argument("--current-week-probs", action="store_true",
                        help="Get the probabilities for this week's matchups")

    args = parser.parse_args()

    # Load shared data

    current_week = get_week_by_cutoff(START_DATE)
    completed_weeks = current_week - 1
    rosters,scores,actual_wins = get_roster_map(LEAGUE_ID)
    team_history = load_team_history("weekly_data.json")

    # --- COMMAND ROUTING ---

    if args.update_week:
        if str(completed_weeks) not in team_history:        
            update_week_data(completed_weeks, scores, team_history)
        else:
            print("Data already up to date")

    power_ranking_map,TEAM_RATING_MAP,expected_wins = get_weekly_arrays(team_history, completed_weeks)

    if args.print_power_rankings:
        prev_week = str(completed_weeks - 1)
        prev_power_ranking_map = None
        if prev_week in team_history:
            prev_power_ranking_map, _, _ = get_weekly_arrays(
                {k: v for k, v in team_history.items() if int(k) <= completed_weeks - 1}, prev_week
            )
        print_power_rankings(power_ranking_map,rosters,current_week,prev_power_ranking_map)

    if args.plot_luck:
        luck = {
            key: actual_wins[key] - expected_wins[str(key)]
            for key in actual_wins
        }
        plot_rating_versus_luck(rosters, luck, power_ranking_map)

    teams_list = sorted(TEAM_RATING_MAP.keys(), key=int)
    team_index = {team: i for i, team in enumerate(teams_list)}
    ratings = np.array([TEAM_RATING_MAP[team] for team in teams_list])

    diff = ratings[:, None] - ratings[None, :]
    z = diff / math.sqrt(2)
    WIN_PROB_MATRIX = 0.5 * (1 + special.erf(z / math.sqrt(2)))

    if args.simulate_season:
        remaining_schedule = get_remaining_schedule(current_week, team_index)
        playoff_counts, seed_counts, expected_wins_acc, championship_odds, toilet_odds = simulate_season(teams_list, remaining_schedule, actual_wins, scores)

        seed_distribution = {}

        for i, team in enumerate(teams_list):
            print(f"Team {rosters[team]}: expected wins = {expected_wins_acc[i] / NUM_SIMS:.3f}")
            print(f"Team {rosters[team]}: playoff probability = {playoff_counts[i] / NUM_SIMS:.3f}")
            print(f"Team {rosters[team]}: championship odds = {championship_odds[i] / NUM_SIMS:.3f}")
            print(f"Team {rosters[team]}: toilet odds = {toilet_odds[i] / NUM_SIMS:.3f}")
            print("\n")
            seed_distribution[rosters[team]] = seed_counts[i] / NUM_SIMS
            
        plot_seed_heatmap(seed_distribution)

    if args.current_week_probs:
        remaining_schedule = get_remaining_schedule(current_week,team_index,current_week)
        print_week_win_probs(remaining_schedule,rosters,current_week,teams_list)

def plot_seed_heatmap(seed_map):
    teams = list(seed_map.keys())
    data = np.array([seed_map[t] for t in teams])

    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect="auto", cmap="Blues")

    plt.colorbar(label="Probability")
    plt.xticks(range(16), range(1, 17))
    plt.yticks(range(16), teams)

    plt.title("Seed Distribution Heatmap")
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
    else:
        prev_pos = {}

    for idx, (team_id, _) in enumerate(current_sorted, start=1):
        name = rosters[team_id]

        # compute delta
        if team_id in prev_pos:
            delta = prev_pos[team_id] - idx
            print(f"{idx:>2}. {name:<22} {delta_emoji(delta)}")

def delta_emoji(delta):
    if delta > 0:
        return f"🟢 ↑{delta}"
    elif delta < 0:
        return f"🔴 ↓{-delta}"
    else:
        return "➖"

def print_week_win_probs(matchups, rosters, week, team_list):
    print(f"\033[4mWin Probabilities - Week {week}\033[0m\n")

    for i, j in matchups:
        pA = WIN_PROB_MATRIX[i,j]
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
    new_data = {}
    for team,(pf,pa,mpf) in scores.items():
        pf_sum = pa_sum = mpf_sum = 0
        for stats in team_history.values():
            pf_sum += stats[str(team)]['pf']
            pa_sum += stats[str(team)]['pa']
            mpf_sum += stats[str(team)]['mpf']
        new_data[str(team)] = {'pf' : pf - pf_sum, 
                                    "pa" : pa - pa_sum, 
                                    "mpf" : mpf - mpf_sum}
                
    team_history[str(completed_weeks)] = new_data
    save_team_history("weekly_data.json", team_history)

if __name__ == "__main__":
    main()
