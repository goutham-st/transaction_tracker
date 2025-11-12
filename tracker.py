import discord
import requests
from datetime import datetime, timedelta
import pytz

TOKEN = "***REMOVED***"
LEAGUE_ID = "***REMOVED***"
START_DATE = datetime(2025, 10, 19)  # Sunday before Week 1

THRESHOLD_SECONDS = 1800  # 30 minutes

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def get_player_map():

    url = "https://api.sleeper.app/v1/players/nba"
    response = requests.get(url)
    data = response.json()
    
    # Build a map: player_id → full_name
    player_map = {
        pid: info.get("full_name", "Unknown")
        for pid, info in data.items()
        if "full_name" in info
    }
    return player_map

def get_week_by_sunday_cutoff(start_date):
    today = datetime.now()
    # Move to previous Sunday
    last_sunday = today - timedelta(days=today.weekday() + 1)
    delta = last_sunday - start_date
    return delta.days // 7 + 1

def get_roster_map(league_id):
    users = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users").json()
    rosters = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters").json()

    user_map = {user['user_id']: {
        'display_name': user.get('display_name', 'Unknown'),
        'team_name': user.get('metadata', {}).get('team_name', 'Unnamed Team')
    } for user in users}

    roster_map = {}
    for roster in rosters:
        owner_id = roster.get('owner_id')
        roster_id = roster.get('roster_id')
        if owner_id in user_map:
            roster_map[roster_id] = user_map[owner_id]['display_name']
        else:
            roster_map[roster_id] = 'Unknown'

    return roster_map

def fetch_transactions(league_id, week):
    url = f"https://api.sleeper.app/v1/league/{league_id}/transactions/{week}"
    response = requests.get(url)
    return response.json()

def filter_valid_pickups(transactions):
    pickups = {}
    for tx in transactions:
        if tx['type'] in ['free_agent','waiver'] and tx['adds']:
            player_id, roster_id = next(iter(tx['adds'].items()))
            # print(tx['status_updated'])
            add_time = datetime.fromtimestamp(tx['status_updated']/1000, pytz.utc)

            # Check for quick drop
            dropped = False
            for other_tx in transactions:
                if other_tx['drops'] and other_tx['roster_ids'][0] == roster_id:
                    if player_id in other_tx['drops']:
                        drop_time = datetime.fromtimestamp(other_tx['status_updated']/1000, pytz.utc)
                        if 0 < (drop_time - add_time).total_seconds() < THRESHOLD_SECONDS:
                            dropped = True
                            break

            if not dropped:
                pickups.setdefault(roster_id, []).append(player_id)
    return pickups

def format_summary():

    week = get_week_by_sunday_cutoff(START_DATE)
    transactions = fetch_transactions(LEAGUE_ID, week)
    valid_pickups = filter_valid_pickups(transactions)
    roster_map = get_roster_map(LEAGUE_ID)
    player_map = get_player_map()

    lines = [f"🏀 Weekly Pickup Summary (Week {week})\n"]
    for roster_id, player_ids in valid_pickups.items():
        team = roster_map.get(roster_id, {})
        count = len(player_ids)
        status = "✅" if count <= 4 else "❌"
        lines.append(f"{team} — {count} pickup(s) {status}")
        for pid in player_ids:
            pname = player_map.get(pid, "Unknown Player")
            lines.append(f"• {pname}")
        lines.append("")  # blank line between teams
    return "\n".join(lines)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    
    summary = format_summary()

    channel = discord.utils.get(client.get_all_channels(), name="general")
    if channel:
        await channel.send(summary)

@client.event
async def on_message(message):

    if message.author == client.user:
        return

    if message.content == "!pickups":
        # Run pickup logic and send summary
        summary = format_summary()
        await message.channel.send(summary)

client.run(TOKEN)

# def main():

#     roster_map = get_roster_map(LEAGUE_ID)

#     current_week = get_week_by_sunday_cutoff(START_DATE)

#     transactions = fetch_transactions(LEAGUE_ID, current_week)
#     valid_pickups = filter_valid_pickups(transactions)

#     print("🏀 Weekly Pickup Summary:")
#     for roster_id, players in valid_pickups.items():
#         count = len(players)
#         status = "✅" if count <= 4 else "❌"
#         print(f"{roster_map[roster_id]}: {count} pickups {status}")

# if __name__ == "__main__":
#     main()