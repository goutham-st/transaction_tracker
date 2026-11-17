import discord
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os
import asyncio
import csv
from asyncio import Semaphore
from collections import defaultdict
import sys

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
LEAGUE_ID = os.getenv("LEAGUE_ID")
CHANNEL_NAME = os.getenv("CHANNEL_NAME")

if not DISCORD_TOKEN:
    print("ERROR: DISCORD_TOKEN environment variable is not set!")
    sys.exit(1)

if not LEAGUE_ID:
    print("ERROR: LEAGUE_ID environment variable is not set!")
    sys.exit(1)

if not CHANNEL_NAME:
    print("ERROR: CHANNEL_NAME environment variable is not set!")
    sys.exit(1)

EASTERN = ZoneInfo("America/New_York")
WEEK0_START = datetime(2026, 10, 12, 3, tzinfo=EASTERN)
SEASON_OPENER = datetime(2026, 10, 20, 15, tzinfo=EASTERN)
DROP_THRESHOLD_MIN = 5
CHECK_INTERVAL_SEC = 60 
PICKUP_MAX = 3
STARTUP = datetime.now(timezone.utc) 
PICKUP_LOG_DIR = os.path.join(os.path.dirname(__file__), "pickup_logs")

seen_transactions = {}
pickup_counts = defaultdict(list)
pickup_counts_by_week = defaultdict(lambda: defaultdict(list))
player_map_cache = {}
roster_map_cache = {}
send_semaphore = Semaphore()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

async def safe_send(channel, message):
    async with send_semaphore:
        await channel.send(message)
        await asyncio.sleep(1.1)  # space out sends to avoid 429

def get_player_map():
    url = "https://api.sleeper.app/v1/players/nba"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        player_map = {
            pid: info.get("full_name", "Unknown")
            for pid, info in data.items()
            if "full_name" in info
        }
        return player_map
    except requests.RequestException as e:
        print(f"Error fetching player map: {e}")
        return {}

def get_week_by_cutoff(start_date, now=None):

    if now is None:
        now = datetime.now(timezone.utc)

    # If it's Monday before 8 AM ET, treat it as part of the previous week
    # TODO Don't hardcode DST
    if now.weekday() == 0 and now.hour < 7:
        now -= timedelta(days=1)

    days_since_monday = (now.weekday() + 7) % 7

    # Find the most recent end of the week
    last_monday = now - timedelta(days=days_since_monday)
    last_monday = last_monday.replace(hour=7,minute=0,second=0,microsecond=0)
    delta = last_monday - start_date
    return delta.days // 7 + 1

def get_next_regular_cutoff(after):
    local_after = after.astimezone(EASTERN)
    days_until_monday = (7 - local_after.weekday()) % 7
    cutoff = (local_after + timedelta(days=days_until_monday)).replace(
        hour=3, minute=0, second=0, microsecond=0
    )
    if cutoff <= local_after:
        cutoff += timedelta(days=7)
    return cutoff

FIRST_REGULAR_WEEK_START = get_next_regular_cutoff(SEASON_OPENER)

def get_tracking_period(now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    now = now.astimezone(EASTERN)

    if now < WEEK0_START:
        return None
    if now < SEASON_OPENER:
        return 0, WEEK0_START, SEASON_OPENER
    if now < FIRST_REGULAR_WEEK_START:
        return 1, SEASON_OPENER, FIRST_REGULAR_WEEK_START

    week = 2
    week_start = FIRST_REGULAR_WEEK_START
    while now >= week_start + timedelta(days=7):
        week += 1
        week_start += timedelta(days=7)
    return week, week_start, week_start + timedelta(days=7)

def get_current_or_previous_period(timestamp, current_week):
    period = get_tracking_period(timestamp)
    if period is None or period[0] not in (current_week - 1, current_week):
        return None
    return period

def get_sleeper_api_week(fallback_week):
    state = fetch_sleeper_week()
    if isinstance(state, dict):
        week = state.get("week")
        if isinstance(week, int) and week > 0:
            return week
    return max(1, fallback_week)

def get_sleeper_api_weeks(tracking_week):
    sleeper_week = get_sleeper_api_week(tracking_week)
    api_weeks = set(range(max(1, sleeper_week - 1), sleeper_week + 2))
    api_weeks.add(max(1, tracking_week - 1))
    return sorted(api_weeks)

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
        for roster in rosters:
            owner_id = roster.get('owner_id')
            roster_id = roster.get('roster_id')
            if owner_id in user_map:
                roster_map[roster_id] = user_map[owner_id]['display_name']
            else:
                roster_map[roster_id] = 'Unknown'

        return roster_map
    except requests.RequestException as e:
        print(f"Error fetching roster map: {e}")
        return {}

def fetch_transactions(league_id, week):
    url = f"https://api.sleeper.app/v1/league/{league_id}/transactions/{week}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching transactions: {e}")
        return []
    
def fetch_sleeper_week():
    url = f"https://api.sleeper.app/v1/state/nba"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching transactions: {e}")
        return []

def log_pickup(transaction_id, pickup_time, roster_id, team, player_id, player):
    period = get_tracking_period(pickup_time)
    if period is None:
        return False
    week, week_start, _ = period
    season_name = f"{SEASON_OPENER.year}-{(SEASON_OPENER.year + 1) % 100:02d}"
    season_log_dir = os.path.join(PICKUP_LOG_DIR, season_name)
    os.makedirs(season_log_dir, exist_ok=True)
    log_path = os.path.join(season_log_dir, f"pickups_week_{week:02d}.csv")

    if os.path.exists(log_path):
        with open(log_path, "r", newline="", encoding="utf-8") as handle:
            if any(row.get("transaction_id") == transaction_id for row in csv.DictReader(handle)):
                return False

    fieldnames = [
        "transaction_id",
        "pickup_time_utc",
        "week",
        "week_start_utc",
        "roster_id",
        "team",
        "player_id",
        "player",
    ]
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "transaction_id": transaction_id,
            "pickup_time_utc": pickup_time.isoformat(),
            "week": week,
            "week_start_utc": week_start.astimezone(timezone.utc).isoformat(),
            "roster_id": roster_id,
            "team": team,
            "player_id": player_id,
            "player": player,
        })
    return True

def format_summary(week):
    lines = [f"🏀 Weekly Pickup Summary (Week {week})\n"]
    for roster_id, player_names in pickup_counts.items():
        count = len(player_names)
        status = "✅" if count <= PICKUP_MAX else "❌"
        lines.append(f"{roster_id} — {count} pickup(s) {status}")
        for pname in player_names:
            lines.append(f"• {pname}")
        lines.append("")
    return "\n".join(lines)

def record_pickup(week, team, player):
    week_counts = pickup_counts_by_week[week]
    week_counts[team].append(player)

    current_period = get_tracking_period()
    if current_period is not None and current_period[0] == week:
        pickup_counts[team].append(player)

    return len(week_counts[team])

def refresh_caches():
    global player_map_cache, roster_map_cache
    print("Refreshing player and roster caches...")
    player_map_cache = get_player_map()
    roster_map_cache = get_roster_map(LEAGUE_ID)
    print(f"Caches refreshed. Players: {len(player_map_cache)}, Rosters: {len(roster_map_cache)}")

async def monitor_pickups():
    global seen_transactions
    await client.wait_until_ready()
    channel = discord.utils.get(client.get_all_channels(), name=CHANNEL_NAME)
    
    if not channel:
        print(f"Error: Could not find {CHANNEL_NAME} channel")
        return
    
    print(f"Monitoring pickups in #{channel.name}")
    refresh_caches()
    current_period = get_tracking_period()
    current_week = current_period[0] if current_period else None
    
    while not client.is_closed():
        try:            
            period = get_tracking_period()
            if period is None:
                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue
            week = period[0]
            
            if current_week is None:
                current_week = week
                refresh_caches()
            elif week != current_week:

                cutoff = datetime.now(timezone.utc) - timedelta(days=14)
                
                seen_transactions = {
                    tx_id: t for tx_id, t in seen_transactions.items()
                    if t >= cutoff
                }

                pickup_counts.clear()
                await safe_send(channel, f"Week {current_week} complete. Starting tracking for Week {week}.")
                current_week = week
                refresh_caches()
            
            curr_transactions = []
            api_weeks = get_sleeper_api_weeks(week)
            for api_week in api_weeks:
                curr_transactions.extend(
                    (tx, api_week) for tx in fetch_transactions(LEAGUE_ID, api_week)
                )
            
            for tx, api_week in reversed(curr_transactions):

                if 'transaction_id' not in tx:
                    continue

                tx_id = tx['transaction_id']

                if tx_id in seen_transactions:
                    continue

                if tx['type'] not in ['free_agent', 'waiver'] or tx['status'] != 'complete':
                    continue

                if not tx['adds']:
                    continue

                add_time = datetime.fromtimestamp(tx['status_updated']/1000, timezone.utc)
                
                transaction_period = get_current_or_previous_period(add_time, week)
                if transaction_period is not None:
                    [(player_id, roster_id)] = list(tx['adds'].items())
                    asyncio.create_task(check_pickup(
                        tx_id, player_id, roster_id, add_time, api_week, channel
                    ))
                    seen_transactions[tx_id] = add_time
            
        except Exception as e:
            print(f"Error in monitor loop: {e}")
        
        await asyncio.sleep(CHECK_INTERVAL_SEC)

async def check_pickup(transaction_id, player_id, roster_id, pickup_time, api_week, channel):
    try:
        now = datetime.now(timezone.utc)

        delay = max(0, (pickup_time+timedelta(minutes=DROP_THRESHOLD_MIN)-now).total_seconds())
        if delay > 0:
            await asyncio.sleep(delay)

        period = get_tracking_period(pickup_time)
        if period is None:
            return
        week = period[0]
        transctions = fetch_transactions(LEAGUE_ID, api_week)

        team = roster_map_cache.get(roster_id, "Unknown")
        player = player_map_cache.get(player_id, "Unknown Player")

        if was_dropped(player_id, roster_id, pickup_time, transctions):
            print(f"Player {player} was dropped within {DROP_THRESHOLD_MIN} minutes by team {team}. No action needed.")
            return

        try:
            log_pickup(transaction_id, pickup_time, roster_id, team, player_id, player)
        except (OSError, csv.Error, UnicodeError) as exc:
            print(f"Unable to log pickup {transaction_id}: {exc}")

        pickup_number = record_pickup(week, team, player)

        if pickup_time >= STARTUP:
            if pickup_number > PICKUP_MAX:
                suffix = "💀 **Limit surpassed!**"
            elif pickup_number == PICKUP_MAX:
                suffix = "🚨 **Limit reached!**"
            elif pickup_number == PICKUP_MAX - 1:
                suffix = "⚠️ One pickup left this week."
            else:
                suffix = ""

            message = f"🏀 **New Pickup Alert!**\n{team} added **{player}** \nPickup #{pickup_number}/{PICKUP_MAX} this week (Week {week})\n{suffix}"
            await safe_send(channel, message)
    except Exception as e:
        print(f"Error in check_pickup: {e}")

def was_dropped(player_id, roster_id, pickup_time, transactions):
    # Query Sleeper API for transactions since `pickup_time`
    # Return True if player_id was dropped by same manager

    cutoff = pickup_time + timedelta(minutes=DROP_THRESHOLD_MIN)

    for tx in transactions:
        if tx['type'] not in ['free_agent', 'waiver']:
            continue
        if not tx['drops'] or player_id not in tx['drops']:
            continue
        if tx['roster_ids'][0] != roster_id:
            continue
        
        drop_time = datetime.fromtimestamp(tx['status_updated'] / 1000, timezone.utc)
        
        if pickup_time <= drop_time <= cutoff:
            return True
        
    return False 

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    client.loop.create_task(monitor_pickups())

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content == "!pickups":
        period = get_tracking_period()
        if period is None:
            await message.channel.send("Pickup tracking has not started yet.")
            return
        summary = format_summary(period[0])
        await message.channel.send(summary)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
