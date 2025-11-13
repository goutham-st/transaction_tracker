import discord
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import asyncio
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

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")
START_DATE = datetime(2025, 10, 20, 8, tzinfo=timezone.utc)
DROP_THRESHOLD_MIN = 5
CHECK_INTERVAL_SEC = 60 

seen_transactions = {}
pickup_counts = defaultdict(list)
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
    if now.weekday() == 0 and now.hour < 8:
        now -= timedelta(days=1)

    days_since_monday = (now.weekday() + 7) % 7

    # Find the most recent end of the week
    last_monday = now - timedelta(days=days_since_monday)
    last_monday = last_monday.replace(hour=8,minute=0,second=0,microsecond=0)
    delta = last_monday - start_date
    return delta.days // 7 + 1

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

def format_summary(week):
    lines = [f"🏀 Weekly Pickup Summary (Week {week})\n"]
    for roster_id, player_names in pickup_counts.items():
        count = len(player_names)
        status = "✅" if count <= 4 else "❌"
        lines.append(f"{roster_id} — {count} pickup(s) {status}")
        for pname in player_names:
            lines.append(f"• {pname}")
        lines.append("")
    return "\n".join(lines)

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
    current_week = get_week_by_cutoff(START_DATE)
    
    while not client.is_closed():
        try:            
            week = get_week_by_cutoff(START_DATE)
            
            if week != current_week:
                channel.send(format_summary(current_week))

                cutoff = datetime.now(timezone.utc) - timedelta(days=14)
                
                seen_transactions = {
                    tx_id: t for tx_id, t in seen_transactions.items()
                    if t >= cutoff
                }

                pickup_counts.clear()
                current_week = week
                refresh_caches()
                print(f"New week detected. Resetting tracking for week {week}")
            
            window_start = START_DATE + timedelta(weeks=week-1)
            window_end = START_DATE + timedelta(weeks=week)

            if week > 1:
                last_week_transactions = fetch_transactions(LEAGUE_ID, week-1)
                for tx in reversed(last_week_transactions):

                    tx_id = tx['transaction_id']

                    if tx_id in seen_transactions:
                        continue

                    if tx['type'] == 'waiver' and tx['status'] == 'complete':
                        add_time = datetime.fromtimestamp(tx['status_updated']/1000, timezone.utc)
                        if window_start <= add_time <= window_end:
                            [(player_id, roster_id)] = list(tx['adds'].items())
                            asyncio.create_task(check_pickup(player_id,roster_id,add_time,channel))
                            seen_transactions[tx_id] = add_time

            curr_transactions = fetch_transactions(LEAGUE_ID, week)
            
            for tx in reversed(curr_transactions):
                tx_id = tx['transaction_id']

                if tx_id in seen_transactions:
                    continue

                if tx['type'] not in ['free_agent', 'waiver'] or tx['status'] != 'complete':
                    continue

                if not tx['adds']:
                    continue

                add_time = datetime.fromtimestamp(tx['status_updated']/1000, timezone.utc)
                
                if window_start <= add_time <= window_end:
                    [(player_id, roster_id)] = list(tx['adds'].items())
                    asyncio.create_task(check_pickup(player_id,roster_id,add_time, channel))
                    seen_transactions[tx_id] = add_time
            
        except Exception as e:
            print(f"Error in monitor loop: {e}")
        
        await asyncio.sleep(CHECK_INTERVAL_SEC)

async def check_pickup(player_id, roster_id, pickup_time, channel):
    try:
        now = datetime.now(timezone.utc)
        delay = 0 if DEBUG_MODE else max(0, (pickup_time+timedelta(minutes=DROP_THRESHOLD_MIN)-now).total_seconds())
        
        if delay > 0:
            await asyncio.sleep(delay)

        week = get_week_by_cutoff(START_DATE,pickup_time)
        transctions = fetch_transactions(LEAGUE_ID, week)

        team = roster_map_cache.get(roster_id, "Unknown")
        player = player_map_cache.get(player_id, "Unknown Player")

        if was_dropped(player_id, roster_id, pickup_time, transctions):
            print(f"Player {player} was dropped within {DROP_THRESHOLD_MIN} minutes by team {team}. No action needed.")
            return

        pickup_counts[team].append(player)
        pickup_number = len(pickup_counts[team])

        if pickup_number > 4:
            suffix = "🚨🔥 **Limit surpassed!**"
        elif pickup_number == 4:
            suffix = "🚨 **Limit reached!**"
        elif pickup_number == 3:
            suffix = "⚠️ One pickup left this week."
        else:
            suffix = ""

        message = f"🏀 **New Pickup Alert!**\n{team} added **{player}** \nPickup #{pickup_number}/4 this week\n{suffix}"
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
        summary = format_summary(get_week_by_cutoff(START_DATE))
        await message.channel.send(summary)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
