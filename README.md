# Sleeper Fantasy Basketball Discord Bot

## Overview
A Discord bot that monitors Sleeper fantasy basketball league for free agent pickups and automatically posts alerts to a Discord channel.

## Features
- **!pickups Command**: Displays a formatted summary of all pickups for the current week
- **Quick Drop Filter**: Ignores pickups where the player is dropped within X minutes (streaming detection)
- **Team Pickup Limits**: Log pickups per team to ensure weekly limit is not exceeded
- **Error Resilience**: HTTP timeout protection and graceful error handling

## Project Architecture

### Core Components
- `main.py`: Main bot application with Discord client and Sleeper API integration
- Environment variables
  - `DISCORD_TOKEN`: Discord bot authentication token
  - `LEAGUE_ID`: Sleeper league identifier
  - 

### Key Configuration
- `START_DATE`: datetime(2025, 10, 20, 8, tzinfo=timezone.utc)
- `THRESHOLD_SECONDS`: 

## Discord Commands
- `!pickups` - Display weekly pickup summary with player names and counts per team

## Technical Details
- **Language**: Python 3.11
- **Dependencies**: discord.py, requests
- **Deployment**: Google Cloud
- **API Integration**: Sleeper API (players, users, rosters, transactions endpoints)

## User Preferences
- Bot responds to !pickups command in any channel
- Filters out "accidental" pickups (players added and dropped within X minutes)

