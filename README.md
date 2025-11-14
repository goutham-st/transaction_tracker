# Sleeper Fantasy Basketball Discord Bot

## Overview
A Discord bot that monitors Sleeper fantasy basketball league for free agent pickups and automatically posts alerts to a Discord channel.

## Features
- **!pickups Command**: Displays a formatted summary of all pickups for the current week
- **Quick Drop Filter**: Ignores pickups where the player is dropped within 5 minutes
- **Team Pickup Limits**: Log pickups per team to ensure weekly limit is not exceeded
- **Error Resilience**: HTTP timeout protection and graceful error handling

## Project Architecture

### Core Components
- `tracker.py`: Main bot application with Discord client and Sleeper API integration
- Environment variables
  - `DISCORD_TOKEN`: Discord bot authentication token
  - `LEAGUE_ID`: Sleeper league identifier
  - `CHANNEL_NAME`: Channel to which alerts are sent and commands are read from

### Key Configuration
- `START_DATE`: October 20, 2025 8 AM UTC
- `DROP_THRESHOLD_MIN`: A player added and dropped within this threshold will not count towards the limit
- `CHECK_INTERVAL_SEC`: How often the Sleeper API is polled

## Discord Commands
- `!pickups` - Display weekly pickup summary with player names and counts per team

## Technical Details
- **Language**: Python 3.11
- **Dependencies**: discord.py, requests
- **Deployment**: Google Cloud
- **API Integration**: Sleeper API (players, users, rosters, transactions endpoints)

