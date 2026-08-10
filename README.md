# Sleeper Fantasy Basketball Discord Bot and Analytics Toolkit

## Overview
This repository contains a Discord bot for monitoring Sleeper fantasy basketball pickups and a companion analytics script for league analysis. The bot watches transactions, filters out very short-lived pickups, and posts alerts to a Discord channel when a roster adds a player that appears to be a meaningful pickup.

## Features
- **!pickups Command**: Displays a formatted summary of pickups tracked for the current week
- **Quick Drop Filter**: Ignores pickups where the player is dropped within 5 minutes
- **Weekly Pickup Limits**: Tracks pickups per team and sends alerts when a roster reaches or exceeds its weekly limit of 4 pickups
- **Analytics CLI**: Includes a separate analytics workflow for weekly data updates, power rankings, simulations, backtesting, and other league insights
- **Error Resilience**: Uses timeout protection and graceful error handling around Sleeper API calls

## Project Architecture

### Core Components
- `tracker.py`: Main Discord bot application with Sleeper API integration, transaction monitoring, and pickup alerting
- `analytics.py`: League analytics tool for weekly data updates, simulations, rankings, and matchup-based analysis

### Environment Variables
- `DISCORD_TOKEN`: Discord bot authentication token
- `LEAGUE_ID`: Sleeper league identifier
- `CHANNEL_NAME`: Discord channel where alerts are sent and commands are read from

### Key Configuration
- `START_DATE`: October 20, 2025 07:00 UTC
- `DROP_THRESHOLD_MIN`: 5 minutes; pickups dropped within this window are ignored
- `CHECK_INTERVAL_SEC`: 60 seconds; how often the tracker polls Sleeper
- `CACHE_FILE`: `matchups_cache.json` used by the analytics script

## Discord Commands
- `!pickups` - Display the weekly pickup summary with the roster and the players it added

## Analytics Commands
The analytics script supports several CLI modes, including:
- `--update-week` to refresh weekly data
- `--print-power-rankings` to print current rankings
- `--plot-luck` to visualize rating vs. luck
- `--simulate-season` to run a season simulation
- `--backtesting` to evaluate forecast accuracy
- `--momentum`, `--parity`, `--strength-of-schedule`, and `--versus-median` for additional analysis views

## Technical Details
- **Language**: Python 3.11
- **Dependencies**: `discord.py`, `requests`, `python-dotenv`, `numpy`, `scipy`, `matplotlib`
- **Data Files**: `weekly_data.json`, `matchups_cache.json`
- **Deployment**: Any host that can run Python and has the required environment variables configured

