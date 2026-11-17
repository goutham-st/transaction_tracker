# Sleeper Fantasy Basketball Discord Bot and Analytics Toolkit

## Overview
This repository contains a Discord bot for monitoring Sleeper fantasy basketball pickups and a companion analytics script for league analysis. The bot watches transactions, filters out very short-lived pickups, and posts alerts to a Discord channel when a roster adds a player that appears to be a meaningful pickup.

## Features
- **!pickups Command**: Displays a formatted summary of pickups tracked for the current week
- **Quick Drop Filter**: Ignores pickups where the player is dropped within 5 minutes
- **Weekly Pickup Limits**: Tracks pickups per team and sends alerts when a roster reaches or exceeds its weekly limit of 3 pickups
- **Analytics CLI**: Includes weekly data updates, power rankings, season simulations, current-week estimates, and other league insights
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
- Pickup schedule: Season-specific tracking windows are defined by `WEEK0_START` and `SEASON_OPENER` in `tracker.py`
- `DROP_THRESHOLD_MIN`: 5 minutes; pickups dropped within this window are ignored
- `CHECK_INTERVAL_SEC`: 60 seconds; how often the tracker polls Sleeper
- Analytics data files are stored locally under the season start year (for example, `2026/weekly_data.json` and `2026/matchups_cache.json`) and are excluded from Git. Missing completed-week entries are initialized with `pf`, `pa`, and `mpf` set to `0`; `--update-week` fills them from roster scoring totals.

## Discord Commands
- `!pickups` - Display the weekly pickup summary with the roster and the players it added

## Analytics Commands
The analytics script supports several CLI modes, including:
- `--update-week` to refresh weekly data
- `--print-power-rankings` to print current rankings
- `--plot-luck` to visualize rating vs. luck
- `--simulate-season` to run a season simulation
- `--playoff-mode` to explicitly include playoff outcome simulation
- `--current-week-probs` to show model-estimated probabilities for current matchups
- `--momentum`, `--parity`, `--strength-of-schedule`, and `--versus-median` for additional analysis views

Backtesting is currently disabled. Power rankings combine per-week standardized scores with equal week weighting: 65% PF and 35% Max PF. Momentum uses the same PF/Max PF blend over the latest three completed weeks. Max PF is the best possible lineup score and provides a roster-potential signal alongside points actually scored.

Win probabilities and season simulations are estimates based on those ratings; the probability model has not been calibrated against historical outcomes.

## Technical Details
- **Language**: Python 3.11
- **Dependencies**: `discord.py`, `requests`, `python-dotenv`, `numpy`, `scipy`, `matplotlib`
- **Data Files**: `weekly_data.json` and `matchups_cache.json` are stored in a folder named for the season start year (for example, `2026/`)
- **Deployment**: Any host that can run Python and has the required environment variables configured

## Setup
Copy `.env.example` to `.env` and set your Discord bot token, Sleeper league ID, and Discord channel name. Keep `.env` private; it is ignored by Git.

## Linux / WSL Development Setup
Create a virtual environment and install the project dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run tests with `python -m pytest`.

In VS Code, select `.venv/bin/python` as the workspace interpreter. If WSL reports that `venv` or `ensurepip` is unavailable, install the matching OS package (for example, `python3.12-venv` on Ubuntu with Python 3.12), then recreate the environment.

Run an analytics mode by passing its option, for example:

```bash
python analytics.py --versus-median
```

In VS Code, the **Analytics: Versus Median** launch configuration runs the same command using the workspace virtual environment.
