import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import tracker


def test_log_pickup_appends_weekly_csv_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "PICKUP_LOG_DIR", str(tmp_path))
    pickup_time = tracker.SEASON_OPENER

    assert tracker.log_pickup(
        "transaction-1", pickup_time, 7, "Team A", "player-1", "Player One"
    )
    assert not tracker.log_pickup(
        "transaction-1", pickup_time, 7, "Team A", "player-1", "Player One"
    )
    assert tracker.log_pickup(
        "transaction-2", pickup_time, 7, "Team A", "player-2", "Player Two"
    )

    log_path = tmp_path / "2026-27" / "pickups_week_01.csv"
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["transaction_id"] == "transaction-1"
    assert rows[0]["team"] == "Team A"
    assert rows[1]["player"] == "Player Two"

    week0_time = tracker.WEEK0_START
    assert tracker.log_pickup(
        "transaction-0", week0_time, 7, "Team A", "player-0", "Player Zero"
    )
    week0_log = tmp_path / "2026-27" / "pickups_week_00.csv"
    with week0_log.open("r", newline="", encoding="utf-8") as handle:
        week0_rows = list(csv.DictReader(handle))
    assert len(week0_rows) == 1
    assert week0_rows[0]["week"] == "0"


def test_tracking_period_separates_week_zero_from_week_one():
    assert tracker.get_tracking_period(tracker.WEEK0_START - timedelta(seconds=1)) is None
    assert tracker.get_tracking_period(tracker.WEEK0_START)[0] == 0
    assert tracker.get_tracking_period(tracker.SEASON_OPENER - timedelta(seconds=1))[0] == 0
    assert tracker.get_tracking_period(tracker.SEASON_OPENER)[0] == 1
    assert tracker.get_tracking_period(tracker.FIRST_REGULAR_WEEK_START)[0] == 2


def test_regular_week_cutoff_tracks_eastern_dst():
    dst_cutoff = datetime(2026, 11, 2, 8, tzinfo=timezone.utc)
    week, week_start, _ = tracker.get_tracking_period(dst_cutoff)

    assert week == 3
    assert week_start.isoformat() == "2026-11-02T03:00:00-05:00"


def test_sleeper_api_week_is_independent_of_tracker_week(monkeypatch):
    monkeypatch.setattr(tracker, "fetch_sleeper_week", lambda: {"week": 9})

    assert tracker.get_sleeper_api_week(0) == 9


def test_early_tracking_periods_always_query_sleeper_week_one(monkeypatch):
    monkeypatch.setattr(tracker, "fetch_sleeper_week", lambda: {"week": 3})

    assert tracker.get_sleeper_api_weeks(0) == [1, 2, 3, 4]
    assert tracker.get_sleeper_api_weeks(1) == [1, 2, 3, 4]
    assert tracker.get_sleeper_api_weeks(2) == [1, 2, 3, 4]


def test_scanner_accepts_current_and_previous_tracking_periods():
    previous_week = tracker.get_current_or_previous_period(
        tracker.WEEK0_START, current_week=1
    )
    current_week = tracker.get_current_or_previous_period(
        tracker.SEASON_OPENER, current_week=1
    )

    assert previous_week[0] == 0
    assert current_week[0] == 1
    assert tracker.get_current_or_previous_period(
        tracker.FIRST_REGULAR_WEEK_START, current_week=1
    ) is None
    assert tracker.get_current_or_previous_period(
        tracker.WEEK0_START, current_week=2
    ) is None


def test_late_pickup_updates_its_week_not_active_week(monkeypatch):
    monkeypatch.setattr(tracker, "pickup_counts", defaultdict(list))
    monkeypatch.setattr(
        tracker,
        "pickup_counts_by_week",
        defaultdict(lambda: defaultdict(list)),
    )
    monkeypatch.setattr(tracker, "get_tracking_period", lambda now=None: (1, None, None))

    assert tracker.record_pickup(0, "Team A", "Late Week 0 Pickup") == 1
    assert tracker.pickup_counts_by_week[0]["Team A"] == ["Late Week 0 Pickup"]
    assert tracker.pickup_counts == {}

    assert tracker.record_pickup(1, "Team A", "Week 1 Pickup") == 1
    assert tracker.pickup_counts["Team A"] == ["Week 1 Pickup"]
