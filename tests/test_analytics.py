from datetime import timedelta, timezone
import json

import analytics
import tracker


def test_season_data_files_are_created_and_initialized(tmp_path, monkeypatch):
    monkeypatch.setattr(analytics, "SEASON_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(analytics, "WEEKLY_DATA_FILE", str(tmp_path / "weekly_data.json"))
    monkeypatch.setattr(analytics, "CACHE_FILE", str(tmp_path / "matchups_cache.json"))

    analytics.ensure_season_data_files()

    assert json.loads((tmp_path / "weekly_data.json").read_text()) == {}
    assert json.loads((tmp_path / "matchups_cache.json").read_text()) == {}


def test_season_data_files_are_not_overwritten(tmp_path, monkeypatch):
    weekly_data_file = tmp_path / "weekly_data.json"
    cache_file = tmp_path / "matchups_cache.json"
    weekly_data_file.write_text('{"existing": true}')
    cache_file.write_text('{"cached": true}')
    monkeypatch.setattr(analytics, "SEASON_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(analytics, "WEEKLY_DATA_FILE", str(weekly_data_file))
    monkeypatch.setattr(analytics, "CACHE_FILE", str(cache_file))

    analytics.ensure_season_data_files()

    assert json.loads(weekly_data_file.read_text()) == {"existing": True}
    assert json.loads(cache_file.read_text()) == {"cached": True}


def test_initialize_team_history_creates_all_zero_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(analytics, "WEEKLY_DATA_FILE", str(tmp_path / "weekly_data.json"))
    team_history = {"1": {"1": {"pf": 12.5}}}

    analytics.initialize_team_history(team_history, 2, ["1", "2"])

    assert team_history == {
        "1": {
            "1": {"pf": 12.5, "pa": 0},
            "2": {"pf": 0, "pa": 0},
        },
        "2": {
            "1": {"pf": 0, "pa": 0},
            "2": {"pf": 0, "pa": 0},
        },
    }
    assert json.loads((tmp_path / "weekly_data.json").read_text()) == team_history


def test_update_week_data_records_pf_and_pa_deltas(monkeypatch, tmp_path):
    monkeypatch.setattr(analytics, "WEEKLY_DATA_FILE", str(tmp_path / "weekly_data.json"))
    team_history = {
        "1": {
            "1": {"pf": 100, "pa": 90},
            "2": {"pf": 90, "pa": 100},
        }
    }

    analytics.update_week_data(
        "2",
        {
            "1": (240.5, 210.25),
            "2": (200.25, 230.5),
        },
        team_history,
    )

    assert team_history["2"] == {
        "1": {"pf": 140.5, "pa": 120.25},
        "2": {"pf": 110.25, "pa": 130.5},
    }
    assert json.loads((tmp_path / "weekly_data.json").read_text()) == team_history


def test_roster_map_includes_sleeper_pf_and_pa_decimal_fields(monkeypatch):
    payloads = {
        "users": [{"user_id": "owner-1", "display_name": "Team 1"}],
        "rosters": [{
            "owner_id": "owner-1",
            "roster_id": 1,
            "settings": {
                "fpts": 757,
                "fpts_decimal": 50,
                "fpts_against": 769,
                "fpts_against_decimal": 25,
            },
            "metadata": {"record": "2-0"},
        }],
    }

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

    monkeypatch.setattr(
        analytics,
        "fetch",
        lambda url: FakeResponse(payloads["users" if url.endswith("/users") else "rosters"]),
    )

    _, scores, _ = analytics.get_roster_map("league")

    assert scores["1"] == (757.5, 769.25)


def test_rating_maps_handle_identical_pf_scores():
    ratings, team_ratings = analytics.get_rating_maps(
        1,
        {"1": [0.0], "2": [0.0]},
    )

    assert ratings == {"1": 0.0, "2": 0.0}
    assert team_ratings == {"1": 0.0, "2": 0.0}


def test_rating_maps_average_weekly_pf_z_scores():
    ratings, team_ratings = analytics.get_rating_maps(
        2,
        {"1": [1.0, 1.0], "2": [-1.0, -1.0]},
    )

    assert ratings == {"1": 1.0, "2": -1.0}
    assert team_ratings == ratings


def test_momentum_averages_latest_three_pf_z_scores():
    momentum = analytics.get_momentum_scores(
        {
            "1": [2.0, 1.0, 0.0, -1.0],
            "2": [-1.0, 0.0, 1.0, 2.0],
        },
    )

    assert momentum == {
        "1": 0.0,
        "2": 1.0,
    }


def test_current_week_uses_tracker_period_boundaries():
    assert analytics.get_current_week(tracker.WEEK0_START - timedelta(seconds=1)) == 1
    assert analytics.get_current_week(tracker.WEEK0_START) == 1
    assert analytics.get_current_week(tracker.SEASON_OPENER - timedelta(seconds=1)) == 1
    assert analytics.get_current_week(tracker.SEASON_OPENER) == 1
    assert analytics.get_current_week(tracker.FIRST_REGULAR_WEEK_START) == 2


def test_current_week_follows_eastern_daylight_saving_time():
    dst_cutoff = tracker.FIRST_REGULAR_WEEK_START + timedelta(days=7)
    now = dst_cutoff.astimezone(timezone.utc)

    assert analytics.get_current_week(now) == 3


def test_current_week_is_capped_after_regular_season():
    after_regular_season = tracker.FIRST_REGULAR_WEEK_START + timedelta(days=7 * 21)

    assert analytics.get_current_week(after_regular_season) == analytics.LAST_WEEK + 1


def test_week_clamps_at_regular_season_end():
    assert analytics.clamp_week(21, analytics.REGULAR_SEASON_LAST_WEEK) == 21
    assert analytics.clamp_week(22, analytics.REGULAR_SEASON_LAST_WEEK) == 21


def test_record_calculation_preserves_original_record():
    assert analytics.format_record(17, 21) == "17-4"
    assert analytics.format_record(10, 21) == "10-11"
    assert analytics.format_record(31, 42) == "31-11"


def test_seed_map_reports_original_and_median_seed_delta():
    actual_seeds = analytics.get_seed_map({"team_a": 8, "team_b": 6, "team_c": 4})
    median_seeds = analytics.get_seed_map({"team_a": 6, "team_b": 8, "team_c": 4})

    assert actual_seeds["team_a"] == 1
    assert median_seeds["team_a"] == 2
    assert actual_seeds["team_a"] - median_seeds["team_a"] == -1
    assert actual_seeds["team_b"] - median_seeds["team_b"] == 1


def test_seed_delta_summary_counts_improvements():
    improved, average_delta = analytics.get_seed_delta_summary(
        {"team_a": 1, "team_b": 2, "team_c": 3},
        {"team_a": 2, "team_b": 1, "team_c": 3},
    )

    assert improved == 1
    assert average_delta == 0


def test_zero_variance_scores_do_not_produce_nan(tmp_path):
    team_history = {
        "1": {"team_a": {"pf": 100}, "team_b": {"pf": 100}},
        "2": {"team_a": {"pf": 100}, "team_b": {"pf": 100}},
    }

    weekly_z, ewtd = analytics.get_weekly_arrays(team_history, 1)

    assert weekly_z["team_a"][0] == 0.0
    assert weekly_z["team_b"][0] == 0.0
    assert ewtd["team_a"][0] == 0.0
