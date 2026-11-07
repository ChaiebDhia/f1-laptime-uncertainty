import urllib.error

import pytest

from f1lap import openf1
from f1lap.openf1 import build_raw, find_session, load_race_openf1, non_green_laps

SESSIONS = [
    {
        "session_key": 100,
        "year": 2024,
        "country_name": "Bahrain",
        "location": "Sakhir",
        "circuit_short_name": "Sakhir",
        "session_name": "Race",
    },
    {
        "session_key": 200,
        "year": 2024,
        "country_name": "Spain",
        "location": "Barcelona",
        "circuit_short_name": "Catalunya",
        "session_name": "Race",
    },
    {
        "session_key": 300,
        "year": 2024,
        "country_name": "United States",
        "location": "Austin",
        "circuit_short_name": "Austin",
        "session_name": "Race",
    },
    {
        "session_key": 301,
        "year": 2024,
        "country_name": "United States",
        "location": "Miami",
        "circuit_short_name": "Miami",
        "session_name": "Race",
    },
]


def _race_fixture(n_laps=30):
    """Two drivers; driver 1 pits end of lap 10 (medium -> hard on a used set)."""
    laps, stints, pits, rc = [], [], [], []
    for d in (1, 2):
        for lap in range(1, n_laps + 1):
            laps.append(
                {
                    "driver_number": d,
                    "lap_number": lap,
                    "lap_duration": 90.0 + 0.05 * lap + 0.1 * d,
                    "is_pit_out_lap": d == 1 and lap == 11,
                }
            )
    stints += [
        {
            "driver_number": 1,
            "stint_number": 1,
            "compound": "MEDIUM",
            "lap_start": 1,
            "lap_end": 10,
            "tyre_age_at_start": 0,
        },
        {
            "driver_number": 1,
            "stint_number": 2,
            "compound": "HARD",
            "lap_start": 11,
            "lap_end": n_laps,
            "tyre_age_at_start": 2,
        },
        {
            "driver_number": 2,
            "stint_number": 1,
            "compound": "SOFT",
            "lap_start": 1,
            "lap_end": n_laps,
            "tyre_age_at_start": 0,
        },
    ]
    pits.append({"driver_number": 1, "lap_number": 10})
    rc += [
        {"date": "2024-03-02T15:20:00", "lap_number": 5, "message": "SAFETY CAR DEPLOYED"},
        {"date": "2024-03-02T15:30:00", "lap_number": 6, "message": "SAFETY CAR IN THIS LAP"},
    ]
    drivers = [
        {"driver_number": 1, "name_acronym": "VER"},
        {"driver_number": 2, "name_acronym": "HAM"},
    ]
    return laps, stints, pits, drivers, rc


def test_find_session_matches_country_location_and_ignores_case(monkeypatch):
    monkeypatch.setattr(openf1, "get_json", lambda *a, **k: SESSIONS)
    assert find_session(2024, "Bahrain")["session_key"] == 100
    assert find_session(2024, "spain")["session_key"] == 200
    assert find_session(2024, "Sakhir")["session_key"] == 100
    assert find_session(2024, "Austin")["session_key"] == 300


def test_find_session_reports_ambiguous_and_unknown(monkeypatch):
    monkeypatch.setattr(openf1, "get_json", lambda *a, **k: SESSIONS)
    with pytest.raises(ValueError, match="Austin.*Miami|Miami.*Austin"):
        find_session(2024, "United States")
    with pytest.raises(ValueError, match="Available"):
        find_session(2024, "Atlantis")


def test_non_green_laps_safety_car_vsc_yellow_and_unfinished():
    msgs = [
        {"date": "1", "lap_number": 5, "message": "SAFETY CAR DEPLOYED"},
        {"date": "2", "lap_number": 7, "message": "SAFETY CAR IN THIS LAP"},
        {"date": "3", "lap_number": 12, "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": "4", "lap_number": 12, "message": "VIRTUAL SAFETY CAR ENDING"},
        {"date": "5", "lap_number": 15, "flag": "DOUBLE YELLOW", "message": "DOUBLE YELLOW"},
        {"date": "6", "lap_number": 20, "flag": "CLEAR", "message": "CLEAR IN TRACK SECTOR 3"},
    ]
    assert non_green_laps(msgs, 30) == {5, 6, 7, 8, 12, 13, 15}
    unfinished = [{"date": "1", "lap_number": 25, "message": "SAFETY CAR DEPLOYED"}]
    assert non_green_laps(unfinished, 30) == set(range(25, 31))


def test_build_raw_tyre_age_pit_laps_and_green_flags():
    laps, stints, pits, drivers, rc = _race_fixture()
    raw = build_raw("2024 Test", laps, stints, pits, drivers, rc)
    ver = raw[raw["driver"] == "VER"].set_index("lap")
    assert ver.loc[5, "tyre_age"] == 5  # new medium: age = lap
    assert ver.loc[12, "tyre_age"] == 2 + (12 - 11) + 1  # used hard set, 2 laps old at start
    assert bool(ver.loc[10, "pit"]) and bool(ver.loc[11, "pit"])  # in-lap and out-lap
    assert not bool(ver.loc[20, "pit"])
    assert not bool(ver.loc[6, "green"]) and bool(ver.loc[20, "green"])  # safety car window
    assert (raw[raw["driver"] == "HAM"]["compound"] == "SOFT").all()
    assert set(raw["driver"]) == {"VER", "HAM"}


def test_build_raw_rejects_schema_changes_with_a_clear_message():
    with pytest.raises(ValueError, match="lap_duration"):
        build_raw("x", [{"driver_number": 1, "lap_number": 1}], [{"driver_number": 1}], [], [], [])
    with pytest.raises(ValueError, match="no laps"):
        build_raw("x", [], [], [], [], [])


def test_load_race_openf1_end_to_end_with_cache(monkeypatch, tmp_path):
    laps, stints, pits, drivers, rc = _race_fixture(n_laps=40)
    table = {
        "sessions": SESSIONS,
        "laps": laps,
        "stints": stints,
        "pit": pits,
        "drivers": drivers,
        "race_control": rc,
    }
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        return table[url.split("/v1/")[1].split("?")[0]]

    monkeypatch.setattr(openf1, "_fetch", fake_fetch)
    df, log = load_race_openf1(2024, "Bahrain", cache_dir=str(tmp_path))
    assert not df.empty and df["lap_time"].notna().all()
    assert not df["lap"].isin([1, 5, 6, 7]).any()  # lap 1 and the safety-car window
    ver = df[df["driver"] == "VER"]
    assert not ver["lap"].isin([10, 11]).any()  # in-lap and out-lap removed
    assert {10, 11} <= set(df[df["driver"] == "HAM"]["lap"])  # driver who did not pit keeps them
    assert log["removed"].sum() > 0
    n_first = len(calls)
    load_race_openf1(2024, "Bahrain", cache_dir=str(tmp_path))
    assert len(calls) == n_first  # second run served from cache


def test_fetch_retries_rate_limits_and_treats_404_as_empty(monkeypatch):
    monkeypatch.setattr(openf1.time, "sleep", lambda s: None)
    answers = iter([urllib.error.HTTPError("u", 429, "slow down", {}, None), "ok"])

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'[{"a": 1}]'

    def fake_urlopen(req, timeout):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return Resp()

    monkeypatch.setattr(openf1.urllib.request, "urlopen", fake_urlopen)
    assert openf1._fetch("http://x") == [{"a": 1}]

    def not_found(req, timeout):
        raise urllib.error.HTTPError("u", 404, "none", {}, None)

    monkeypatch.setattr(openf1.urllib.request, "urlopen", not_found)
    assert openf1._fetch("http://x") == []
