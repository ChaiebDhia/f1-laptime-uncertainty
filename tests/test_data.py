import sys
import types

import numpy as np
import pandas as pd

import f1lap
from f1lap.data import clean_laps, validate_laps


def _row(**kw):
    base = dict(
        race="r",
        driver="A",
        stint=1,
        lap=5,
        compound="SOFT",
        tyre_age=3,
        lap_time=90.0,
        green=True,
        accurate=True,
        pit=False,
    )
    return {**base, **kw}


def test_clean_removes_each_kind_of_bad_lap_and_logs_it():
    rows = [
        _row(lap=5),
        _row(lap=6, pit=True),
        _row(lap=7, green=False),
        _row(lap=8, accurate=False),
        _row(lap=9, compound="WET"),
        _row(lap=1),
        _row(lap=10, lap_time=99.0),
        _row(lap=11, lap_time=np.nan),
    ]
    out, log = clean_laps(pd.DataFrame(rows))
    assert list(out["lap"]) == [5]
    assert log.iloc[0]["remaining"] == len(rows)
    assert log["removed"].sum() == len(rows) - 1
    assert "stint_id" in out.columns


def test_clean_rejects_missing_columns():
    try:
        clean_laps(pd.DataFrame({"lap": [1]}))
    except ValueError as e:
        assert "missing columns" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_validate_laps_flags_problems(race):
    assert validate_laps(race) == []
    bad = race.copy()
    bad.loc[0, "lap_time"] = -1.0
    bad = pd.concat([bad, bad.iloc[[1]]])
    issues = validate_laps(bad)
    assert any("non-positive" in i for i in issues)
    assert any("duplicate" in i for i in issues)


def test_synthetic_is_deterministic_and_clean():
    a = f1lap.synthetic_race(seed=3)
    b = f1lap.synthetic_race(seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert validate_laps(a) == []
    assert not a.equals(f1lap.synthetic_race(seed=4))


def test_load_race_transforms_fastf1_schema(monkeypatch, tmp_path):
    """Feed load_race a fake FastF1 session that uses FastF1's documented column names/dtypes."""
    laps = pd.DataFrame(
        {
            "Driver": ["VER"] * 4,
            "Stint": [1.0, 1.0, 1.0, 2.0],
            "LapNumber": [1.0, 2.0, 3.0, 4.0],
            "Compound": ["SOFT", "SOFT", "SOFT", "HARD"],
            "TyreLife": [1.0, 2.0, 3.0, 1.0],
            "LapTime": pd.to_timedelta([95.0, 90.0, 90.4, np.nan], unit="s"),
            "TrackStatus": ["1", "1", "12", "1"],
            "IsAccurate": [False, True, True, np.nan],
            "PitInTime": pd.to_timedelta([np.nan] * 4, unit="s"),
            "PitOutTime": pd.to_timedelta([np.nan] * 4, unit="s"),
        }
    )

    class FakeSession:
        def __init__(self):
            self.laps = laps

        def load(self, **kwargs):
            assert kwargs["telemetry"] is False

    fake = types.SimpleNamespace(
        Cache=types.SimpleNamespace(enable_cache=lambda p: None),
        get_session=lambda year, gp, ident: FakeSession(),
    )
    monkeypatch.setitem(sys.modules, "fastf1", fake)
    df, log = f1lap.load_race(2024, "Testland", cache_dir=str(tmp_path / "c"))
    assert list(df["lap"]) == [2]  # lap 1, yellow-flag lap 3 and NaT lap 4 removed
    assert df.iloc[0]["lap_time"] == 90.0
    assert df.iloc[0]["race"] == "2024 Testland"
    assert log["removed"].sum() == 3
