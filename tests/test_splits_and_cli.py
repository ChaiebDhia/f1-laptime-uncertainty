import numpy as np

from f1lap.cli import main
from f1lap.splits import late_stint_split, random_stint_split, stint_holdout_splits


def test_late_stint_split_never_leaks(race):
    tr, te = late_stint_split(race)
    assert len(te) > 0 and not set(tr) & set(te)
    train, test = race.iloc[tr], race.iloc[te]
    last_train = train.groupby("stint_id")["lap"].max()
    for sid, g in test.groupby("stint_id"):
        assert g["lap"].min() > last_train[sid]


def test_stint_holdout_keeps_stints_whole(race):
    for tr, te in stint_holdout_splits(race):
        assert not set(race.iloc[tr]["stint_id"]) & set(race.iloc[te]["stint_id"])


def test_random_stint_split_is_reproducible_and_disjoint(race):
    a = random_stint_split(race, seed=1)
    b = random_stint_split(race, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not set(race.iloc[a[0]]["stint_id"]) & set(race.iloc[a[1]]["stint_id"])


def test_cli_synthetic_end_to_end(tmp_path):
    out = tmp_path / "res"
    rc = main(["run", "--synthetic", "--synthetic-races", "1", "--n-boot", "30", "--out", str(out)])
    assert rc == 0
    for f in [
        "REPORT.md",
        "validation_pooled.csv",
        "validation_by_race.csv",
        "physical_params.csv",
        "degradation.png",
        "late_stint_forecast.png",
        "diagnostics.png",
    ]:
        assert (out / f).exists(), f
    text = (out / "REPORT.md").read_text()
    assert "SYNTHETIC DATA" in text and "Coverage" in text


def test_cli_verify(tmp_path):
    assert main(["verify", "--n-races", "3", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "verification.png").exists()
