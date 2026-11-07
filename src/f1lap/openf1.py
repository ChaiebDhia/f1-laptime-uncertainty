"""OpenF1 data source: historical timing data from 2023 onward, free and without an account.

FastF1 now needs an F1TV account (since Formula 1 put its data APIs behind a login in
October 2026). OpenF1 (https://openf1.org) serves the same kind of data openly, so it is the
default source. Only the standard library is used for HTTP.

Mapping to the lap table used by the models:
  * lap time          laps.lap_duration (seconds)
  * compound, stint   stints.compound, stints.stint_number
  * tyre age          stints.tyre_age_at_start + laps run in the stint (FastF1 TyreLife style)
  * pit in/out laps   laps.is_pit_out_lap, pit.lap_number, and the last lap of a non-final stint
  * green flag        approximated from race-control messages: laps under a safety car or
                      virtual safety car (plus the following lap), and laps with a yellow,
                      double-yellow or red flag. This is stricter than FastF1's TrackStatus
                      filter for local yellows, and is documented as an approximation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from .data import clean_laps

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
USER_AGENT = "f1lap/1.0 (+https://github.com/ChaiebDhia/f1-laptime-uncertainty)"
MIN_INTERVAL_S = 0.4  # the API allows 30 requests per 10 s per IP
_last_request = [0.0]


def _fetch(url: str, retries: int = 5, timeout: float = 60.0):
    """GET a URL and parse JSON. Retries timeouts, 429 and 5xx with backoff; 404 means no rows."""
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []  # OpenF1 answers 404 when a query matches no rows
            if exc.code != 429 and exc.code < 500:
                raise
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = str(exc)
        delay = 2.0 * (attempt + 1)
        log.warning("OpenF1 request failed (%s); retrying in %.0f s", reason, delay)
        time.sleep(delay)
    raise RuntimeError(f"OpenF1 request failed after {retries} attempts: {url}")


def get_json(endpoint: str, params: dict, cache_dir: str | None = ".openf1_cache") -> list:
    """Fetch one endpoint, caching the raw JSON on disk so reruns are instant and reproducible."""
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}/{endpoint}?{query}"
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"{endpoint}_{digest}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    data = _fetch(url)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    return data


def find_session(year: int, gp: str, cache_dir: str | None = ".openf1_cache") -> dict:
    """Find the Race session for a year and a country, location or circuit name."""
    rows = get_json("sessions", {"year": year, "session_name": "Race"}, cache_dir)
    if not rows:
        raise ValueError(f"OpenF1 has no race sessions for {year} (data starts in 2023)")
    keys = ("country_name", "location", "circuit_short_name")
    want = gp.casefold().strip()
    exact = [r for r in rows if any(str(r.get(k, "")).casefold() == want for k in keys)]
    hits = exact or [r for r in rows if any(want in str(r.get(k, "")).casefold() for k in keys)]
    if not hits:
        known = sorted({f"{r.get('country_name')} ({r.get('location')})" for r in rows})
        raise ValueError(f"no {year} race matches {gp!r}. Available: {', '.join(known)}")
    if len(hits) > 1:
        names = ", ".join(sorted({str(r.get("location")) for r in hits}))
        raise ValueError(f"{gp!r} matches several {year} races ({names}); use the location name")
    return hits[0]


def non_green_laps(messages: list[dict], last_lap: int) -> set[int]:
    """Race laps that were not under green-flag conditions, from race-control messages."""
    bad: set[int] = set()
    sc_start = None
    for m in sorted(messages, key=lambda r: str(r.get("date") or "")):
        lap = m.get("lap_number")
        if lap is None:
            continue
        lap = int(lap)
        text = str(m.get("message") or "").upper()
        flag = str(m.get("flag") or "").upper()
        if "SAFETY CAR" in text:  # covers real and virtual safety car
            if "DEPLOYED" in text:
                sc_start = lap if sc_start is None else sc_start
            elif ("IN THIS LAP" in text or "ENDING" in text) and sc_start is not None:
                bad.update(range(sc_start, lap + 2))  # include the restart lap
                sc_start = None
        elif flag == "RED":
            bad.update(range(lap, lap + 3))
        elif flag in {"YELLOW", "DOUBLE YELLOW"}:
            bad.add(lap)
    if sc_start is not None:  # never ended (or ended after the chequered flag)
        bad.update(range(sc_start, last_lap + 1))
    return bad


def build_raw(
    race: str, laps: list, stints: list, pits: list, drivers: list, race_control: list
) -> pd.DataFrame:
    """Combine OpenF1 endpoints into the raw lap table expected by `clean_laps`."""
    if not laps or not stints:
        raise ValueError("OpenF1 returned no laps or no stints for this session")
    lap_df = pd.DataFrame(laps)
    for col in ("driver_number", "lap_number", "lap_duration"):
        if col not in lap_df.columns:
            raise ValueError(f"OpenF1 laps are missing '{col}' (columns: {sorted(lap_df.columns)})")
    if "is_pit_out_lap" not in lap_df.columns:
        lap_df["is_pit_out_lap"] = False
    lap_df = lap_df[["driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]].copy()
    lap_df["is_pit_out_lap"] = lap_df["is_pit_out_lap"].fillna(False).astype(bool)
    last_lap = int(lap_df["lap_number"].max())

    rows = []
    stint_df = pd.DataFrame(stints).sort_values(["driver_number", "stint_number"])
    final_stint = stint_df.groupby("driver_number")["stint_number"].transform("max")
    for (_, s), final in zip(stint_df.iterrows(), final_stint, strict=True):
        start = 1 if pd.isna(s.get("lap_start")) else int(s["lap_start"])
        if pd.isna(s.get("lap_end")):
            end = int(lap_df.loc[lap_df["driver_number"] == s["driver_number"], "lap_number"].max())
        else:
            end = int(s["lap_end"])
        age0 = 0 if pd.isna(s.get("tyre_age_at_start")) else int(s["tyre_age_at_start"])
        for lap in range(start, end + 1):
            rows.append(
                {
                    "driver_number": s["driver_number"],
                    "lap_number": lap,
                    "stint": s["stint_number"],
                    "compound": str(s.get("compound") or "").upper() or None,
                    "tyre_age": age0 + (lap - start) + 1,
                    "is_stint_last_lap": bool(lap == end and s["stint_number"] != final),
                }
            )
    cover = pd.DataFrame(rows).drop_duplicates(["driver_number", "lap_number"], keep="first")
    df = lap_df.merge(cover, on=["driver_number", "lap_number"], how="left")

    in_laps = {
        (int(p["driver_number"]), int(p["lap_number"]))
        for p in pits
        if p.get("driver_number") is not None and p.get("lap_number") is not None
    }
    in_lap = pd.Series(
        [
            (int(d), int(lp)) in in_laps
            for d, lp in zip(df["driver_number"], df["lap_number"], strict=True)
        ],
        index=df.index,
    )
    acronym = {
        int(d["driver_number"]): d.get("name_acronym") or f"#{d['driver_number']}"
        for d in drivers
        if d.get("driver_number") is not None
    }
    bad = non_green_laps(race_control, last_lap)
    return pd.DataFrame(
        {
            "race": race,
            "driver": df["driver_number"].map(lambda n: acronym.get(int(n), f"#{int(n)}")),
            "stint": df["stint"],
            "lap": df["lap_number"],
            "compound": df["compound"],
            "tyre_age": df["tyre_age"],
            "lap_time": pd.to_numeric(df["lap_duration"], errors="coerce"),
            "green": ~df["lap_number"].isin(bad),
            "accurate": df["lap_duration"].notna(),
            "pit": df["is_pit_out_lap"]
            | in_lap
            | df["is_stint_last_lap"].fillna(False).astype(bool),
        }
    )


def load_race_openf1(
    year: int, gp: str, cache_dir: str | None = ".openf1_cache"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download one race from OpenF1 and return (clean laps, drop log)."""
    session = find_session(year, gp, cache_dir)
    key = session["session_key"]
    log.info("OpenF1: %s %s (session %s)", year, session.get("location"), key)

    def fetch(endpoint: str) -> list:
        return get_json(endpoint, {"session_key": key}, cache_dir)

    raw = build_raw(
        f"{year} {gp}",
        fetch("laps"),
        fetch("stints"),
        fetch("pit"),
        fetch("drivers"),
        fetch("race_control"),
    )
    return clean_laps(raw)


def describe(year: int, gp: str, cache_dir: str | None = ".openf1_cache") -> dict:
    """Diagnostic: what OpenF1 returns for a race (row counts and field names per endpoint)."""
    session = find_session(year, gp, cache_dir)
    out = {"session": session}
    for ep in ("laps", "stints", "pit", "drivers", "race_control"):
        rows = get_json(ep, {"session_key": session["session_key"]}, cache_dir)
        out[ep] = {"rows": len(rows), "fields": sorted(rows[0]) if rows else []}
    return out


__all__ = ["describe", "find_session", "load_race_openf1", "non_green_laps", "build_raw"]
