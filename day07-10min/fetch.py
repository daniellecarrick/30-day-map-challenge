"""Download station-level OMNY and MetroCard entries for the two periods Day 7 compares.

Source: MTA Subway Hourly Ridership (data.ny.gov). Writes data/ridership_<year>.csv.
The API times out on a month grouped this finely, so this asks for a few days at a time.
Run:  ../.venv/bin/python fetch.py
"""
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
PERIODS = {  # year: (dataset, first day, last day)
    2024: ("wujg-7c2s", date(2024, 5, 28), date(2024, 6, 28)),
    2026: ("5wq4-mkjj", date(2026, 10, 1), date(2026, 10, 29)),
}
FIELDS = ["station_complex_id", "station_complex", "borough", "latitude", "longitude", "transit_mode", "payment_method"]
STEP = 3  # days per request


def get(dataset, start, stop):
    query = urllib.parse.urlencode({
        "$select": ",".join(FIELDS) + ",sum(ridership) as entries", "$group": ",".join(FIELDS), "$limit": 50000,
        "$where": f"transit_timestamp>='{start}T00:00:00' AND transit_timestamp<'{stop}T00:00:00'",
    })
    req = urllib.request.Request(f"https://data.ny.gov/resource/{dataset}.json?{query}",
                                 headers={"User-Agent": "30daymap-build"})
    for attempt in range(5):
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=150))
            print(dataset, start, len(rows), flush=True)
            return rows
        except Exception as e:
            print(dataset, start, "retrying:", e, flush=True)
            time.sleep(5)
    raise SystemExit(f"could not download {dataset} from {start}")


for year, (dataset, first, last) in PERIODS.items():
    spans, day = [], first
    while day <= last:
        spans.append((dataset, day, min(day + timedelta(STEP), last + timedelta(1))))
        day += timedelta(STEP)
    with ThreadPoolExecutor(4) as pool:
        rows = [r for chunk in pool.map(lambda s: get(*s), spans) for r in chunk]
    df = pd.DataFrame(rows)
    df["entries"] = df.entries.astype(float)
    df = df.groupby(FIELDS, as_index=False).entries.sum()
    df["entries"] = df.entries.round().astype(int)
    df.to_csv(DATA / f"ridership_{year}.csv", index=False)
    print(year, "wrote", len(df), "rows, total", df.entries.sum(), flush=True)
