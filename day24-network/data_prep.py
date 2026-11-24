"""Reduce September 2026 Citi Bike trips (raw CSVs from day02-lines/data/raw) to trips per
station pair for the whole month. Writes data/od_202609_all.csv.gz and data/stations_202609.csv."""
import glob
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE.parent / "day02-lines" / "data" / "raw"
cols = ["started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name",
        "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"]
ids = {"start_station_id": str, "end_station_id": str}
df = pd.concat([pd.read_csv(f, usecols=cols, dtype=ids) for f in sorted(glob.glob(str(RAW / "202609-*.csv")))], ignore_index=True)
n0 = len(df)
for c in ("started_at", "ended_at"):
    df[c] = pd.to_datetime(df[c], format="ISO8601")
dur = (df.ended_at - df.started_at).dt.total_seconds()
df = df[(df.started_at.dt.strftime("%Y-%m") == "2026-09") & dur.between(60, 2 * 3600)
        & df.start_station_id.notna() & df.end_station_id.notna()]
print(f"{n0:,} trips -> {len(df):,} with both stations, 1 min-2 h (round trips kept)")
pts = pd.concat([
    df[["start_station_id", "start_station_name", "start_lat", "start_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1),
    df[["end_station_id", "end_station_name", "end_lat", "end_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1)])
st = pts.groupby("id").agg(name=("name", lambda s: s.mode().iat[0]), lat=("lat", "median"), lng=("lng", "median")).reset_index()
st.to_csv(HERE / "data" / "stations_202609.csv", index=False)
df["min"] = dur[df.index] / 60; df["mem"] = df.member_casual == "member"
od = (df.groupby(["start_station_id", "end_station_id"])
        .agg(trips=("min", "size"), med_min=("min", "median"),
             member=("mem", "mean"))
        .reset_index().rename(columns={"start_station_id": "a", "end_station_id": "b"}))
od["med_min"] = od.med_min.round(1); od["member"] = od.member.round(3)
od.to_csv(HERE / "data" / "od_202609_all.csv.gz", index=False)
print(len(od), "pairs", od.trips.sum(), "trips")
