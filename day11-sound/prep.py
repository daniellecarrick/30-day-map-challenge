"""Day 11 prep: 2025 noise complaints -> one row per address, day and sound ("complaint-days").

One address (655 E 230 St, Bronx) filed 115k complaints in 2025, 14% of the city. Counting each
address at most once per day per sound keeps a few repeat callers from defining a neighborhood.
Writes data/complaint_days.csv.gz.
"""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
BORO = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5}

# 8 sounds, in the fixed categorical order (most common first)
SOUNDS = ["Music & parties", "Banging & pounding", "Talking & TV", "Cars & horns",
          "Construction & machines", "Helicopters", "Dogs & animals", "Ice cream trucks"]


def sound(t: str, d: str) -> str:
    d = (d or "").lower()
    if "Helicopter" in t: return "Helicopters"
    if "ice cream" in d: return "Ice cream trucks"
    if "dog" in d or "animal" in d: return "Dogs & animals"
    if t == "Noise - Vehicle" or "car/truck" in d or "boat" in d: return "Cars & horns"
    if "banging" in d: return "Banging & pounding"
    if "talking" in d or "television" in d: return "Talking & TV"
    if "music" in d: return "Music & parties"
    return "Construction & machines"  # construction, jackhammers, alarms, AC, carting, lawn care, other


def main():
    df = pd.read_csv(DATA / "raw" / "noise_2025.csv.gz", dtype=str)
    n0 = len(df)
    t = pd.to_datetime(df.created_date)
    df["day"], df["hour"] = t.dt.date, t.dt.hour
    df["sound"] = [sound(a, b) for a, b in zip(df.complaint_type, df.descriptor.fillna(""))]
    cb = df.community_board.str.extract(r"^(\d\d) (.+)$")
    df["boro_cd"] = cb[1].map(BORO) * 100 + pd.to_numeric(cb[0], errors="coerce")
    df = df.dropna(subset=["boro_cd"])
    df = df[df.boro_cd % 100 <= 18]  # the 59 residential districts; drops parks, airports, "Unspecified"
    df["boro_cd"] = df.boro_cd.astype(int)
    df["where"] = (df.incident_address.fillna(df.latitude.str[:7] + "," + df.longitude.str[:8])
                   .fillna(pd.Series(df.index.astype(str), index=df.index)))  # no location: keep as its own row
    df = df.sort_values("created_date").drop_duplicates(["where", "day", "sound"])
    print(f"{n0:,} complaints -> {len(df):,} complaint-days")
    df[["boro_cd", "hour", "sound", "where"]].to_csv(DATA / "complaint_days.csv.gz", index=False)


if __name__ == "__main__":
    main()
