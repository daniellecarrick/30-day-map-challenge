"""Day 9 · Urban-rural: tidy the county inputs into data/counties.csv.
(Also reads data/usrc2020_ny_county.csv, the NY rows of the 2020 US Religion Census group detail.)

Reads data/acs/*.dat (ACS 2019-2023 5-year summary-file rows for NY counties,
streamed by fetch.sh) and data/votes2024_nysboe.csv (certified county results, copied from
the NYS Board of Elections results database; it sits behind a bot check, so fetch.sh can't download it).
Run: ../.venv/bin/python prep.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA = Path(__file__).parent / "data"
NYC = {"36005", "36047", "36061", "36081", "36085"}   # Bronx, Kings, New York, Queens, Richmond
# Religion Census groups counted as non-Christian (Messianic Jewish and Spiritualist churches excluded)
NON_CHRISTIAN = {
    "Orthodox Judaism", "Reform Judaism", "Conservative Judaism", "Reconstructionist Judaism",
    "Independent Judaism", "Chabad Judaism", "Muslim Estimate", "Hindu Temples", "Hindu Yoga and Meditation",
    "Mahayana Buddhist", "Theravada Buddhist", "Vajarayana Buddhist", "Baha'i Faith USA",
    "American Sikh Council", "Jain", "Tao", "Zoroastrian", "Vedanta Society",
}


def table(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / "acs" / f"{name}.dat", sep="|", dtype=str)
    df["fips"] = df.GEO_ID.str[-5:]
    return df.set_index("fips")


def num(df, col):
    return pd.to_numeric(df[col])


def main():
    b01, b05, b08, b0830, b03, b24, c24, b21, b0833, b19 = (table(t) for t in
        ["b01003", "b05002", "b08201", "b08301", "b25003", "b25024", "c24030", "b21001", "b08303", "b19001"])
    d = pd.DataFrame(index=b01.index)
    d["pop"] = num(b01, "B01003_E001")
    # counts (numerator, denominator) so NYC and the rest of the state can be summed honestly
    d["nocar_n"], d["nocar_d"] = num(b08, "B08201_E002"), num(b08, "B08201_E001")
    d["transit_n"], d["transit_d"] = num(b0830, "B08301_E010"), num(b0830, "B08301_E001")
    d["foreign_n"], d["foreign_d"] = num(b05, "B05002_E013"), num(b05, "B05002_E001")
    d["rich_n"], d["rich_d"] = num(b19, "B19001_E017"), num(b19, "B19001_E001")       # household income $200k+
    d["own_n"], d["own_d"] = num(b03, "B25003_E002"), num(b03, "B25003_E001")
    d["mobile_n"], d["mobile_d"] = num(b24, "B25024_E010"), num(b24, "B25024_E001")
    d["mfg_n"] = num(c24, "C24030_E007") + num(c24, "C24030_E034")     # manufacturing, male + female
    d["mfg_d"] = num(c24, "C24030_E001")
    d["veteran_n"], d["veteran_d"] = num(b21, "B21001_E002"), num(b21, "B21001_E001")   # civilians 18+
    d["commute_n"] = num(b0833, "B08303_E012") + num(b0833, "B08303_E013")             # 60-89 and 90+ min
    d["commute_d"] = num(b0833, "B08303_E001")                                          # workers not at home

    # 2020 US Religion Census: adherents of non-Christian congregations / county population
    ny_csv = DATA / "usrc2020_ny_county.csv"
    if not ny_csv.exists():   # first run: cut the NY rows out of the 8 MB workbook (needs openpyxl)
        x = pd.read_excel(DATA / "usrc2020_group_detail.xlsx", "2020 Group by County")
        x[x["State Name"] == "New York"].to_csv(ny_csv, index=False)
    r = pd.read_csv(ny_csv)
    r["fips"] = r.FIPS.astype(str).str.zfill(5)
    pct = r["Adherents as % of Total Population"]
    pop2020 = (r.Adherents / pct).where(pct > 0).groupby(r.fips).median().round()
    nonchr = r[r["Group Name"].isin(NON_CHRISTIAN)].groupby("fips").Adherents.sum()
    d["faith_n"] = nonchr.reindex(d.index).fillna(0)
    d["faith_d"] = pop2020

    # Certified 2024 results, NYS Board of Elections (results.elections.ny.gov/contest/5591).
    # Candidate totals include every party line (Harris = Dem + WFP, Trump = Rep + Con);
    # denominator = votes for any candidate, i.e. total minus blank and void ballots.
    v = pd.read_csv(DATA / "votes2024_nysboe.csv")
    shp = gpd.read_file(f"zip://{DATA / 'counties.zip'}")
    fips = shp[shp.STATEFP == "36"].set_index("NAME").GEOID
    v.index = v.county.map(fips)
    assert v.index.notna().all(), v.county[v.index.isna()]
    d["gop_n"] = v.trump_rep + v.trump_con
    d["dem_n"] = v.harris_dem + v.harris_wfp
    d["vote_d"] = v.total - v.blank - v.void
    d["nyc"] = d.index.isin(NYC)
    d.index.name = "fips"
    assert d.notna().all().all(), d[d.isna().any(axis=1)]
    d.to_csv(DATA / "counties.csv")
    print(f"wrote data/counties.csv ({len(d)} counties)")


if __name__ == "__main__":
    main()
