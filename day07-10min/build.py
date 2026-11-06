"""Day 7 · 10 minute map: OMNY vs MetroCard at every subway station, 2024 vs 2026.

A remake of my 2024 Tableau map (data/raw/workbook.twbx), rebuilt from the MTA's full data and
extended to the latest month. As in the original, color = which method had more entries,
size = the gap in entries between them. Size uses one scale across
both periods, so dots can be compared when you switch years.

Writes out/omny_vs_metrocard.png and out/omny_vs_metrocard.html
Data: MTA Subway Hourly Ridership (data.ny.gov), downloaded by fetch.py
Run:  ../.venv/bin/python fetch.py   (once)
      ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 7: 10 minute map"
CRS_M = "EPSG:32618"
YEARS = (2024, 2026)
PERIOD = {2024: "May 28–Jun 28, 2024", 2026: "Oct 1–29, 2026"}
BORO_ORDER = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
# Either side of a 50/50 split, so the two sides of the diverging scale.
COLOR = {"OMNY": S.DIVERGING[1], "MetroCard": S.DIVERGING[3]}


def load_year(year):
    raw = pd.read_csv(DATA / f"ridership_{year}.csv")
    raw = raw[~raw.transit_mode.str.contains("tram", case=False)]  # Roosevelt Island Tramway isn't subway
    wide = raw.pivot_table(index="station_complex_id", columns="payment_method", values="entries",
                           aggfunc="sum", fill_value=0).astype(int)
    info = raw.drop_duplicates("station_complex_id").set_index("station_complex_id")[
        ["station_complex", "borough", "latitude", "longitude"]]
    df = info.join(wide).rename(columns={"metrocard": "mc", "omny": "omny"})
    df["total"] = df.mc + df.omny
    df["pct"] = df.omny / df.total * 100
    df["gap"] = (df.omny - df.mc).abs()
    df["winner"] = np.where(df.omny > df.mc, "OMNY", "MetroCard")
    return df


def tally(df):
    t = df.groupby("borough").agg(stations=("winner", "size"), omny=("winner", lambda s: (s == "OMNY").sum()))
    return [{"boro": b, "omny": int(t.omny[b]), "stations": int(t.stations[b])} for b in BORO_ORDER]


def make_png(boroughs, years, title):
    S.apply()
    gap_max = max(d.gap.max() for d in years.values())
    fig, axes = plt.subplots(1, 2, figsize=S.MAP["png_size_in"])
    # Frame the four main boroughs plus Staten Island's two stations; the rest of Staten Island is cropped.
    x0, y0, x1, y1 = boroughs[boroughs.boroname != "Staten Island"].total_bounds
    si = years[2026][years[2026].borough == "Staten Island"]
    si = gpd.GeoSeries(gpd.points_from_xy(si.longitude, si.latitude), crs=4326).to_crs(CRS_M)
    x0, y0, x1, y1 = min(x0, si.x.min()), min(y0, si.y.min()), max(x1, si.x.max()), max(y1, si.y.max())
    pad = 0.02 * (y1 - y0)
    for ax, y in zip(axes, YEARS):
        d = years[y].sort_values("gap", ascending=False)
        pts = gpd.GeoSeries(gpd.points_from_xy(d.longitude, d.latitude), crs=4326).to_crs(CRS_M)
        boroughs.boundary.plot(ax=ax, color=S.INK, linewidth=0.4, zorder=1)
        ax.scatter(pts.x, pts.y, s=np.maximum(d.gap.values / gap_max * 520, 7),
                   c=d.winner.map(COLOR), alpha=0.85, edgecolors=S.EDGE, linewidths=0.3, zorder=3)
        ax.set_axis_off()
        ax.set_aspect("equal", adjustable="datalim")  # keep the full panel, so labels have room beside the map
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        d = d.assign(x=pts.x.values, y=pts.y.values)
        share = d.omny.sum() / d.total.sum() * 100
        note = "\nMetroCard sales ended on December 31, 2025" if y == 2026 else ""
        S.panel_heading(ax, PERIOD[y], f"OMNY was the majority at {(d.winner == 'OMNY').sum()} of {len(d)} stations\n"
                                       f"and {share:.1f}% of all entries{note}")
        if y == 2024:
            om = d[d.winner == "OMNY"].iloc[0]
            mc = d[d.winner == "MetroCard"].iloc[0]
            S.callout(ax, (om.x, om.y), (0.0, 0.70), om.station_complex.split(" (")[0],
                      f"Widest OMNY margin: {om.gap:,}", halo=True)
            S.callout(ax, (mc.x, mc.y), (0.74, 0.42), mc.station_complex.split(" (")[0],
                      f"Widest MetroCard margin: {mc.gap:,}", halo=True)
        else:
            top = d.iloc[0]
            low = d.sort_values("pct").iloc[0]
            S.callout(ax, (top.x, top.y), (0.0, 0.70), top.station_complex.split(" (")[0].split("/")[0],
                      f"Widest OMNY margin: {top.gap:,}", halo=True)
            S.callout(ax, (low.x, low.y), (0.72, 0.05), low.station_complex.split(" (")[0],
                      f"Lowest OMNY share: {low.pct:.1f}%", halo=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.09, wspace=0.02)
    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=COLOR["OMNY"], mec="none", label="More OMNY entries"),
               Line2D([], [], marker="o", ls="", ms=8, mfc=COLOR["MetroCard"], mec="none", label="More MetroCard entries")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False,
               fontsize=S.SIZE["legend"], title="Bigger dot = wider gap, same scale on both maps",
               title_fontsize=S.SIZE["legend"])
    S.title(fig, title, y=0.985)
    S.credit(fig, "Data: MTA Subway Hourly Ridership  ·  Remake of my 2024 Tableau map  ·  "
                  f"#30DayMapChallenge {DAY}")
    fig.savefig(OUT / "omny_vs_metrocard.png", dpi=S.MAP["png_dpi"])


def main():
    years = {y: load_year(y) for y in YEARS}
    tallies = {y: tally(years[y]) for y in YEARS}
    summary = {y: {"share": round(float(years[y].omny.sum() / years[y].total.sum() * 100), 1),
                   "wins": int((years[y].winner == "OMNY").sum()), "n": len(years[y])} for y in YEARS}
    print(tallies, summary)

    meta = json.loads((HERE / "meta.json").read_text())
    boroughs = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)
    make_png(boroughs, years, meta["title"])

    ids = sorted(set(years[2024].index) | set(years[2026].index))
    stations = []
    for i in ids:
        latest = years[2026].loc[i] if i in years[2026].index else years[2024].loc[i]
        x, y = gpd.GeoSeries(gpd.points_from_xy([latest.longitude], [latest.latitude]), crs=4326).to_crs(CRS_M).iloc[0].coords[0]
        s = {"name": latest.station_complex, "boro": latest.borough, "x": round(x, 1), "y": round(y, 1)}
        for yr in YEARS:
            if i in years[yr].index:
                r = years[yr].loc[i]
                s[str(yr)] = {"mc": int(r.mc), "omny": int(r.omny), "winner": r.winner, "gap": int(r.gap),
                              "pct": round(float(r.pct), 1)}
        stations.append(s)
    b = boroughs[["boroname", "geometry"]].copy()
    b["geometry"] = b.geometry.simplify(40)
    data = {"boroughs": json.loads(b.to_json(drop_id=True)), "stations": stations, "colors": COLOR,
            "title": meta["title"], "period": {str(y): PERIOD[y] for y in YEARS},
            "tally": {str(y): tallies[y] for y in YEARS}, "summary": {str(y): summary[y] for y in YEARS}}
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "omny_vs_metrocard.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data) + ";")
    print("wrote out/omny_vs_metrocard.png and .html", f"({size / 1e6:.2f} MB)", len(stations), "stations")


if __name__ == "__main__":
    main()
