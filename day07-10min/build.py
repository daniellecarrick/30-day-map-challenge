"""Day 7 · 10 minute map: OMNY vs MetroCard at every subway station, 2024 vs 2026.

A remake of my 2024 Tableau map (data/raw/workbook.twbx), rebuilt from the MTA's full data and
extended to the latest month. As in the original, color = which method had more entries,
size = the gap in entries between them. Size uses one scale across
both periods, so dots can be compared when you switch years.

Writes out/omny_vs_metrocard.png, out/omny_vs_metrocard.html and
out/omny_vs_metrocard_linkedin.png (1080x1350 portrait for the LinkedIn feed; bigger type)
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


def frame(boroughs, years):
    """The four main boroughs plus Staten Island's two fare-collecting stations; the rest of Staten Island is cropped."""
    x0, y0, x1, y1 = boroughs[boroughs.boroname != "Staten Island"].total_bounds
    si = years[2026][years[2026].borough == "Staten Island"]
    si = gpd.GeoSeries(gpd.points_from_xy(si.longitude, si.latitude), crs=4326).to_crs(CRS_M)
    return min(x0, si.x.min()), min(y0, si.y.min()), max(x1, si.x.max()), max(y1, si.y.max())


def projected(d):
    pts = gpd.GeoSeries(gpd.points_from_xy(d.longitude, d.latitude), crs=4326).to_crs(CRS_M)
    return d.assign(x=pts.x.values, y=pts.y.values)


def make_png(boroughs, years, title):
    S.apply()
    gap_max = max(d.gap.max() for d in years.values())
    fig, axes = plt.subplots(1, 2, figsize=S.MAP["png_size_in"])
    x0, y0, x1, y1 = frame(boroughs, years)
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
    S.credit(fig, "Data: MTA Subway Hourly Ridership  ·  Staten Island Railway collects fares at only 2 of its 21 stations  ·  Remake of my 2024 Tableau map  ·  "
                  f"#30DayMapChallenge {DAY}")
    fig.savefig(OUT / "omny_vs_metrocard.png", dpi=S.MAP["png_dpi"])


def draw_linkedin(boroughs, years, title):
    """4:5 portrait, the largest shape the LinkedIn feed shows, with type sized to read on a phone.
    The two periods sit side by side, as on the page, on one dot scale."""
    S.apply()
    fig = plt.figure(figsize=(5.4, 6.75))
    gap_max = max(d.gap.max() for d in years.values())
    x0, y0, x1, y1 = frame(boroughs, years)
    pad = 0.015 * (y1 - y0)
    for k, y in enumerate(YEARS):
        ax = fig.add_axes([0.01 + k * 0.495, 0.24, 0.485, 0.525])
        d = projected(years[y]).sort_values("gap", ascending=False)
        boroughs.boundary.plot(ax=ax, color=S.INK, linewidth=0.3, zorder=1)
        ax.scatter(d.x, d.y, s=np.maximum(d.gap.values / gap_max * 120, 2.2), c=d.winner.map(COLOR),
                   alpha=0.85, edgecolors=S.EDGE, linewidths=0.2, zorder=3)
        ax.set_axis_off()
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal", adjustable="box", anchor="N")  # map hangs just below its stats
        share = d.omny.sum() / d.total.sum() * 100
        cx = 0.01 + k * 0.495 + 0.2425
        fig.text(cx, 0.862, PERIOD[y], ha="center", fontsize=11, fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK)
        fig.text(cx, 0.822, f"{share:.1f}%", ha="center", fontsize=20, fontweight="bold", fontfamily=S.HEAD_FONT, color=S.INK)
        fig.text(cx, 0.815, f"of entries were OMNY\nmajority at {(d.winner == 'OMNY').sum()} of {len(d)} stations",
                 ha="center", va="top", fontsize=8.5, color=S.MUTED, linespacing=1.3)
        if y == 2024:
            mc = d[d.winner == "MetroCard"].iloc[0]
            S.callout(ax, (mc.x, mc.y), (0.80, 0.70), mc.station_complex.split("-")[0],
                      f"Widest MetroCard\nmargin: {mc.gap:,}", halo=True)
        else:
            low = d.sort_values("pct").iloc[0]
            S.callout(ax, (low.x, low.y), (0.06, -0.045), low.station_complex.split(" (")[0].replace("-", "–"),
                      f"Lowest OMNY share: {low.pct:.1f}%", halo=True)

    fig.text(0.5, 0.975, title.replace(". ", ".\n", 1), ha="center", va="top", fontsize=15, linespacing=1.25,
             fontweight="bold", color=S.INK, fontfamily=S.HEAD_FONT)
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=COLOR["OMNY"], mec="none", label="More OMNY entries"),
               Line2D([], [], marker="o", ls="", ms=7, mfc=COLOR["MetroCard"], mec="none", label="More MetroCard entries")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=2, frameon=False, fontsize=9,
               title="Each dot is a station. Bigger dot = wider gap.", title_fontsize=9, handletextpad=0.3, columnspacing=1.2)
    S.credit(fig, "MetroCard sales ended Dec 31, 2025. Staten Island Railway collects fares at only 2 of its 21 stations.\n"
                  f"Data: MTA Subway Hourly Ridership  ·  #30DayMapChallenge {DAY}\n"
                  "daniellecarrick.github.io/30-day-map-challenge/day07", y=0.014)
    fig.savefig(OUT / "omny_vs_metrocard_linkedin.png", dpi=200)
    plt.close(fig)


def main():
    years = {y: load_year(y) for y in YEARS}
    tallies = {y: tally(years[y]) for y in YEARS}
    summary = {y: {"share": round(float(years[y].omny.sum() / years[y].total.sum() * 100), 1),
                   "wins": int((years[y].winner == "OMNY").sum()), "n": len(years[y])} for y in YEARS}
    print(tallies, summary)

    meta = json.loads((HERE / "meta.json").read_text())
    boroughs = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)
    make_png(boroughs, years, meta["title"])
    draw_linkedin(boroughs, years, meta["title"])
    if "--linkedin-only" in sys.argv:
        return

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
