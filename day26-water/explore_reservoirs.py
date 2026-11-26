"""Day 26 · Water: New York City's water falls from the Catskills, and the city now uses a third less of it.

Map of the Catskill/Delaware reservoirs and the aqueducts that carry their water to the city,
each reservoir filled by how full it was on a chosen day (1982 to 2026), next to the system's
total storage and the city's yearly water use.

Writes:
  - out/nyc_water.png    (static, for posting)
  - out/nyc_water.html   (interactive, self-contained)

Data: NYC DEP Reservoir Operations (NYC Open Data zkky-n5j3); NYC water consumption (ia2d-e54m);
      usable capacities from DEP's reservoir levels page (Nov 2026); reservoirs and aqueducts from
      OpenStreetMap; state outlines from Census cartographic boundaries; boroughs from NYC Open Data.
Run:  ../.venv/bin/python explore_reservoirs_data.py  (once)  then  ../.venv/bin/python explore_reservoirs.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
ROOT = HERE.parent

DAY = "Day 26: Water"
CRS_M = "EPSG:32618"
SOURCE_HTML = ('Data: <a href="https://data.cityofnewyork.us/d/zkky-n5j3">NYC DEP reservoir operations</a> and '
               '<a href="https://data.cityofnewyork.us/d/ia2d-e54m">water consumption</a>, NYC Open Data; '
               'aqueducts and reservoirs © OpenStreetMap contributors')
SOURCE = "Data: NYC DEP reservoir operations and water consumption (NYC Open Data); aqueducts © OpenStreetMap"

# Usable storage, billion gallons (DEP reservoir levels page, Nov 2026)
CAPACITY = {"Ashokan": 123.8, "Schoharie": 18.2, "Rondout": 49.5, "Neversink": 35.2,
            "Pepacton": 142.8, "Cannonsville": 94.4}
TOTAL_CAP = sum(CAPACITY.values())  # 463.9
SYSTEM = {"Ashokan": "Catskill", "Schoharie": "Catskill", "Rondout": "Delaware", "Neversink": "Delaware",
          "Pepacton": "Delaware", "Cannonsville": "Delaware"}

# Aqueduct/tunnel ways to draw, by system (OSM names). Shafts, gatehouses, the Old Croton
# trail and New Jersey's Wanaque Aqueduct are left out.
CONDUITS = {
    "Delaware": ["West Delaware Tunnel", "East Delaware Aqueduct", "Neversink Tunnel",
                 "Roundout - West Branch Tunnel", "Delaware Aqueduct", "Delaware Aqueduct Bypass",
                 "Delaware Aqueduct Kensico Bypass"],
    "Catskill": ["Shandaken Tunnel", "Catskill Aqueduct", "Rondout Pressure Tunnel", "Wallkill Pressure Tunnel",
                 "Mohonk Grade Tunnel", "Catskill Aqueduct Kensico Bypass"],
    "Croton": ["New Croton Aqueduct"],
}
TITLE = "New York uses a third less water than in 1991, and its reservoirs haven’t dipped below half full since 2002"


def load_series():
    d = pd.read_csv(DATA / "reservoir_ops.csv")
    s = d[d.parameter == "Storage_derived"].pivot_table(index="date", columns="reservoir", values="value")
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D").interpolate(limit=3)[list(CAPACITY)] / 1000  # billion gallons
    pct = s / pd.Series(CAPACITY) * 100
    total = s.sum(axis=1) / TOTAL_CAP * 100
    c = pd.DataFrame(json.load(open(DATA / "consumption.json"))).astype(float).sort_values("year")
    c = c[c.year >= 1982]
    return s, pct, total, c


def findings(total, use):
    lows = total.groupby(total.index.year).agg(["min", "idxmin"])
    since = total["2003":]
    u = use.set_index("year").nyc_consumption_million_gallons_per_day
    f = {
        "low_all": (total.idxmin(), total.min()),
        "low_since_2003": (since.idxmin(), since.min()),
        "drought_2024": (total["2024"].idxmin(), total["2024"].min()),
        "use_1991": u[1991], "use_last": u.iloc[-1], "use_last_year": int(u.index[-1]),
        "last": (total.index[-1], total.iloc[-1]),
    }
    f["use_cut"] = 1 - f["use_last"] / f["use_1991"]
    print(f"lowest ever {f['low_all'][1]:.1f}% on {f['low_all'][0]:%Y-%m-%d}; "
          f"lowest since 2003 {f['low_since_2003'][1]:.1f}% on {f['low_since_2003'][0]:%Y-%m-%d}; "
          f"2024 low {f['drought_2024'][1]:.1f}% on {f['drought_2024'][0]:%Y-%m-%d}")
    print(f"use {f['use_1991']:.0f} MGD (1991) -> {f['use_last']:.0f} ({f['use_last_year']}): "
          f"{f['use_cut']:.0%} less")
    assert f["low_since_2003"][1] >= 50, "headline says never below half full since 2002"
    assert 0.28 < f["use_cut"] < 0.36, "headline says a third less"
    print("years below half full:", sorted({d.year for d in total[total < 50].index}))
    return f, lows


def load_geo():
    bounds = gpd.GeoSeries([box(-75.75, 40.45, -73.25, 42.6)], crs=4326).to_crs(CRS_M).iloc[0]
    osm = gpd.read_file(DATA / "osm_water.geojson").to_crs(CRS_M)
    res = osm[osm.kind == "reservoir"].copy()
    res["key"] = res.name.str.replace(r"^(Upper |Lower )", "", regex=True).str.replace(r" Reservoir.*$", "", regex=True)
    res = res.dissolve("key", as_index=False)
    res["system"] = res.key.map(SYSTEM).fillna("Croton")
    res.loc[res.key.isin(["Kensico", "West Branch", "Hillview"]), "system"] = "Balancing"
    res = res[res.geometry.area > 1e5]  # drop slivers (lines tagged as water)
    con = osm[osm.kind == "conduit"].copy()
    con["system"] = con.name.map({n: k for k, v in CONDUITS.items() for n in v})
    con = con.dropna(subset=["system"]).dissolve(["system", "name"], as_index=False)
    counties = gpd.read_file(f"zip://{ROOT / 'day09-urban-rural/data/counties.zip'}")
    states = counties[counties.STUSPS.isin(["NY", "NJ", "PA", "CT", "MA", "VT"])].to_crs(CRS_M)
    states = states.dissolve("STUSPS", as_index=False).clip(bounds)
    boro = gpd.read_file(ROOT / "day01-points/data/boroughs.geojson").to_crs(CRS_M)
    nyc = boro.dissolve()
    return res, con, states, nyc, bounds


def png(res, con, states, nyc, pct, total, use, f):
    S.apply()
    blues = S.make_scale(list(zip(np.linspace(0, 100, len(S.SEQUENTIAL_BLUE)), S.SEQUENTIAL_BLUE)))
    day = f["drought_2024"][0]
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.03, 0.07, 0.42, 0.80])
    states.boundary.plot(ax=ax, color=S.C["other"], linewidth=S.MAP["boundary_width"])
    nyc.plot(ax=ax, color=S.C["other"], edgecolor=S.INK, linewidth=0.3)
    for _, r in con.iterrows():
        gpd.GeoSeries([r.geometry]).plot(ax=ax, color=S.INK, linewidth=0.9 if r.system != "Croton" else 0.6,
                                         linestyle="-")
    col = res.key.map(lambda k: blues(min(pct.loc[day, k], 100)) if k in CAPACITY else S.C["closed"])
    res.plot(ax=ax, color=col, edgecolor=S.INK, linewidth=0.3)
    off = {"Rondout": (10, -2, "left", "center"), "Neversink": (-10, 0, "right", "center"),
           "Ashokan": (12, 0, "left", "center"), "Schoharie": (10, 0, "left", "center"),
           "Kensico": (10, 0, "left", "center")}
    for _, r in res[res.key.isin(list(CAPACITY) + ["Kensico"])].iterrows():
        p = r.geometry.representative_point()
        lbl = r.key + (f" {min(pct.loc[day, r.key], 100):.0f}%" if r.key in CAPACITY else "")
        dx, dy, ha, va = off.get(r.key, (0, -8, "center", "top"))
        ax.annotate(lbl, (p.x, p.y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=S.SIZE["annotation"] + 0.5, color=S.INK, fontweight="semibold")
    b = nyc.total_bounds
    ax.annotate("New York City", (b[2], (b[1] + b[3]) / 2), xytext=(8, 0), textcoords="offset points",
                fontsize=S.SIZE["annotation"] + 1, color=S.INK, fontweight="semibold", va="center", ha="left")
    xmin, ymin, xmax, ymax = res.total_bounds
    ax.set_xlim(xmin - 15e3, xmax + 15e3)
    ax.set_ylim(nyc.total_bounds[1] - 8e3, ymax + 10e3)
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.text(0.98, 0.98, f"Reservoirs on {day:%b %-d, %Y},\nthe low point of the 2024 drought.\n"
            f"Darker blue = fuller.\nGrey: no daily data (Croton system,\nKensico, West Branch).",
            transform=ax.transAxes, fontsize=S.SIZE["legend"], color=S.MUTED, va="top", ha="right")

    # storage chart
    a1 = fig.add_axes([0.52, 0.47, 0.45, 0.33])
    t = total.values
    a1.fill_between(total.index, t, 50, where=t < 50, color=S.DIVERGING[3], lw=0, interpolate=True)
    a1.plot(total.index, t, color=S.SEQUENTIAL_BLUE[5], lw=0.7)
    a1.axhline(50, color=S.INK, lw=0.6, ls=(0, (3, 3)))
    a1.text(total.index[-1], 52, "half full", fontsize=S.SIZE["annotation"], color=S.INK, ha="right", va="bottom")
    a1.set_ylim(0, 105); a1.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    for sp in ("top", "right"): a1.spines[sp].set_visible(False)
    a1.set_title("Catskill/Delaware reservoirs, % full. Amber: below half full", loc="left",
                 fontsize=S.SIZE["panel_title"], fontweight="semibold", fontfamily=S.HEAD_FONT, pad=10)
    for (d, v), lab, ty in [(f["low_all"], "Lowest", 14), (f["low_since_2003"], "Lowest since 2002", 14),
                            (f["drought_2024"], "2024 drought", 27)]:
        a1.annotate(f"{lab}: {v:.0f}%, {d:%b %Y}", (d, v), xytext=(d, ty), ha="center", va="top",
                    fontsize=S.SIZE["annotation"], color=S.INK,
                    arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.6, shrinkA=1, shrinkB=1))
    # use chart
    a2 = fig.add_axes([0.52, 0.09, 0.45, 0.27], sharex=a1)
    yrs = pd.to_datetime(use.year.astype(int).astype(str) + "-07-01")
    a2.bar(yrs, use.nyc_consumption_million_gallons_per_day, width=300, color=S.SEQUENTIAL_BLUE[4])
    a2.set_yticks([0, 500, 1000, 1500], ["0", "500", "1,000", "1,500"])
    for sp in ("top", "right"): a2.spines[sp].set_visible(False)
    a2.set_title("New York City water use, million gallons a day", loc="left",
                 fontsize=S.SIZE["panel_title"], fontweight="semibold", fontfamily=S.HEAD_FONT, pad=10)
    a2.set_ylim(0, 1600)
    fig.text(0.5, 0.915, "The six Catskill and Delaware reservoirs hold 464 billion gallons and send about 90% of "
             "the city’s water downhill through tunnels. The 2024 drought left them fuller than the droughts of "
             "1985–2002.", ha="center", fontsize=S.SIZE["subtitle"] + 1.5, color=S.MUTED, wrap=True)
    S.title(fig, TITLE)
    S.credit(fig, f"% full = DEP daily storage ÷ usable capacity  ·  {SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "nyc_water.png", dpi=S.MAP["png_dpi"])


def page(res, con, states, nyc, pct, total, use, f, lows):
    def gj(g, cols, tol):
        g = g[cols + ["geometry"]].copy()
        g["geometry"] = g.geometry.simplify(tol)
        return json.loads(g.to_json(drop_id=True))
    res = res.copy()
    res["cap"] = res.key.map(CAPACITY)
    marks = [{"date": f"{d:%Y-%m-%d}", "v": round(v, 1), "label": lab} for (d, v), lab in
             [(f["low_all"], "Lowest"), (f["low_since_2003"], "Lowest since 2002"), (f["drought_2024"], "2024 drought")]]
    data = {
        "title": TITLE, "day": DAY, "source": SOURCE_HTML,
        "start": f"{total.index[0]:%Y-%m-%d}",
        "res_keys": list(CAPACITY), "cap": CAPACITY, "total_cap": TOTAL_CAP,
        # daily % full x10, one array per reservoir plus total
        "pct": {k: [None if np.isnan(v) else int(round(v * 10)) for v in pct[k].values] for k in CAPACITY},
        "total": [None if np.isnan(v) else int(round(v * 10)) for v in total.values],
        "use": [[int(y), round(v)] for y, v in zip(use.year, use.nyc_consumption_million_gallons_per_day)],
        "marks": marks, "default": marks[-1]["date"],
        "reservoirs": gj(res, ["key", "system", "cap"], 60),
        "conduits": gj(con, ["system", "name"], 80),
        "states": gj(states, ["STUSPS"], 300),
        "nyc": gj(nyc[["geometry"]].assign(n="New York City"), ["n"], 120),
        "seq": S.SEQUENTIAL_BLUE,
    }
    size = S.build_page(HERE / "web" / "explore_reservoirs.html", data, OUT / "nyc_water.html")
    print("wrote out/nyc_water.png and out/nyc_water.html", f"({size / 1e6:.1f} MB)")


# ---- mock: reservoirs drawn 2.2x and drained toward their dams ----
GLASS_SCALE = 2.2
# Nudge (km) so the enlarged shapes don't overlap.
NUDGE = {"Cannonsville": (-7, 1), "Pepacton": (2, 3), "Neversink": (-7, -5),
         "Rondout": (6, -5), "Ashokan": (5, 1), "Schoharie": (0, 4)}
# Every reservoir fills like a gauge from the same kind of end: the west end if the lake is wider
# than tall, the south end if it's taller than wide. Math degrees, pointing from the bottom to the top.
AXIS = {"W": 0, "S": 90}


def glass_geo(res):
    from shapely import affinity
    out = []
    for _, r in res[res.key.isin(list(CAPACITY))].iterrows():
        dx, dy = NUDGE[r.key]
        g = affinity.scale(r.geometry, GLASS_SCALE, GLASS_SCALE, origin="centroid")
        g = affinity.translate(g, dx * 1e3, dy * 1e3)
        x0, y0, x1, y1 = g.bounds
        side = "W" if x1 - x0 >= y1 - y0 else "S"
        print(f"  {r.key}: fills from the {'west' if side == 'W' else 'south'}")
        a = np.radians(AXIS[side]); ux, uy = np.cos(a), np.sin(a)
        xs, ys = np.array(g.exterior.coords).T if g.geom_type == "Polygon" else \
            np.concatenate([np.array(p.exterior.coords) for p in g.geoms]).T
        t = xs * ux + ys * uy
        t0, t1 = t.min(), t.max()
        i0 = int(np.argmin(t)); origin = (xs[i0], ys[i0])
        out.append({"key": r.key, "geometry": g, "origin": [round(origin[0], 1), round(origin[1], 1)],
                    "angle": AXIS[side], "length": round(t1 - t0, 1)})
    return gpd.GeoDataFrame(out, crs=CRS_M)


def normals(x):
    """Normal range for each day of the year (1..366): 10th, 50th and 90th percentile over 1991-2020,
    smoothed with a 15-day window that wraps around New Year."""
    base = x["1991":"2020"].dropna()
    g = base.groupby(base.index.dayofyear)
    q = pd.DataFrame({p: g.quantile(p) for p in (0.1, 0.5, 0.9)}).reindex(range(1, 367)).interpolate()
    ext = pd.concat([q.iloc[-15:], q, q.iloc[:15]]).rolling(15, center=True).mean().iloc[15:-15]
    ext.index = range(1, 367)
    return ext


def low_days(total):
    """Days the six reservoirs together sat below the normal range (under the 10th percentile for the date)."""
    n = normals(total)
    low = pd.Series(total.values < n[0.1].reindex(total.index.dayofyear).values, index=total.index)
    before, after = int(low[:"2002"].sum()), int(low["2003":].sum())
    print(f"days below normal range: {before:,} in 1982-2002, {after} since "
          f"({sorted(set(low['2003':][low['2003':]].index.year))})")
    assert before > 1500 and after < 60, "headline numbers changed"
    return before, after


def mock_page(res, con, states, nyc, pct, total, use, f):
    gl = glass_geo(res)
    before, after = low_days(total)
    tn = normals(total)
    tenths = lambda s: [int(round(v * 10)) for v in s.values]
    g = gl[["key", "geometry"]].copy(); g["geometry"] = g.geometry.simplify(60)
    other = res[~res.key.isin(list(CAPACITY))][["key", "geometry"]].copy()
    other["geometry"] = other.geometry.simplify(60)
    con2 = con[["system", "name", "geometry"]].copy(); con2["geometry"] = con2.geometry.simplify(80)
    marks = [{"date": f"{d:%Y-%m-%d}", "v": round(v, 1), "label": lab} for (d, v), lab in
             [(f["low_all"], "Lowest"), (f["low_since_2003"], "Lowest since 2002"), (f["drought_2024"], "2024 drought")]]
    data = {
        "title": f"New York’s reservoirs ran unusually low for {before:,} days from 1982 to 2002, "
                 f"and for just {after} days since",
        "day": DAY, "source": SOURCE_HTML, "start": f"{total.index[0]:%Y-%m-%d}",
        "cap": CAPACITY,
        "pct": {k: [None if np.isnan(v) else int(round(v * 10)) for v in pct[k].values] for k in CAPACITY},
        "total": [None if np.isnan(v) else int(round(v * 10)) for v in total.values],
        "use": [[int(y), round(v)] for y, v in zip(use.year, use.nyc_consumption_million_gallons_per_day)],
        "marks": marks,
        # normal for each day of the year, in tenths of a percent: median per reservoir and the
        # system's 10th-90th percentile band
        "norm": {k: tenths(normals(pct[k])[0.5]) for k in CAPACITY},
        "normLow": {k: tenths(normals(pct[k])[0.1]) for k in CAPACITY},
        "tnorm": {"lo": tenths(tn[0.1]), "mid": tenths(tn[0.5]), "hi": tenths(tn[0.9])},
        "glass": json.loads(g.to_json(drop_id=True)),
        "fill": {r.key: {"origin": r.origin, "angle": r.angle, "length": r.length} for r in gl.itertuples()},
        "other": json.loads(other.to_json(drop_id=True)),
        "conduits": json.loads(con2.to_json(drop_id=True)),
        "states": json.loads(states[["STUSPS", "geometry"]].assign(geometry=states.geometry.simplify(300)).to_json(drop_id=True)),
        "nyc": json.loads(nyc[["geometry"]].assign(geometry=nyc.geometry.simplify(120)).to_json(drop_id=True)),
    }
    for name in ("mock_glasses", "mock_3d"):
        if (HERE / "web" / f"{name}.html").exists():
            size = S.build_page(HERE / "web" / f"{name}.html", data, OUT / f"{name}.html", data_src="mock.js")
            print(f"wrote out/{name}.html", f"({size / 1e6:.1f} MB)")


def main():
    s, pct, total, use = load_series()
    f, lows = findings(total, use)
    res, con, states, nyc, bounds = load_geo()
    print("reservoirs:", sorted(res.key), "\nconduits:", sorted(con.name))
    png(res, con, states, nyc, pct, total, use, f)
    page(res, con, states, nyc, pct, total, use, f, lows)
    if (HERE / "web" / "mock_glasses.html").exists():
        mock_page(res, con, states, nyc, pct, total, use, f)


if __name__ == "__main__":
    main()
