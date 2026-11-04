"""Day 2 · Lines — Which way does Citi Bike flow?

Every Citi Bike trip in September 2026 becomes a direction (straight line, start station ->
end station, reduced to a unit vector so short and long rides count equally). The city is
cut into ~400 m hexagons; each trip adds its direction to the hex where it starts AND the
hex where it ends, so a hex's summed vector is the direction traffic moves *through* it.
  - arrow angle  = main direction of travel
  - arrow length, width and color = size of that vector per average day: net trips in that
    direction (trips heading opposite ways cancel out)
Labor Day is excluded.

Writes:
  - out/citibike_flow.html   (unpublished alternative: the arrows hour by hour through an
                              average weekday or weekend day, with play/pause and a timeline)
  - out/citibike_flow.png    (static, for posting: three 3-hour windows as streamlines:
                              weekday 7-10am, weekday 4-7pm, weekend 1-4pm)
  - out/linkedin_citibike_flow.png  (1080x1350 feed image: morning vs evening)
  - out/citibike_windmap.html (the published page: particle wind map, 15-minute steps, canvas)
  - data/hex_hourly_202609.csv.gz (hex sums per day type and start hour; cache for the page)
  - data/hex_15min_202609.csv.gz  (same per 15-minute slot; cache for the wind map)
Earlier interactive versions (three-window toggle; rolling 3-hour animation) are kept as
self-contained snapshots in experiments/.
  - data/od_202609.csv.gz    (trips per station pair per window; cache so raw CSVs aren't needed)
  - data/stations_202609.csv

Data: Citi Bike System Data, https://citibikenyc.com/system-data (trip CSVs from
      s3.amazonaws.com/tripdata, unzipped into data/raw/). Borough Boundaries (NYC Open Data).
Run:  ../.venv/bin/python build.py
"""
import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 2: Lines"
SOURCE_URL = "https://citibikenyc.com/system-data"
CRS_M = "EPSG:32618"  # UTM 18N, metres (same as days 1 and 3)
MONTH = "2026-09"
HOLIDAYS = {"2026-09-07"}  # Labor Day: not a normal weekday
HEX_R = 230  # hex centre-to-corner, metres -> ~400 m across
MIN_TRIPS_PER_DAY = 8  # hexes with fewer trip ends per day are too noisy to draw (3-hour windows)
MIN_TRIPS_PER_HOUR = 4  # same idea for the one-hour frames on the page
REF_HOUR = 30  # net trips in an hour that draws the longest, darkest arrow on the page

WINDOWS = {  # key -> (label, weekday?, hours [start, end))
    "am": ("Weekday mornings, 7–10am", True, (7, 10)),
    "pm": ("Weekday evenings, 4–7pm", True, (16, 19)),
    "we": ("Weekend afternoons, 1–4pm", False, (13, 16)),
}


# ---------------------------------------------------------------- trips -> station pairs
def read_trips():
    """Read the raw trip CSVs and drop trips that carry no direction or look broken."""
    cols = ["started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name",
            "end_station_name", "start_lat", "start_lng", "end_lat", "end_lng"]
    ids = {"start_station_id": str, "end_station_id": str}
    files = sorted(glob.glob(str(RAW / f"{MONTH.replace('-', '')}-citibike-tripdata_*.csv")))
    if not files:
        sys.exit(f"No trip CSVs in {RAW}. Download {MONTH.replace('-', '')}-citibike-tripdata.zip "
                 "from https://s3.amazonaws.com/tripdata/ and unzip it there.")
    df = pd.concat(pd.read_csv(f, usecols=cols, dtype=ids) for f in files)
    n0 = len(df)
    for c in ("started_at", "ended_at"):
        df[c] = pd.to_datetime(df[c], format="ISO8601")
    dur = (df.ended_at - df.started_at).dt.total_seconds()
    df = df[(df.started_at.dt.strftime("%Y-%m") == MONTH)
            & dur.between(60, 2 * 3600)
            & df.start_station_id.notna() & df.end_station_id.notna()
            & (df.start_station_id != df.end_station_id)]
    print(f"{n0:,} trips -> {len(df):,} after dropping round trips, missing ends, <1 min, >2 h")
    return df


def build_od():
    """Reduce the raw trips to trips per (window, start, end) station, plus station locations."""
    df = read_trips()

    # one fixed location per station (e-bike trips carry GPS jitter around the dock)
    pts = pd.concat([
        df[["start_station_id", "start_station_name", "start_lat", "start_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1),
        df[["end_station_id", "end_station_name", "end_lat", "end_lng"]].set_axis(["id", "name", "lat", "lng"], axis=1)])
    stations = pts.groupby("id").agg(name=("name", lambda s: s.mode().iat[0]),
                                     lat=("lat", "median"), lng=("lng", "median")).reset_index()
    stations.to_csv(DATA / f"stations_{MONTH.replace('-', '')}.csv", index=False)

    date = df.started_at.dt.strftime("%Y-%m-%d")
    weekday = (df.started_at.dt.dayofweek < 5) & ~date.isin(HOLIDAYS)
    weekend = df.started_at.dt.dayofweek >= 5
    hour = df.started_at.dt.hour
    parts = []
    for key, (_, wd, (h0, h1)) in WINDOWS.items():
        m = (weekday if wd else weekend) & hour.between(h0, h1 - 1)
        g = df[m].groupby(["start_station_id", "end_station_id"]).size().rename("trips").reset_index()
        g.insert(0, "window", key)
        g["days"] = date[m].nunique()
        parts.append(g)
    od = pd.concat(parts).rename(columns={"start_station_id": "a", "end_station_id": "b"})
    od.to_csv(DATA / f"od_{MONTH.replace('-', '')}.csv.gz", index=False)
    return od, stations


def load():
    tag = MONTH.replace("-", "")
    od_f, st_f = DATA / f"od_{tag}.csv.gz", DATA / f"stations_{tag}.csv"
    if od_f.exists() and st_f.exists():
        od = pd.read_csv(od_f, dtype={"a": str, "b": str})
        stations = pd.read_csv(st_f, dtype={"id": str})
    else:
        od, stations = build_od()
    st = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.lng, stations.lat),
                          crs="EPSG:4326").to_crs(CRS_M)
    stations["x"], stations["y"] = st.geometry.x.values, st.geometry.y.values
    boros = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)[["boroname", "geometry"]]
    return od, stations.set_index("id"), boros


# ---------------------------------------------------------------- hour by hour (all-day page)
DAYTYPES = {"wd": "Weekdays", "we": "Weekends"}


def load_hourly(st):
    """Hex sums per (day type, start hour) for the whole month, cached in
    data/hex_hourly_YYYYMM.csv.gz (small, unlike station pairs x 24 hours).
    Day types: 'wd' = weekdays (Labor Day excluded), 'we' = weekends; `days` = how many."""
    f = DATA / f"hex_hourly_{MONTH.replace('-', '')}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    df = read_trips()
    date = df.started_at.dt.strftime("%Y-%m-%d")
    dow = df.started_at.dt.dayofweek
    df = df.assign(daytype=np.where(dow >= 5, "we", np.where(date.isin(HOLIDAYS), "", "wd")),
                   hour=df.started_at.dt.hour, date=date)
    df = df[df.daytype != ""]
    days = df.groupby("daytype").date.nunique()
    od = (df.groupby(["daytype", "hour", "start_station_id", "end_station_id"]).size().rename("trips")
            .reset_index().rename(columns={"start_station_id": "a", "end_station_id": "b"}))
    h = hex_sums(od, st, ["daytype", "hour"])
    h["days"] = h.daytype.map(days)
    h.round(3).to_csv(f, index=False)
    return h


SLOT_MIN = 15  # wind-map page: time step in minutes
# wind-map page frame (lon/lat): ~110th St to Prospect Heights, the Hudson to western Queens
WINDMAP_BBOX = (-74.028, 40.672, -73.855, 40.802)


def load_slots(st):
    """Hex sums per (day type, 15-minute slot of the day), cached in data/hex_15min_YYYYMM.csv.gz.
    slot = hour * 4 + minute // 15. Same day types and `days` as load_hourly."""
    f = DATA / f"hex_15min_{MONTH.replace('-', '')}.csv.gz"
    if f.exists():
        return pd.read_csv(f)
    df = read_trips()
    date = df.started_at.dt.strftime("%Y-%m-%d")
    dow = df.started_at.dt.dayofweek
    df = df.assign(daytype=np.where(dow >= 5, "we", np.where(date.isin(HOLIDAYS), "", "wd")),
                   slot=df.started_at.dt.hour * (60 // SLOT_MIN) + df.started_at.dt.minute // SLOT_MIN,
                   date=date)
    df = df[df.daytype != ""]
    days = df.groupby("daytype").date.nunique()
    od = (df.groupby(["daytype", "slot", "start_station_id", "end_station_id"]).size().rename("trips")
            .reset_index().rename(columns={"start_station_id": "a", "end_station_id": "b"}))
    h = hex_sums(od, st, ["daytype", "slot"])
    h["days"] = h.daytype.map(days)
    h.round(3).to_csv(f, index=False)
    return h


def hour_range(h0, n=1):
    """'7–8am', '11am–12pm', '12–1am' for trips starting h0:00 to (h0+n):00."""
    def part(h):
        h %= 24
        return (12 if h % 12 == 0 else h % 12), ("am" if h < 12 else "pm")
    (a, pa), (b, pb) = part(h0), part(h0 + n)
    return f"{a}–{b}{pb}" if pa == pb else f"{a}{pa}–{b}{pb}"


def hourly_frames(hh, n=1, min_trips=MIN_TRIPS_PER_HOUR):
    """One frame per start hour covering `n` hours (frame 8 with n=1 = trips starting 8–9am).
    Values per average day; quiet hexes dropped."""
    out = []
    for dt, g in hh.groupby("daytype"):
        days = g.days.iat[0]
        for i in range(24):
            hrs = [(i + k) % 24 for k in range(n)]
            f = g[g.hour.isin(hrs)].groupby(["q", "r"])[["vx", "vy", "n", "dep", "arr"]].sum() / days
            f = f[f.n >= min_trips].reset_index()
            out.append(f.assign(daytype=dt, frame=i))
    fr = pd.concat(out)
    fr["x"], fr["y"] = hex_center(fr.q, fr.r)
    fr["mag"] = np.hypot(fr.vx, fr.vy)
    return fr


# ---------------------------------------------------------------- hex binning
SQ3 = np.sqrt(3)


def hex_of(x, y, R=HEX_R):
    """Pointy-top hex (axial q, r) containing each point."""
    q = (SQ3 / 3 * x - y / 3) / R
    r = (2 / 3 * y) / R
    # cube rounding
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


def hex_center(q, r, R=HEX_R):
    return R * SQ3 * (q + r / 2), R * 1.5 * r


def hex_polygon(x, y, R=HEX_R):
    from shapely.geometry import Polygon
    ang = np.radians(30 + 60 * np.arange(6))
    return Polygon(zip(x + R * np.cos(ang), y + R * np.sin(ang)))


COMPASS = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]


def compass(vx, vy):
    """Map bearing as a word. The grid is UTM, so 'north' is true north (Manhattan's
    avenues run ~29° east of it)."""
    a = np.degrees(np.arctan2(vy, vx)) % 360
    return COMPASS[int(((a + 22.5) % 360) // 45)]


def hex_sums(od, st, keys):
    """Each trip (station pair x trips) adds its unit direction to the hex at both ends.
    Returns summed vx, vy, trip ends n, departures, arrivals per `keys` + hex (q, r)."""
    od = od.join(st[["x", "y"]], on="a").join(st[["x", "y"]], on="b", rsuffix="_b").dropna().reset_index(drop=True)
    dx, dy = od.x_b - od.x, od.y_b - od.y
    d = np.hypot(dx, dy)
    od = od[d > 0].assign(ux=dx / d, uy=dy / d)
    qa, ra = hex_of(od.x.values, od.y.values)
    qb, rb = hex_of(od.x_b.values, od.y_b.values)
    w = od.trips
    start = od[keys].assign(q=qa, r=ra, vx=od.ux * w, vy=od.uy * w, n=w, dep=w, arr=0)
    end = od[keys].assign(q=qb, r=rb, vx=od.ux * w, vy=od.uy * w, n=w, dep=0, arr=w)
    return pd.concat([start, end]).groupby(keys + ["q", "r"]).sum().reset_index()


def hex_names(od, st):
    """Name each hex after its busiest station (busy = trips in `od`)."""
    st = st.copy()
    st["q"], st["r"] = hex_of(st.x.values, st.y.values)
    busy = pd.concat([od.groupby("a").trips.sum(), od.groupby("b").trips.sum()]).groupby(level=0).sum()
    st["busy"] = busy.reindex(st.index).fillna(0)
    return st.sort_values("busy").groupby(["q", "r"]).name.last()


# ---------------------------------------------------------------- smooth field for the PNG
def field(g, xs, ys, sigma=280, reach=380):
    """Gaussian-weighted average of hex vectors on a regular grid (Nadaraya–Watson).
    Cells farther than `reach` from any drawn hex are masked so lines stop at the edge."""
    X, Y = np.meshgrid(xs, ys)
    px, py = X.ravel(), Y.ravel()
    hx, hy = g.x.values, g.y.values
    U = np.zeros(px.size); V = np.zeros(px.size)
    W = np.zeros(px.size); near = np.full(px.size, np.inf)
    # (np.errstate: Apple's Accelerate BLAS raises spurious matmul warnings on underflowed weights)
    with np.errstate(all="ignore"):
        for s in range(0, px.size, 4000):
            d2 = (px[s:s + 4000, None] - hx) ** 2 + (py[s:s + 4000, None] - hy) ** 2
            w = np.exp(-d2 / (2 * sigma ** 2))
            W[s:s + 4000] = w.sum(1)
            U[s:s + 4000] = w @ g.vx.values
            V[s:s + 4000] = w @ g.vy.values
            near[s:s + 4000] = np.sqrt(d2.min(1))
    mask = (near > reach) | (W < 1e-6)
    W[W == 0] = 1
    shp = X.shape
    U, V = (np.ma.array((arr / W).reshape(shp), mask=mask.reshape(shp)) for arr in (U, V))
    return U, V


STILL_HOURS = [8, 11, 14, 17]  # weekday hours shown in the PNG and LinkedIn still: the tide turning


def pull_text(p):
    """Panel subtitle from the pooled pull toward Midtown (share of trip ends)."""
    if p >= 0.05:
        return "Strong pull toward Midtown"
    if p >= 0.02:
        return "Still leaning toward Midtown"
    if p > -0.02:
        return "Nearly balanced: the tide turns"
    return "Flowing back out of Midtown"


def draw_flow(ax, g, frame, land, LW_REF, cmap, norm, density, lw=(0.3, 1.9), step=60):
    """Streamlines of one hour's net flow, framed like the wind map."""
    x0, y0, x1, y1 = frame
    xs, ys = np.arange(x0 - 800, x1 + 800, step), np.arange(y0 - 800, y1 + 800, step)
    land.plot(ax=ax, color=S.TOKENS["color"]["light"]["ctl"], edgecolor="none", zorder=0)
    land.boundary.plot(ax=ax, color=S.INK, linewidth=0.3, alpha=0.4, zorder=1)
    U, V = field(g, xs, ys)
    speed = np.hypot(U, V)
    ax.streamplot(xs, ys, U, V, density=density, color=speed, cmap=cmap, norm=norm,
                  linewidth=lw[0] + lw[1] * np.clip(speed / LW_REF, 0, 1), arrowsize=0.5,
                  arrowstyle="-|>", minlength=0.08, maxlength=3, broken_streamlines=True, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_axis_off()


def still_parts(fr, boros):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    frame = windmap_frame()
    x0, y0, x1, y1 = frame
    land = boros.clip((x0, y0, x1, y1))
    cmap = LinearSegmentedColormap.from_list("blue", S.SEQUENTIAL_BLUE[1:])
    return frame, land, cmap, Normalize(0, REF_HOUR)


def draw_png(fr, tide_, boros, out):
    """Site / gallery PNG: four weekday hours side by side, as streamlines."""
    import matplotlib.pyplot as plt
    frame, land, cmap, norm = still_parts(fr, boros)
    wd = fr[fr.daytype == "wd"]
    # wider than the series default (16 x 9.6): four square maps in a row would leave empty bands
    fig = plt.figure(figsize=(16, 6.6))
    S.title(fig, tide_["title"], y=0.965)
    fig.text(0.5, 0.885, "Each line follows the net flow of Citi Bike trips in one hour of an average weekday; "
             "faster, darker lines carry more trips.\nThe current pulls toward Midtown all morning, "
             f"strongest {hour_range(tide_['peak'])}, and turns around about {tide_['turn_clock']}.",
             ha="center", va="top", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED, linespacing=1.4)
    axes = {}
    for i, h in enumerate(STILL_HOURS):
        ax = fig.add_axes([0.015 + i * 0.247, 0.135, 0.23, 0.557])
        axes[h] = ax
        draw_flow(ax, wd[wd.frame == h], frame, land, REF_HOUR, cmap, norm, density=1.6)
        ax.text(0.0, 1.075, hour_range(h), transform=ax.transAxes, fontsize=S.SIZE["panel_title"],
                fontweight="semibold", fontfamily=S.HEAD_FONT, color=S.INK, va="bottom")
        ax.text(0.0, 1.03, pull_text(tide_["pull"]["wd"][h]), transform=ax.transAxes,
                fontsize=S.SIZE["subtitle"], color=S.MUTED, va="bottom")
    # callouts: the biggest morning sink and the strongest evening current
    g = wd[wd.frame == STILL_HOURS[0]]
    r = g.loc[(g.arr - g.dep).idxmax()]
    S.callout(axes[STILL_HOURS[0]], (r.x, r.y), (0.55, 0.93), r["name"],
              f"{r.arr - r.dep:.0f} more bikes arrive\nthan leave, {hour_range(STILL_HOURS[0])}",
              outline=hex_polygon(r.x, r.y), halo=True)
    g = wd[wd.frame == STILL_HOURS[-1]]
    r = g.loc[g.mag.idxmax()]
    S.callout(axes[STILL_HOURS[-1]], (r.x, r.y), (0.55, 0.93), "Strongest current",
              f"{r['name']}: {r.mag:.0f} trips\nan hour, net, heading {compass(r.vx, r.vy)}",
              outline=hex_polygon(r.x, r.y), halo=True)
    # key: width and shade = net trips that hour
    fig.text(0.02, 0.078, "Net trips that hour in that direction", fontsize=S.SIZE["legend"],
             fontweight="semibold", va="bottom")
    kax = fig.add_axes([0.02, 0.04, 0.25, 0.03])
    kax.set_xlim(0, 30); kax.set_ylim(-1, 1); kax.set_axis_off()
    for k, v in enumerate([5, 15, REF_HOUR]):
        kax.plot([k * 9, k * 9 + 3.5], [0, 0], color=cmap(norm(v)), lw=0.3 + 1.9 * min(v / REF_HOUR, 1),
                 solid_capstyle="butt")
        kax.text(k * 9 + 4.2, 0, f"{v}{'+' if v == REF_HOUR else ''}", va="center", fontsize=S.SIZE["legend"])
    S.credit(fig, "Each trip is a straight line from start to end station; directions are summed per ~400 m hexagon "
             "and smoothed. Average weekday, September 2026, Labor Day excluded.  "
             f"Data: Citi Bike System Data (citibikenyc.com/system-data)  ·  #30DayMapChallenge {DAY}")
    fig.savefig(out, dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote", out)


def social_card(fr, tide_, boros, out):
    """1080x1350 (4:5) feed image: the same four hours in a 2x2 grid, type sized for a phone."""
    import matplotlib.pyplot as plt
    frame, land, cmap, norm = still_parts(fr, boros)
    wd = fr[fr.daytype == "wd"]
    W_IN, H_IN, DPI = 5.4, 6.75, 200
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
    fig.text(0.06, 0.962, "Citi Bike's daily tide", ha="left", va="top", fontsize=19, fontweight="bold",
             color=S.INK, fontfamily=S.HEAD_FONT)
    fig.text(0.06, 0.905, f"Every weekday, bikes flow toward Midtown until about {tide_['turn_clock']},\n"
             "then turn around. Faster, darker lines carry more trips.",
             ha="left", va="top", fontsize=8.3, color=S.MUTED, linespacing=1.35)
    pos = [(0.06, 0.47), (0.52, 0.47), (0.06, 0.08), (0.52, 0.08)]
    for (px, py), h in zip(pos, STILL_HOURS):
        ax = fig.add_axes([px, py, 0.42, 0.336])
        draw_flow(ax, wd[wd.frame == h], frame, land, REF_HOUR, cmap, norm, density=1.15, lw=(0.25, 1.4))
        ax.text(0.0, 1.085, hour_range(h), transform=ax.transAxes, fontsize=10, fontweight="semibold",
                fontfamily=S.HEAD_FONT, color=S.INK, va="bottom")
        ax.text(0.0, 1.02, pull_text(tide_["pull"]["wd"][h]), transform=ax.transAxes, fontsize=7.2,
                color=S.MUTED, va="bottom")
    fig.text(0.06, 0.018, "Net flow of every Citi Bike trip, average weekday, Sept 2026. Data: Citi Bike System Data\n"
             "#30DayMapChallenge Day 2: Lines  ·  daniellecarrick.github.io/30-day-map-challenge",
             ha="left", va="bottom", fontsize=6.3, color=S.MUTED, linespacing=1.4)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


def tide(st):
    """Hourly frames plus the headline numbers: pooled net flow toward Grand Central per hour
    (share of trip ends), its peak, and the hour it turns."""
    fr = hourly_frames(load_hourly(st))
    gc = st.loc[st.name == "Park Ave & E 42 St"].iloc[0]
    bx, by = gc.x - fr.x, gc.y - fr.y
    fr["toward"] = (fr.vx * bx + fr.vy * by) / np.hypot(bx, by)
    pull = {dt: [float(g[g.frame == i].toward.sum() / g[g.frame == i].n.sum()) for i in range(24)]
            for dt, g in fr.groupby("daytype")}
    wd = pull["wd"]
    peak = int(np.argmax(wd))
    turn = next(i for i in range(peak, peak + 24) if wd[i % 24] < 0) % 24  # first hour pointing away
    in_frames = [i for i in range(24) if wd[i] > 0]
    assert in_frames == list(range(in_frames[0], in_frames[-1] + 1)) and in_frames[0] <= 5, in_frames  # 'early morning' holds
    ratio = max(pull["we"]) / max(wd)
    print("pull toward Midtown, weekdays:", [round(100 * v, 1) for v in wd])
    print(f"peak {hour_range(peak)}, turns {hour_range(turn)}, weekend/weekday peak {ratio:.2f}")
    turn_clock = f"{turn % 12 or 12}{'am' if turn < 12 else 'pm'}"
    title = f"Every weekday, Citi Bike flows toward Midtown until about {turn_clock}, then turns around"
    return fr, {"title": title, "turn_clock": turn_clock, "peak": peak, "turn": turn, "ratio": ratio, "pull": pull}


def main():
    S.apply()
    od, st, boros = load()
    fr, tide_ = tide(st)
    fr = fr.join(hex_names(od, st), on=["q", "r"])
    draw_png(fr, tide_, boros, OUT / "citibike_flow.png")
    social_card(fr, tide_, boros, OUT / "linkedin_citibike_flow.png")
    web_page(od, st, boros, fr, tide_)
    windmap_page(od, st, boros, tide_)


def web_page(od, st, boros, fr, tide_):
    """Unpublished alternative: arrows hour by hour through an average weekday / weekend day."""
    title, turn_clock, peak, ratio = tide_["title"], tide_["turn_clock"], tide_["peak"], tide_["ratio"]
    print("net trips/hour quantiles:", fr.mag.quantile([.5, .9, .98, 1]).round(1).tolist())
    lede = (f"Each frame is one hour of trips on an average day. On weekdays the net flow points toward Midtown from early "
            f"morning until about {turn_clock}, strongest {hour_range(peak)}, then away from it through the evening "
            f"and night. Weekends follow the same shape, "
            f"{'about half' if 0.4 <= ratio <= 0.6 else f'{100 * ratio:.0f}%'} as strong.")

    names = hex_names(od, st)
    # compact: hex grid position (q, r) instead of coordinates; whole trips per day per frame
    hexes = []
    for (q, r), g in fr.groupby(["q", "r"]):
        rec = {"q": int(q), "r": int(r), "name": names.get((q, r), "")}
        for dt in DAYTYPES:
            arr = [0] * 24  # 0 = fewer than MIN_TRIPS_PER_HOUR trips that hour
            for _, w in g[g.daytype == dt].iterrows():
                arr[int(w.frame)] = [int(round(v)) for v in (w.vx, w.vy, w.n, w.arr - w.dep)]
            rec[dt] = arr
        hexes.append(rec)
    hh = load_hourly(st)
    trips_by_hour = {dt: [round(float(g[g.hour == i].dep.sum() / g.days.iat[0])) for i in range(24)]
                     for dt, g in hh.groupby("daytype")}
    pad = 800
    x0, y0, x1, y1 = fr.x.min() - pad, fr.y.min() - pad, fr.x.max() + pad, fr.y.max() + pad
    b = boros.clip((x0 - 5000, y0 - 5000, x1 + 5000, y1 + 5000))
    b["geometry"] = b.geometry.simplify(35)
    web = {
        "title": title, "lede": lede, "day": DAY, "source_url": SOURCE_URL,
        "daytypes": [{"key": k, "label": v} for k, v in DAYTYPES.items()],
        "labels": [hour_range(i) for i in range(24)], "trips_by_hour": trips_by_hour,
        "hex_r": HEX_R, "ref": REF_HOUR, "keys": [5, 15, REF_HOUR], "min_trips": MIN_TRIPS_PER_HOUR,
        "blues": S.SEQUENTIAL_BLUE[1:],
        "hexes": hexes, "boroughs": json.loads(b.to_json(drop_id=True)), "extent": [x0, y0, x1, y1],
    }
    # data.js lets web/index.html open straight from disk while editing (gitignored)
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "index.html", web, OUT / "citibike_flow.html")
    print("wrote", OUT / "citibike_flow.html", f"({size / 1e6:.2f} MB)")


def windmap_frame():
    """The wind map's frame (WINDMAP_BBOX) in projected metres: x0, y0, x1, y1."""
    lo0, la0, lo1, la1 = WINDMAP_BBOX
    c = gpd.GeoSeries(gpd.points_from_xy([lo0, lo0, lo1, lo1], [la0, la1, la0, la1]), crs="EPSG:4326").to_crs(CRS_M)
    return c.x.min(), c.y.min(), c.x.max(), c.y.max()


def slot_flow(st):
    """Net flow per 15-minute slot and hex inside the wind-map frame, as trips per hour on an
    average day. Returns (hex keys [(q, r)], {daytype: float array [slot, hex, (vx, vy)]},
    {daytype: trips started per hour, by slot})."""
    raw = load_slots(st)
    per_hour = 60 // SLOT_MIN
    nslot = 24 * per_hour
    sl = raw.assign(vx=raw.vx / raw.days * per_hour, vy=raw.vy / raw.days * per_hour, n=raw.n / raw.days * per_hour)
    sl = sl[sl.n >= MIN_TRIPS_PER_HOUR]  # same threshold as the hourly page
    x0, y0, x1, y1 = windmap_frame()
    cx, cy = hex_center(sl.q, sl.r)
    sl = sl[(cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)]
    keep = list(sl.groupby(["q", "r"]).size().index)
    idx = {k: i for i, k in enumerate(keep)}
    flow = {}
    for dt in DAYTYPES:
        a = np.zeros((nslot, len(keep), 2), dtype=np.float32)
        g = sl[sl.daytype == dt]
        hi = np.array([idx[k] for k in zip(g.q, g.r)])
        a[g.slot.values, hi, 0] = g.vx.values
        a[g.slot.values, hi, 1] = g.vy.values
        flow[dt] = a
    trips = {dt: [round(float(g[g.slot == i].dep.sum() / g.days.iat[0] * per_hour)) for i in range(nslot)]
             for dt, g in raw.groupby("daytype")}
    print("wind map: hexes", len(keep), "| |v| quantiles", np.hypot(sl.vx, sl.vy).quantile([.5, .9, .99]).round(1).tolist())
    return keep, flow, trips


def windmap_page(od, st, boros, summary):
    """The published page: particle wind map. 15-minute slots of net flow per hex, as a rate per hour on
    an average day; the page smooths them into a field and lets particles drift along it."""
    import base64
    keep, flowf, trips = slot_flow(st)
    x0, y0, x1, y1 = windmap_frame()
    names = hex_names(od, st)
    hexes = [{"q": int(q), "r": int(r), "name": names.get((q, r), "")} for q, r in keep]
    flow = {dt: base64.b64encode(np.clip(np.round(a), -127, 127).astype(np.int8).tobytes()).decode()
            for dt, a in flowf.items()}  # net trips/hour, clamped to int8
    b = boros.clip((x0, y0, x1, y1))  # land stops at the frame
    b["geometry"] = b.geometry.simplify(25)
    print(f"wind map frame: {(x1 - x0) / 1000:.1f} x {(y1 - y0) / 1000:.1f} km")
    t = summary["turn_clock"]
    web = {
        "title": summary["title"],
        "lede": (f"Each streak drifts with the net flow of bike trips. Faster and darker streaks represent more trips. "
                 f"On weekdays the current runs toward Midtown from early morning until about {t}, "
                 f"strongest {hour_range(summary['peak'])}, then back out through the evening and night."),
        "day": DAY, "source_url": SOURCE_URL,
        "daytypes": [{"key": k, "label": v} for k, v in DAYTYPES.items()],
        "slot_min": SLOT_MIN, "trips": trips, "hex_r": HEX_R, "ref": REF_HOUR, "min_trips": MIN_TRIPS_PER_HOUR,
        "blues": S.SEQUENTIAL_BLUE[1:], "hexes": hexes, "flow": flow,
        "boroughs": json.loads(b.to_json(drop_id=True)), "extent": [x0, y0, x1, y1],
    }
    (HERE / "web" / "windmap_data.js").write_text("window.DATA = " + json.dumps(web, separators=(",", ":")) + ";")
    size = S.build_page(HERE / "web" / "windmap.html", web, OUT / "citibike_windmap.html", data_src="windmap_data.js")
    print("wrote", OUT / "citibike_windmap.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
