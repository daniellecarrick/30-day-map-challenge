"""Day 17 · Light & dark — every lighthouse blinks its own signal.

Every light on NOAA's general-scale nautical charts has an official rhythm (S-57 SIGSEQ),
e.g. "00.4+(03.6)" = 0.4 s of light, then 3.6 s of dark. Lighthouses near each other are
given different rhythms so a sailor can tell which one they are looking at. From that: a
30-second ring of each lighthouse's signal for the PNG, ordered along the coast, and a page
where every lighthouse blinks its real rhythm.

Kept: fixed lights in US waters (nearest country is the US); airport beacons, air-obstruction,
flood and emergency lights dropped. Lights at the same spot with the same rhythm (colour
sectors of one light) are merged. "Lighthouse" = a light within 300 m of a charted
landmark whose name contains "Light", minus numbered pier/channel lights and instrument towers.
Same signal = same character, colour, period and on/off sequence.

Writes out/map.png and out/map.html
Data: NOAA ENC Direct (General charts: Light_point, Landmark_point); Natural Earth
Run:  ../.venv/bin/python build.py
"""
import json
import math
import re
import textwrap
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "noaa"
RAW.mkdir(parents=True, exist_ok=True)
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 17: Light & dark"
SOURCE = "NOAA Electronic Navigational Charts (ENC Direct)"
SOURCE_URL = "https://encdirect.noaa.gov/"
DROP_CAT = {"aero light", "air obstruction light", "flood light", "emergency"}
US = {"United States of America"}
WINDOW = 30  # seconds of rhythm drawn on each ring

# S-57 codes -> chart abbreviations
COLOUR = {"1": "W", "3": "R", "4": "G", "6": "Y", "11": "Y", "9": "Y"}
LITCHR = {1: "F", 2: "Fl", 3: "LFl", 4: "Q", 5: "VQ", 6: "UQ", 7: "Iso", 8: "Oc", 9: "IQ", 12: "Mo",
          13: "FFl", 19: "Al", 25: "Al.Oc", 26: "Al.LFl", 28: "Al.Fl", 29: "Al.Fl"}
# Light colours. Real signal colours, so they are drawn as themselves, taken from the series scales.
DARK = S.TOKENS["color"]["dark"]
CAT_D = S.TOKENS["scales"]["categorical_dark"]
LIGHT = {"W": S.SEQUENTIAL_AMBER[0], "Y": S.SEQUENTIAL_AMBER[1], "R": CAT_D[7], "G": CAT_D[2]}
LIGHT_NAME = {"W": "White", "Y": "Yellow", "R": "Red", "G": "Green"}


# ---------------- data ----------------
SCALES = ["general", "coastal", "approach", "harbour"]   # NOAA chart scales, least to most detailed (fetch.py)
MATCH_M = 300                                             # a lighthouse's name and its light: this close


def read(name):
    f = RAW / name
    return json.loads(f.read_text()) if f.exists() else []


def parse_seq(s):
    """'00.3+(00.7)' -> [(True, 0.3), (False, 0.7)]"""
    out = []
    for part in (s or "").replace(" ", "").split("+"):
        if not part:
            continue
        try:
            out.append((not part.startswith("("), float(part.strip("()"))))
        except ValueError:
            return None
    return out if out and any(on for on, _ in out) and sum(t for _, t in out) > 0 else None


def label(a, colours):
    """Chart-style description, e.g. 'Fl(2) W 15s'."""
    c = LITCHR.get(a.get("LITCHR"), "")
    g = (a.get("SIGGRP") or "").strip()
    if g not in ("", "()", "(1)"):
        c += g
    per = a.get("SIGPER")
    per = f" {per:g}s" if per else ""
    return f"{c} {''.join(colours)}{per}".strip()


# names that aren't lighthouses: numbered pier/channel lights, range (front/rear) pairs, towers, buoys
NOT_LIGHTHOUSE = re.compile(r"Ruins|Former|Historic|Instrumentation|Buoy|Range|Front|Rear|Daybeacon|Light \d|"
                            r"Lights? [A-Z]?\d+[A-Z]?$|"
                            r"\b\d+[A-Z]?$", re.I)


def clean_name(n):
    """'Grays Reef Light; MMSI: 993682389' -> 'Grays Reef Light'; 'Chesapeake Lights (2)' -> 'Chesapeake Light'"""
    n = re.split(r"[,;]", n)[0].replace(" (2)", "").replace("Lights", "Light").strip()
    return re.sub(r"\s+", " ", n.replace(" and Directional Light", ""))


def lights_all():
    """Every charted light with a rhythm, all chart scales, in US waters. The same light drawn on
    several charts (or as several colour sectors) is merged."""
    merged = {}
    for sc in SCALES:
        for f in read(f"lights_{sc}.json") + read(f"lights_{sc}_fixed.json"):
            a, g = f["attributes"], f["geometry"]
            if (a.get("CATLIT") or "").strip() in DROP_CAT:
                continue
            # a fixed light (LITCHR 1) shines steadily: one long "on" with no dark
            seq = [(True, 1.0)] if a.get("LITCHR") == 1 and not a.get("SIGSEQ") else parse_seq(a.get("SIGSEQ"))
            if not seq:
                continue
            key = (round(g["x"], 3), round(g["y"], 3), a.get("SIGSEQ") or "F")      # ~100 m
            cols = [COLOUR[c] for c in str(a.get("COLOUR") or "").split(",") if c in COLOUR]
            m = merged.setdefault(key, dict(lon=g["x"], lat=g["y"], seq=seq, a=a, colours=[], range=0))
            m["colours"] += [c for c in cols if c not in m["colours"]]
            m["range"] = max(m["range"], a.get("VALNMR") or 0)
    rows = list(merged.values())
    ne = gpd.read_file(DATA / "ne" / "ne_10m_admin_0_countries.zip")[["ADMIN", "geometry"]].to_crs(5070)
    pts = gpd.GeoDataFrame(geometry=[Point(r["lon"], r["lat"]) for r in rows], crs=4326).to_crs(5070)
    near = gpd.sjoin_nearest(pts, ne, how="left", max_distance=200_000)
    near = near[~near.index.duplicated()]
    rows = [r for r, adm in zip(rows, near["ADMIN"]) if adm in US]
    for r in rows:
        r["period"] = sum(t for _, t in r["seq"])
        r["colours"] = r["colours"] or ["W"]
        r["label"] = label(r["a"], r["colours"])
    return rows


def names_all():
    """Named lighthouse points, best source first: OpenStreetMap lighthouses, then NOAA's charted
    lighthouse structures (landmarks), then names written on the lights themselves."""
    out = []
    for e in read("osm_lighthouses.json"):
        t = e.get("tags", {})
        nm = t.get("name") or t.get("seamark:name")
        c = e.get("center") or e
        if nm and "lat" in c:
            out.append((c["lon"], c["lat"], clean_name(nm), 0))
    for sc in SCALES:
        for m in read(f"landmarks_{sc}.json"):
            nm = (m["attributes"].get("OBJNAM") or "").strip()
            if "light" in nm.lower() and "x" in m.get("geometry", {}):
                out.append((m["geometry"]["x"], m["geometry"]["y"], clean_name(nm), 1))
        for f in read(f"lights_{sc}.json"):
            nm = (f["attributes"].get("OBJNAM") or "").strip()
            if "light" in nm.lower():
                out.append((f["geometry"]["x"], f["geometry"]["y"], clean_name(nm), 2))
    return [o for o in out if not NOT_LIGHTHOUSE.search(o[2])]


def region(r):
    lat, lon = r["lat"], r["lon"]
    if lon < -150 and lat < 30:
        return "Hawaii"
    if lat > 51 and lon < -129:
        return "Alaska"
    if lon < -115:
        return "Pacific"
    if lat > 41.2 and -93 < lon < -76:
        return "Great Lakes"
    if lat < 31 and lon < -81.2:
        return "Gulf of Mexico"
    if lat < 19 and lon > -68:
        return "Puerto Rico"
    return "Atlantic"


# regions in reading order, and how to walk along each coast
# Atlantic: New England east -> west (Maine, Cape Cod, Long Island Sound, the Hudson), then New York south
REGIONS = [("Atlantic", lambda r: (0, -r["lon"]) if r["lat"] > 40.55 else (1, -r["lat"])), ("Gulf of Mexico", lambda r: -r["lon"]),
           ("Puerto Rico", lambda r: r["lon"]), ("Pacific", lambda r: -r["lat"]), ("Alaska", lambda r: r["lon"]),
           ("Hawaii", lambda r: r["lon"]), ("Great Lakes", lambda r: r["lon"])]


def lighthouses():
    """Named lighthouses, each with the strongest light within MATCH_M of its name point,
    ordered along the coast."""
    L = lights_all()
    lx = np.array([[r["lon"], r["lat"]] for r in L])
    picked = {}                                   # light index -> (source rank, name)
    for x, y, nm, src in names_all():
        d = np.hypot((lx[:, 0] - x) * np.cos(np.radians(y)), lx[:, 1] - y) * 111_000
        near = np.flatnonzero(d < MATCH_M)
        if not len(near):
            continue
        i = int(max(near, key=lambda j: (L[j]["range"], -d[j])))   # main light: seen farthest
        if i not in picked or src < picked[i][0]:
            picked[i] = (src, nm)
    best = {}                                     # one entry per lighthouse name
    for i, (src, nm) in picked.items():
        r = dict(L[i], name=nm)
        k = short(nm).lower()
        if k not in best or r["range"] > best[k]["range"]:
            best[k] = r
    lh = list(best.values())
    for r in lh:
        r["region"] = region(r)
    ordered = []
    for reg, key in REGIONS:
        ordered += sorted([r for r in lh if r["region"] == reg], key=key)
    return ordered


# ---------------- PNG: barcode rings ----------------
def ring(ax, r, rad0=0.62, rad1=1.0):
    """30 s of the light's rhythm around a clock face, clockwise from 12."""
    col = LIGHT[r["colours"][0]]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(t, rad0, rad1, color=DARK["ctl"], lw=0)       # the dark
    t, k = 0.0, 0
    while t < WINDOW:
        on, d = r["seq"][k % len(r["seq"])]
        if on:
            w = max(min(d, WINDOW - t), 0.06)                        # keep the briefest flash visible
            ax.bar(np.radians(t / WINDOW * 360), rad1 - rad0, width=np.radians(w / WINDOW * 360),
                   bottom=rad0, align="edge", color=col, lw=0)
        t += d
        k += 1
    ax.set_ylim(0, 1.0)
    ax.set_axis_off()


def short(name):
    for s in (" and Directional", " Lighthouse", " Light Station", " Light"):
        name = name.replace(s, "")
    return name


def draw_png(lh, stats):
    """Every lighthouse as a ring, walking the coast region by region, so neighbours sit side by side."""
    S.apply()
    plt.rcParams.update({"text.color": DARK["ink"], "figure.facecolor": DARK["bg"], "savefig.facecolor": DARK["bg"]})
    cols = 14 if len(lh) <= 220 else 20      # more lighthouses: more, smaller rings
    f = 14 / cols
    # layout: a heading line per region, then its rings in rows of `cols`
    lines = []
    for reg, _ in REGIONS:
        rs = [r for r in lh if r["region"] == reg]
        if not rs:
            continue
        lines.append(("head", reg))
        lines += [("rings", rs[i:i + cols]) for i in range(0, len(rs), cols)]
    HEAD_H, RING_H, TOP_H, BOT_H = 0.42, 0.3 + 1.2 * f, 1.75, 0.45
    W = 16
    H = TOP_H + BOT_H + sum(HEAD_H if k == "head" else RING_H for k, _ in lines)
    fig = plt.figure(figsize=(W, H))
    cw = 0.96 / cols
    y = H - TOP_H
    for kind, item in lines:
        if kind == "head":
            y -= HEAD_H
            fig.text(0.02, (y + 0.08) / H, item, fontsize=13, fontweight="semibold", fontfamily=S.HEAD_FONT,
                     color=DARK["ink"], va="bottom")
            continue
        y -= RING_H
        for j, r in enumerate(item):
            cx = 0.02 + cw * j
            size = 0.86 * f / W                        # ring diameter, inches -> figure fraction
            ax = fig.add_axes([cx + cw / 2 - size / 2, (y + 0.56 * f + 0.02) / H, size, 0.86 * f / H], projection="polar")
            ax.set_facecolor(DARK["bg"])
            ring(ax, r)
            fig.text(cx + cw / 2, (y + 0.2 + 0.16 * f) / H, textwrap.fill(short(r["name"]), 19), ha="center", va="top",
                     fontsize=7.6 * max(f, 0.85), color=DARK["ink"], linespacing=1.15)
            fig.text(cx + cw / 2, (y + 0.02) / H, r["label"], ha="center", va="bottom", fontsize=7 * max(f, 0.85),
                     color=DARK["muted"])
    fig.text(0.5, 1 - 0.55 / H, stats["headline"], ha="center", va="center",
             fontsize=S.SIZE["title"] + 4, fontweight="bold", fontfamily=S.HEAD_FONT, color=DARK["ink"])
    fig.text(0.5, 1 - 1.22 / H,
             "Every lighthouse has an official signal, its own rhythm of flashes, so sailors can tell which one they are "
             "looking at.\nEach ring is 30 seconds of one lighthouse's signal, clockwise from the top. Below each name, "
             "its chart code: Fl = flashing, Oc = mostly on, Iso = equal on and off, F = steady, (2) = groups of two.",
             ha="center", va="center", fontsize=S.SIZE["subtitle"] + 2, color=DARK["muted"], linespacing=1.5)
    fig.text(0.5, 0.2 / H,
             f"Named lighthouses matched to the charted light within {MATCH_M} m; ordered along each coast  ·  "
             f"Data: {SOURCE}; names © OpenStreetMap contributors and NOAA  ·  #30DayMapChallenge {DAY}",
             ha="center", fontsize=S.SIZE["credit"] + 1, color=DARK["muted"])
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


# ---------------- page ----------------
def land_geojson():
    st = gpd.read_file(HERE.parent / "day08-utopia" / "data" / "raw" / "states" / "states.zip")[["NAME", "geometry"]]
    st["geometry"] = st.geometry.simplify(0.01)
    return json.loads(st.to_json(drop_id=True))


def main():
    lh = lighthouses()
    sigs = Counter((r["label"], tuple(r["seq"])) for r in lh)
    top_label, top_n = Counter(r["label"] for r in lh).most_common(1)[0]
    stats = dict(n=len(lh), n_sig=len(sigs), top=top_label, top_n=top_n,
                 one_of_a_kind=sum(1 for v in sigs.values() if v == 1))
    stats["headline"] = f"{len(lh)} lighthouses, {len(sigs)} ways to blink"
    print(json.dumps(stats, indent=1))
    print("regions", Counter(r["region"] for r in lh))
    print("signals", Counter(r["label"] for r in lh).most_common(8))
    print("families", Counter(LITCHR.get(r["a"].get("LITCHR"), "?") for r in lh).most_common())

    draw_png(lh, stats)

    page = [dict(
        x=round(r["lon"], 4), y=round(r["lat"], 4), n=short(r["name"]), reg=r["region"],
        c=r["colours"][0], cs="".join(r["colours"]), l=r["label"], r=r["range"] or None, p=round(r["period"], 2),
        s=[round(t if on else -t, 2) for on, t in r["seq"]],      # +on / -off durations
    ) for r in lh]
    data = dict(lights=page, land=land_geojson(), colors=LIGHT, colorNames=LIGHT_NAME, stats=stats,
                title=stats["headline"], day=DAY, source=SOURCE, sourceUrl=SOURCE_URL)
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    (HERE / "web" / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
