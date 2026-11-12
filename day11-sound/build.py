"""Day 11 · Senses: Sound — what each NYC neighborhood complains about, and when.

Every 2025 311 noise complaint, grouped into 8 sounds, counted as complaint-days (an address counts
once per day per sound, see prep.py). Each community district's "signature sound" is the one it
complains about most above the city average (location quotient), among sounds at least 50 different
addresses there complained about.
Writes:
  - out/sound.png    (static: signature-sound map + when each sound peaks)
  - out/sound.html   (interactive: 24-hour timelapse you can listen to; the cursor is a microphone)

Data: NYC 311 Service Requests (NYC Open Data, erm2-nwe9), Community Districts (5crt-au7u).
Run:  sh fetch.sh; ../.venv/bin/python prep.py; ../.venv/bin/python sounds.py; ../.venv/bin/python build.py
"""
import base64
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

from prep import SOUNDS  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 11: Senses: Sound"
SOURCE_URL = "https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9"
SOURCE = "Data: NYC 311 Service Requests, 2025 (NYC Open Data)"
CRS_M = "EPSG:32618"
MIN_ADDR, MIN_DAYS = 50, 100     # a signature sound needs this many addresses and complaint-days
KEYS = ["music", "bang", "talk", "horn", "build", "heli", "dogs", "icecream"]   # audio file per SOUNDS entry
CLIP_CREDITS = ("Clips: dog barking by Amada44 (CC BY-SA 3.0), helicopter by Doggo19292, car horn by "
                "15HPanska_Ruttner_Jan (CC0), crowd by SoundSnap (CC0), all via Wikimedia Commons; "
                "music, banging, jackhammer and jingle synthesized.")

NAMES = {
    101: "Financial District & Tribeca", 102: "Greenwich Village & SoHo", 103: "Lower East Side & Chinatown",
    104: "Chelsea & Hell's Kitchen", 105: "Midtown", 106: "Murray Hill & Stuyvesant Town",
    107: "Upper West Side", 108: "Upper East Side", 109: "Morningside & Hamilton Heights",
    110: "Central Harlem", 111: "East Harlem", 112: "Washington Heights & Inwood",
    201: "Mott Haven & Melrose", 202: "Hunts Point & Longwood", 203: "Morrisania & Crotona",
    204: "Highbridge & Concourse", 205: "Fordham & University Heights", 206: "Belmont & East Tremont",
    207: "Kingsbridge Heights & Bedford Park", 208: "Riverdale & Fieldston", 209: "Parkchester & Soundview",
    210: "Throgs Neck & Co-op City", 211: "Morris Park & Pelham Parkway", 212: "Williamsbridge & Wakefield",
    301: "Williamsburg & Greenpoint", 302: "Downtown Brooklyn & Fort Greene", 303: "Bedford-Stuyvesant",
    304: "Bushwick", 305: "East New York & Starrett City", 306: "Park Slope & Carroll Gardens",
    307: "Sunset Park & Windsor Terrace", 308: "Crown Heights North & Prospect Heights",
    309: "Crown Heights South & Lefferts Gardens", 310: "Bay Ridge & Dyker Heights",
    311: "Bensonhurst & Bath Beach", 312: "Borough Park & Kensington", 313: "Coney Island & Brighton Beach",
    314: "Flatbush & Midwood", 315: "Sheepshead Bay & Gravesend", 316: "Brownsville & Ocean Hill",
    317: "East Flatbush", 318: "Canarsie & Flatlands",
    401: "Astoria & Long Island City", 402: "Sunnyside & Woodside", 403: "Jackson Heights & North Corona",
    404: "Elmhurst & South Corona", 405: "Ridgewood & Maspeth", 406: "Forest Hills & Rego Park",
    407: "Flushing & Whitestone", 408: "Hillcrest & Fresh Meadows", 409: "Kew Gardens & Woodhaven",
    410: "Howard Beach & Ozone Park", 411: "Bayside & Little Neck", 412: "Jamaica & Hollis",
    413: "Queens Village & Rosedale", 414: "The Rockaways",
    501: "North Shore, Staten Island", 502: "Mid-Island, Staten Island", 503: "South Shore, Staten Island",
}


def clock(h):
    return "midnight" if h == 0 else "noon" if h == 12 else f"{h % 12} {'am' if h < 12 else 'pm'}"


def load():
    d = pd.read_csv(DATA / "complaint_days.csv.gz")
    d["s"] = d.sound.map({s: i for i, s in enumerate(SOUNDS)})
    days = d.pivot_table(index="boro_cd", columns="s", values="hour", aggfunc="size", fill_value=0)
    addrs = d.groupby(["boro_cd", "s"])["where"].nunique().unstack(fill_value=0)
    days, addrs = days.reindex(columns=range(8), fill_value=0), addrs.reindex(columns=range(8), fill_value=0)
    city = days.sum() / days.values.sum()
    lq = days.div(days.sum(1), axis=0) / city
    ok = (addrs >= MIN_ADDR) & (days >= MIN_DAYS)
    sig = lq.where(ok).idxmax(1).astype(int)
    # index shown on the page: too few addresses to trust -> no higher than the city average,
    # so the tallest bar is always the district's signature sound (its map color)
    lq_shown = lq.where(ok, np.minimum(lq, 1.0))
    hourly = np.zeros((len(days), 24, 8), int)   # district x hour x sound
    idx = {cd: i for i, cd in enumerate(days.index)}
    for (cd, h, s), n in d.groupby(["boro_cd", "hour", "s"]).size().items():
        hourly[idx[cd], h, s] = n

    gdf = gpd.read_file(DATA / "cd.geojson").to_crs(CRS_M)
    gdf["boro_cd"] = gdf.boro_cd.astype(int)
    gdf["res"] = gdf.boro_cd.isin(days.index)
    gdf = gdf.sort_values("boro_cd").reset_index(drop=True)
    gdf["name"] = gdf.boro_cd.map(NAMES).fillna("Park or airport (no residents)")
    gdf["sig"] = gdf.boro_cd.map(sig)
    gdf["lq"] = gdf.apply(lambda r: lq.loc[r.boro_cd, r.sig] if r.res else np.nan, axis=1)
    gdf["top"] = gdf.boro_cd.map(days.idxmax(1))
    gdf["total"] = gdf.boro_cd.map(days.sum(1))
    gdf["idx"] = gdf.boro_cd.map(lambda cd: [round(float(v), 2) for v in lq_shown.loc[cd]] if cd in lq_shown.index else None)
    return gdf, days, lq, city, hourly, idx


def sample_points(geom, n=24, seed=0):
    """Random points inside a district: where ripples start."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = geom.bounds
    pts = []
    while len(pts) < n:
        p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
        if geom.contains(p):
            pts.append([round(p.x), round(p.y)])
    return pts


def png(gdf, days, city, hourly, title):
    S.apply()
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    ax = fig.add_axes([0.02, 0.06, 0.62, 0.84])
    res, other = gdf[gdf.res], gdf[~gdf.res]
    other.plot(ax=ax, color="#e4e3df", edgecolor=S.EDGE, linewidth=S.MAP["cell_edge_width"])
    res.plot(ax=ax, color=[S.CATEGORICAL[int(i)] for i in res.sig], edgecolor=S.EDGE, linewidth=0.8)
    gdf.dissolve(by=gdf.boro_cd // 100).boundary.plot(ax=ax, color=S.INK, linewidth=S.MAP["boundary_width"])
    ax.set_axis_off()
    ax.set_aspect("equal")

    def at(cd):
        return gdf.loc[gdf.boro_cd == cd]

    def call(cd, xy, head, ha="left"):
        r = at(cd).iloc[0]
        c = r.geometry.representative_point()
        S.callout(ax, (c.x, c.y), xy, head, f"{r['name']}: {r.lq:.0f}× the city share", ha=ha,
                  outline=at(cd).geometry)

    call(107, (0.06, 0.80), "Helicopters")
    call(211, (0.80, 0.93), "Ice cream trucks")
    call(503, (0.40, 0.04), "Dogs")
    top = gdf[gdf.res & (gdf.sig == SOUNDS.index("Construction & machines"))].sort_values("lq").iloc[-1]
    call(int(top.boro_cd), (0.03, 0.55), "Construction")

    # legend: signature sounds, with the number of districts each
    n_sig = gdf[gdf.res].sig.value_counts()
    lx, ly = 0.87, 0.29
    ax.text(lx, ly + 0.04, "Signature sound", transform=ax.transAxes, fontsize=S.SIZE["legend"] + 0.5,
            fontweight="semibold", color=S.INK)
    ax.text(lx, ly + 0.012, "complained about most above the city average", transform=ax.transAxes,
            fontsize=S.SIZE["legend"] - 1, color=S.MUTED)
    for k, i in enumerate(sorted(int(v) for v in n_sig.index)):
        y = ly - 0.035 * (k + 0.7)
        ax.add_patch(plt.Rectangle((lx, y - 0.008), 0.022, 0.022, transform=ax.transAxes, color=S.CATEGORICAL[i]))
        ax.text(lx + 0.032, y + 0.003, f"{SOUNDS[i]}  ({n_sig[i]})", transform=ax.transAxes, va="center",
                fontsize=S.SIZE["legend"], color=S.INK)

    # right panel: when each sound peaks (share of that sound's complaints by hour)
    cityh = hourly.sum(0)   # hour x sound
    pk = cityh / cityh.sum(0)
    fig.text(0.68, 0.875, "When each sound peaks", fontsize=S.SIZE["panel_title"], fontweight="semibold",
             color=S.INK, fontfamily=S.HEAD_FONT)
    fig.text(0.68, 0.85, "Complaints by hour of day, each sound scaled to its own peak", fontsize=S.SIZE["subtitle"],
             color=S.MUTED)
    order = np.argsort(pk.argmax(0) + (pk.argmax(0) < 5) * 24)  # by peak hour, 5am first
    rowh = 0.088
    for k, s in enumerate(order):
        y0 = 0.76 - k * rowh
        a = fig.add_axes([0.68, y0, 0.29, rowh * 0.7])
        v = pk[:, s] / pk[:, s].max()
        a.fill_between(range(24), v, step="mid", color=S.CATEGORICAL[s], alpha=0.9, linewidth=0)
        a.set_xlim(-0.5, 23.5)
        a.set_ylim(0, 1.05)
        a.set_yticks([])
        for sp in ("top", "right", "left"):
            a.spines[sp].set_visible(False)
        a.spines["bottom"].set_color(S.MUTED)
        a.spines["bottom"].set_linewidth(0.5)
        h = int(pk[:, s].argmax())
        a.text(-0.5, 1.12, SOUNDS[s], fontsize=S.SIZE["legend"], fontweight="semibold", color=S.INK,
               transform=a.get_xaxis_transform())
        a.text(23.5, 1.12, f"peaks {clock(h)}", ha="right", fontsize=S.SIZE["legend"] - 0.5, color=S.MUTED,
               transform=a.get_xaxis_transform())
        a.set_xticks([0, 6, 12, 18, 23] if k == len(order) - 1 else [])
        if k == len(order) - 1:
            a.set_xticklabels(["12 am", "6 am", "noon", "6 pm", "11 pm"], fontsize=S.SIZE["legend"] - 1,
                              color=S.MUTED)
        a.tick_params(length=0)

    S.title(fig, title)
    fig.text(0.5, 0.925, "Loud music and parties are half of New York's noise complaints. Coloring each district by the "
             "sound it reports most above the city average shows what else keeps it up.",
             ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, "Each address counts once per day per sound; 59 community districts; a signature sound needs "
             f"{MIN_ADDR}+ complaining addresses.  ·  {SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "sound.png", dpi=S.MAP["png_dpi"])


def main():
    gdf, days, lq, city, hourly, idx = load()
    n_music = int((days.idxmax(1) == 0).sum())
    n = len(days)
    title = f"Loud music tops noise complaints in {n_music} of {n} NYC districts. What else they hear sets them apart"
    print(title)
    print(gdf[gdf.res].groupby("sig").size().rename(index=dict(enumerate(SOUNDS))))
    png(gdf, days, city, hourly, title)

    # ---- interactive page ----
    g = gdf.copy()
    g["geometry"] = g.geometry.simplify(25)
    feats = []
    for _, r in g.iterrows():
        c = r.geometry.representative_point()
        f = {"type": "Feature", "geometry": json.loads(gpd.GeoSeries([r.geometry]).to_json())["features"][0]["geometry"],
             "properties": {"cd": int(r.boro_cd), "name": r["name"], "res": bool(r.res)}}
        if r.res:
            i = idx[r.boro_cd]
            f["properties"].update(sig=int(r.sig), lq=round(float(r.lq), 1), idx=r.idx, total=int(r.total),
                                   cx=round(c.x), cy=round(c.y), h=hourly[i].tolist(),
                                   pts=sample_points(gdf.geometry[_], seed=int(r.boro_cd)))
        feats.append(f)
    audio = {k: "data:audio/mp4;base64," + base64.b64encode((DATA / "audio" / f"{k}.m4a").read_bytes()).decode()
             for k in KEYS}
    data = {
        "features": {"type": "FeatureCollection", "features": feats},
        "sounds": [{"label": s, "key": k, "share": round(float(city[i]), 4)} for i, (s, k) in enumerate(zip(SOUNDS, KEYS))],
        "audio": audio, "makeup": json.loads((DATA / "audio" / "gains.json").read_text()), "title": title, "day": DAY, "source": SOURCE, "source_url": SOURCE_URL,
        "clip_credits": CLIP_CREDITS, "n": n, "n_music": n_music,
        "city_hour": hourly.sum(0).tolist(),
    }
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "sound.html")
    print("wrote out/sound.png and out/sound.html", f"({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
