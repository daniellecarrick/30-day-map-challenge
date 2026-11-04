"""Day 5 · Senses: Sight — Most of New York's street henges vanish when the trees leaf out.

Twice a year the setting sun lines up with Manhattan's cross streets ("Manhattanhenge"). Every
straight street has its own two dates; whether you can SEE the sun down the street depends on
what's at the end of it: buildings, hills and, from late April to mid-November, leaves.
henge.py does the astronomy and the building/terrain line of sight; trees.py adds tree crowns
and the leaf season. Writes:
  - out/nyc_henge.png    (static: bare branches vs leaves out, streets colored by henge dates)
  - out/nyc_henge.html   (interactive: pick a day, see which streets frame the sun)

Data: NYC Open Data street centerlines (inkn-q76z), building footprints (5zhs-2jue), Forestry
      Tree Points (hn5i-inap), Parks Properties (enfh-gkve); Copernicus GLO-30 surface model.
      Sun positions computed (NOAA). Street View spot checks: data/spotcheck*.csv.
Set aside for the sky map (build.py), kept for reference.
Run:  ./fetch.sh, then ../.venv/bin/python build_henge.py   (first run ~5 min; cached in data/raw/)
"""
import datetime as dt
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402
import henge as H  # noqa: E402
import trees as T  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 5: Senses: Sight"
SOURCE = "NYC Open Data (streets, buildings, trees, parks), Copernicus GLO-30"
SOURCE_URL = "https://data.cityofnewyork.us/d/5zhs-2jue"
BORO = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens", "5": "Staten Island"}
Y0 = dt.date(H.YEAR, 1, 1)
DEC21 = (dt.date(H.YEAR, 12, 21) - Y0).days     # 354
JUN21 = (dt.date(H.YEAR, 6, 21) - Y0).days      # 171

# Color: diverging around the equinox (a street running due east-west). Each street's sunset (or
# sunrise) henge comes as a pair of dates mirrored around the June solstice (Dec for sunrise);
# the color is how far its spring date is from Dec 21: blue = winter half (sun sets south of
# west), amber -> brown = summer half (north of west). There is no "neutral" season, so the scale
# steps from light blue to light amber at the equinox instead of passing through the pale
# neutral, which would vanish on white (and match the never-lines-up streets).
SPAN = 182
EQUINOX = (dt.date(H.YEAR, 3, 20) - dt.date(H.YEAR - 1, 12, 21)).days   # 89
STOPS = [(0, S.DIVERGING[0]), (EQUINOX / 2, S.DIVERGING[1]), (EQUINOX - 0.01, S.SEQUENTIAL_BLUE[0]),
         (EQUINOX + 0.01, S.SEQUENTIAL_AMBER[0]), ((EQUINOX + SPAN) / 2, S.DIVERGING[3]), (SPAN, S.DIVERGING[4])]
color_for = S.make_scale(STOPS)
NONE_COLOR = S.TOKENS["color"]["light"]["ctl"]   # streets that never line up (north-south)

ABBR = [(r"\bAv\b", "Ave"), (r"\bBl\b", "Blvd"), (r"\bPkwy\b", "Pkwy"), (r"^W\b", "W"), (r"^E\b", "E")]


def nice(s) -> str:
    """'W 42 ST' -> 'W 42nd St'."""
    if not isinstance(s, str) or not s.strip():
        return "Unnamed street"
    s = re.sub(r"\s+", " ", s.strip()).title()
    for pat, rep in ABBR:
        s = re.sub(pat, rep, s)

    def ordinal(m):
        n = int(m.group(1))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    return re.sub(r"\b(\d+)\b(?= (St|Ave|Rd|Dr|Pl|Ter|Blvd|Ln|Way)\b)", ordinal, s)


def doy(d) -> int:
    return -1 if pd.isna(d) else (pd.Timestamp(d).date() - Y0).days % 365   # Dec 31 of the year before -> Dec 31


def fmt_day(k: int) -> str:
    d = Y0 + dt.timedelta(days=int(k) % 365)
    return f"{d:%b} {d.day}"


def spring_offset(d1: int, d2: int, sunset: bool) -> float:
    """Color value: days from Dec 21 to the pair's date in the half-year when the sun is moving
    north (Dec 21 -> Jun 21). Both sunset and sunrise use the same 0..182 scale."""
    ds = [d for d in (d1, d2) if d >= 0]
    if not ds:
        return np.nan
    rising = [d for d in ds if d <= JUN21 or d >= DEC21]
    d = rising[0] if rising else ds[0]
    return float((d - DEC21) % 365)


def prepare():
    s = T.compute()
    s["street"] = s.name.map(nice)
    s["boro"] = s.boroughcode.map(BORO)
    for end in "we":
        for k in (1, 2):
            s[f"{end}{k}"] = s[f"{end}{k}_date"].map(doy)
        s[f"t_{end}"] = [spring_offset(a, b, end == "w") for a, b in zip(s[f"{end}1"], s[f"{end}2"])]
    s["henge"] = s.t_w.notna() | s.t_e.notna()
    return s


def per_day_counts(s, end):
    """How many distinct streets (runs) see the sun in their frame, per day (leaf season included)."""
    seen = [set() for _ in range(365)]
    for run, runs in zip(s.run, s[f"{end}_days"]):
        for a, b, vis in runs:
            if vis:
                for d in range(a, b + 1):
                    seen[d].add(run)
    return np.array([len(x) for x in seen])


# ---------------------------------------------------------------- PNG ----------------------
def draw_png(s, stats):
    """Two panels: the same sunset henges seen past bare branches, and with leaves out."""
    S.apply()
    fig = plt.figure(figsize=S.MAP["png_size_in"])
    x0, y0, x1, y1 = s.total_bounds
    has = s.t_w.notna()
    panels = [("skyoff_w", "Bare branches", stats["sub_off"]), ("skyon_w", "Leaves out", stats["sub_on"])]
    axes = []
    for k, (col, head, sub) in enumerate(panels):
        ax = fig.add_axes([0.005 + 0.5 * k, 0.05, 0.49, 0.80])
        ok = has & (s[col] <= H.MAX_SKYLINE)

        def lines(g, color, lw, z):
            segs = [np.asarray(c.coords) for geom in g.geometry for c in getattr(geom, "geoms", [geom])]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=z, capstyle="round"))
        lines(s[~has], NONE_COLOR, 0.25, 1)
        lines(s[has & ~ok], S.CLOSED, 0.25, 2)
        clear = s[ok].sort_values("t_w")
        lines(clear, [color_for(v) for v in clear.t_w], 0.55, 3)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_axis_off()
        S.panel_heading(ax, head, sub)
        axes.append(ax)

    S.title(fig, stats["title"])
    fig.text(0.5, 0.925, stats["subtitle"], ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, f"{stats['how']}\nData: {SOURCE}  ·  #30DayMapChallenge {DAY}", y=0.008)

    # legend: left panel, under its heading (New Jersey, empty on this map)
    lx, ly = 0.02, 0.69
    lax = fig.add_axes([lx, ly, 0.17, 0.014])
    grad = np.linspace(0, SPAN, 256)
    lax.imshow([[plt.matplotlib.colors.to_rgb(color_for(v)) for v in grad]], aspect="auto", extent=[0, SPAN, 0, 1])
    lax.set_yticks([])
    tk = [stats["ticks"][i] for i in (0, 2, 4)]
    lax.set_xticks([t for t, _ in tk])
    lax.set_xticklabels([lab.replace(" · ", "\n") for _, lab in tk], fontsize=S.SIZE["legend"] - 1)
    lax.tick_params(length=3, color=S.MUTED, pad=2)
    for sp in lax.spines.values():
        sp.set_visible(False)
    fig.text(lx, ly + 0.026, "Sunset henge dates (each street gets two)", fontsize=S.SIZE["legend"], fontweight="semibold")
    for i, (col, lab, lw) in enumerate(((S.CLOSED, "Lines up, but the view is blocked", 1.2),
                                        (NONE_COLOR, "Runs too north–south to line up", 2.4))):
        ky = ly - 0.065 - 0.025 * i
        fig.add_artist(plt.Line2D([lx, lx + 0.016], [ky + 0.004] * 2, color=col, lw=lw, transform=fig.transFigure))
        fig.text(lx + 0.022, ky, lab, fontsize=S.SIZE["legend"] - 0.5, color=S.INK)

    for c in stats["callouts"]:
        S.callout(axes[c["panel"]], c["xy"], c["text_xy"], c["heading"], c["detail"], ha=c.get("ha", "left"), halo=True)
    fig.savefig(OUT / "nyc_henge.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)


# ---------------------------------------------------------------- web data ------------------
def web_data(s, stats):
    x0, y0 = np.floor(s.total_bounds[:2])

    def flat(geom, tol):
        g = shapely.simplify(geom, tol)
        parts = getattr(g, "geoms", [g])
        return [np.round(shapely.get_coordinates(p) - [x0, y0]).astype(int).ravel().tolist() for p in parts]

    bg = shapely.line_merge(shapely.multilinestrings(list(s[~s.henge].geometry.values)))
    bg_lines = []
    for p in getattr(bg, "geoms", [bg]):
        bg_lines += flat(p, 4)

    h = s[s.henge].reset_index(drop=True)
    names = sorted(set(zip(h.street, h.boro)))
    name_ix = {n: i for i, n in enumerate(names)}
    kinds = {"bldg": 0, "ground": 1, "dsm": 2, "tree": 3, "park": 4}
    species = sorted({w.split(":")[2] for e in "we" for k in (1, 2) for w in h[f"{e}{k}_why"] if w.count(":") >= 2})
    sp_ix = {n: i for i, n in enumerate(species)}

    def why(w):
        """'tree:120:london planetree' -> [kind, metres, species index]; '' -> []"""
        if not w:
            return []
        p = w.split(":")
        return [kinds.get(p[0], -1), int(float(p[1])), sp_ix.get(p[2], -1) if len(p) > 2 else -1]

    seg = {"c": [flat(g, 1.5)[0] for g in h.geometry], "n": [name_ix[k] for k in zip(h.street, h.boro)],
           "r": h.run.astype(int).tolist()}
    for end in "we":
        seg[end] = {
            "d1": h[f"{end}1"].tolist(), "d2": h[f"{end}2"].tolist(),
            "tm1": [-1 if np.isnan(t) else int(round(t * 60)) for t in h[f"{end}1_time"]],
            "tm2": [-1 if np.isnan(t) else int(round(t * 60)) for t in h[f"{end}2_time"]],
            "t": [None if np.isnan(t) else round(t) for t in h[f"t_{end}"]],
            "ok1": h[f"{end}1_ok"].astype(int).tolist(), "ok2": h[f"{end}2_ok"].astype(int).tolist(),
            "lv1": h[f"{end}1_leaves"].astype(int).tolist(), "lv2": h[f"{end}2_leaves"].astype(int).tolist(),
            "why1": [why(w) for w in h[f"{end}1_why"]], "why2": [why(w) for w in h[f"{end}2_why"]],
            "days": [[x for run in r for x in run] for r in h[f"{end}_days"]],
        }

    return {
        "title": stats["title"], "lede": stats["lede"], "how": stats["how"],
        "day": DAY, "source": SOURCE, "source_url": SOURCE_URL,
        "extent": [float(s.total_bounds[2] - x0), float(s.total_bounds[3] - y0)],
        "bg": bg_lines, "seg": seg, "names": [list(n) for n in names], "species": species,
        "leaf_on": list(T.LEAF_ON),
        "counts": {"w": stats["counts_w"].tolist(), "e": stats["counts_e"].tolist()},
        "stops": STOPS, "ticks": stats["ticks"], "equinox": EQUINOX, "span": SPAN,
        "year": H.YEAR, "max_sky": H.MAX_SKYLINE,
    }


def ticks():
    """Legend ticks at spring dates, each labelled with its partner date in the fall (sunset
    pairs mirror around Jun 21; the partner is found from the sun's declination)."""
    out = []
    jd0 = 2461406.5
    dec = np.degrees(H._sun(jd0 + np.arange(365) + 0.5)[0])
    for m, d in ((12, 21), (2, 5), (3, 20), (5, 5), (6, 21)):
        day = dt.date(H.YEAR - 1 if m == 12 else H.YEAR, m, d)
        t = (day - dt.date(H.YEAR - 1, 12, 21)).days
        if m in (12, 6):
            out.append((t, f"{day:%b} {day.day}"))
            continue
        k = (day - Y0).days
        later = np.arange(JUN21, DEC21)
        partner = later[np.abs(dec[later] - dec[k]).argmin()]
        out.append((t, f"{day:%b} {day.day} · {fmt_day(partner)}"))
    return out


def main():
    s = prepare()
    cw, ce = per_day_counts(s, "w"), per_day_counts(s, "e")
    assert (cw > 0).all(), "lede says some street frames the sunset every day"
    has = s[s.t_w.notna()]
    share_off = (has.skyoff_w <= H.MAX_SKYLINE).mean()
    share_on = (has.skyon_w <= H.MAX_SKYLINE).mean()
    spot = pd.concat([pd.read_csv(DATA / "spotcheck.csv"), pd.read_csv(DATA / "spotcheck_holdout.csv")])
    judged = spot[spot.match.isin(["yes", "NO"])]
    n_ok, n_judged = int((judged.match == "yes").sum()), len(judged)

    def pair(g):
        """Most common sunset date pair among blocks that SEE each date, e.g. 'May 30 & Jul 12'
        (blocked blocks carry open-horizon dates, which would skew the mode)."""
        a, b = g[g.w1_ok].w1.mode().iat[0], g[g.w2_ok].w2.mode().iat[0]
        return a, b, f"{fmt_day(a)} & {fmt_day(b)}"

    def target(g, near=None):
        """Midpoint of the block nearest the group's middle (or nearest a lon/lat)."""
        mid = shapely.line_interpolate_point(g.geometry.values, 0.5, normalized=True)
        xy = shapely.get_coordinates(mid)
        c = np.median(xy, 0) if near is None else np.array(H.to_m.transform(*near))
        return tuple(xy[np.argmin(((xy - c) ** 2).sum(1))])

    man = has[(has.boro == "Manhattan") & (has.west_az.sub(299).abs() < 0.3)]
    bk_az = has[has.boro == "Brooklyn"].west_az.round().mode().iat[0]
    bk = has[(has.boro == "Brooklyn") & (has.west_az.round() == bk_az)]
    *_, man_p = pair(man)
    b1, b2, _ = pair(bk)
    bk1, bk2 = bk.w1_ok.mean(), bk.w2_ok.mean()   # share of the grid's blocks that see each henge
    wide = man[man.street.isin(["W 34th St", "W 42nd St", "W 57th St", "E 34th St", "E 42nd St", "E 57th St"])]
    bk_xy = target(bk, near=(-73.955, 40.655))
    callouts = [
        {"panel": 0, "xy": bk_xy, "text_xy": (0.03, 0.43), "ha": "left",
         "heading": f"Brooklyn's grid, {fmt_day(b1)} (bare branches)",
         "detail": f"{bk1:.0%} of its blocks see the sunset"},
        {"panel": 1, "xy": bk_xy, "text_xy": (0.03, 0.43), "ha": "left",
         "heading": f"Same streets, {fmt_day(b2)} (leaves out)",
         "detail": f"{bk2:.0%} of its blocks see the sunset"},
        {"panel": 1, "xy": target(man[man.street == "W 42nd St"]), "text_xy": (0.03, 0.66), "ha": "left",
         "heading": f"Manhattanhenge, {man_p}",
         "detail": f"Wide 34th, 42nd and 57th Sts\nstay clear ({wide.w1_ok.mean():.0%} of blocks)"},
    ]
    stats = {
        "counts_w": cw, "counts_e": ce, "ticks": ticks(), "callouts": callouts,
        "sub_off": f"{share_off:.0%} of street blocks see their henge sunset",
        "sub_on": f"{share_on:.0%} do",
        "title": "Most of New York's street henges vanish when the trees leaf out",
        "subtitle": ("Every straight street lines up with the setting sun on two days a year (Manhattanhenge is the "
                     f"famous one). Past bare branches, {share_off:.0%} of blocks would see it; with leaves out, {share_on:.0%}."),
        "lede": ("Every straight street in New York lines up with the setting or rising sun on two days a year; "
                 "Manhattanhenge is just the famous one. Whether you can see it depends on what's at the end of the "
                 "street: buildings, hills and, from late April to mid-November, leaves. Pick a day to see which "
                 "streets frame the sun."),
        "how": ("A henge: the sun, resting on the skyline seen down the street, sits on the street's line. Skyline from "
                "building heights, terrain and 898k city trees (crown size from trunk size; leaves out Apr 20 – Nov 15). "
                f"Agreed with Street View in {n_ok} of {n_judged} spot checks."),
    }
    print(f"bare {share_off:.2f} vs leaves {share_on:.2f}; Brooklyn {fmt_day(b1)} {bk1:.2f} / {fmt_day(b2)} {bk2:.2f}; "
          f"Manhattan {man_p}, wide streets {wide.w1_ok.mean():.2f}; spot checks {n_ok}/{n_judged}; "
          f"min streets/day {cw.min()}")
    draw_png(s, stats)
    data = web_data(s, stats)
    size = S.build_page(HERE / "web" / "henge.html", data, OUT / "nyc_henge.html")
    print("wrote out/nyc_henge.png and out/nyc_henge.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
