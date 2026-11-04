"""Day 5 · trees in the sightline (used by build.py, after henge.py's building + terrain pass).

Looking down a street at the low sun, a street tree blocks it if its crown reaches over the
line of sight and the sun sits between the crown's lowest branches and its top. Each living tree
in NYC Parks' Forestry Tree Points gets a crown estimated from its trunk diameter (generic urban
allometry); the crown is a vertical cylinder from crown base to treetop.

Leaves matter: a deciduous tree is treated as see-through while bare (LEAF_OFF dates) and solid
in leaf; evergreens are solid all year. So each street end gets three skylines: buildings and
terrain only (henge.py), plus evergreens (leaf-off), plus every tree (leaf-on). A henge date is
judged with the skyline of its season.

Cached in data/raw/henge_trees.pkl; delete it (or pass force=True) to recompute.
"""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

import henge as H

CACHE = H.RAW / "henge_trees.pkl"
TREE_KM = 1.5            # trees farther than this sit far below a ~1 deg sun
YEAR0 = dt.date(H.YEAR, 1, 1)
LEAF_ON = ((dt.date(H.YEAR, 4, 20) - YEAR0).days, (dt.date(H.YEAR, 11, 15) - YEAR0).days)   # inclusive
EVERGREEN = ("Pinus", "Picea", "Abies", "Tsuga", "Juniperus", "Thuja", "Cedrus", "Taxus", "Chamaecyparis",
             "Cryptomeria", "Pseudotsuga", "Ilex opaca", "Magnolia grandiflora", "Cupressus", "x Cupressocyparis")


def leaf_on(day) -> np.ndarray:
    d = np.asarray(day)
    return (d >= LEAF_ON[0]) & (d <= LEAF_ON[1])


def load_trees(grid: H.NYCGrid) -> pd.DataFrame:
    t = pd.concat([pd.read_csv(p) for p in sorted(H.RAW.glob("trees_*.csv"))], ignore_index=True)
    xy = t.location.str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    t = t[xy[0].notna()].copy()
    t["x"], t["y"] = H.to_m.transform(xy.loc[t.index, 0].values, xy.loc[t.index, 1].values)
    dbh_cm = pd.to_numeric(t.dbh, errors="coerce").fillna(3).clip(1, 60) * 2.54
    # generic urban broadleaf allometry (after McPherson et al. 2016, Urban Tree Database)
    t["r"] = (1.5 + 0.24 * dbh_cm).clip(upper=20) / 2                 # crown radius, m (spot checks: wider than the
                                                                       # generic formula's 1.2 + 0.19 * dbh)
    height = (2.5 + 0.28 * dbh_cm).clip(upper=30)                      # m
    ground = grid.ground_at(t.x.values, t.y.values)
    t["top"] = ground + height
    t["base"] = ground + np.maximum(3.0, 0.3 * height)                 # lowest branches (pruned for clearance)
    t["ever"] = t.genusspecies.fillna("").str.startswith(EVERGREEN)
    t["species"] = t.genusspecies.fillna("tree").str.split(" - ").str[-1].str.strip().str.lower()
    return t[["x", "y", "r", "top", "base", "ever", "species"]].reset_index(drop=True)


PARK_STEP = 10
PARK_MIN_M = 150         # nearer than this the 30 m surface model can't tell park trees from the street
PARK_ERODE = 20          # m trimmed off park edges, so street-side pixels don't count as woodland


def park_mask(grid: H.NYCGrid):
    """Park land (eroded), rasterized on a 10 m grid aligned with the city grid."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    parks = gpd.read_file(H.RAW / "parks.geojson").to_crs(H.CRS_M)
    parks = parks[~parks.typecategory.isin(["Triangle/Plaza", "Garden", "Mall", "Strip"])]
    geoms = [g for g in parks.geometry.buffer(-PARK_ERODE).values if g is not None and not g.is_empty]
    res = 10
    h = int(grid.roof.shape[0] * grid.RES / res) + 1
    w = int(grid.roof.shape[1] * grid.RES / res) + 1
    m = rasterize(geoms, out_shape=(h, w), transform=from_origin(grid.x0, grid.y1, res, res), fill=0,
                  default_value=1, dtype="uint8")
    return m, res


def park_skyline(origins, az_true, eye, grid, dsm, mask, res):
    """Highest angle of park woodland (surface-model canopy over park land) within TREE_KM."""
    n = len(origins)
    back = Transformer.from_crs(H.CRS_M, "EPSG:4326", always_xy=True)
    lon, lat = back.transform(*origins.T)
    L = TREE_KM * 1000
    elon, elat, _ = H.geod.fwd(lon, lat, az_true, np.full(n, L))
    ex, ey = H.to_m.transform(elon, elat)
    ux, uy = (ex - origins[:, 0]) / L, (ey - origins[:, 1]) / L
    steps = np.arange(PARK_MIN_M, L, PARK_STEP, dtype=float)
    best = np.full(n, -90.0); dist = np.zeros(n)
    for k in range(0, n, 4000):
        sl = slice(k, k + 4000)
        px = origins[sl, 0, None] + ux[sl, None] * steps
        py = origins[sl, 1, None] + uy[sl, None] * steps
        r = ((grid.y1 - py) / res).astype(np.int64); c = ((px - grid.x0) / res).astype(np.int64)
        inb = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        park = inb & (mask[np.where(inb, r, 0), np.where(inb, c, 0)] == 1)
        _, is_b, _ = grid.sample(px, py)
        park &= ~is_b
        z = np.full(px.shape, np.nan)
        if park.any():
            z[park] = dsm.sample(px[park], py[park])
            ground = grid.ground_at(px, py)
            z = np.where(park & (z - ground >= 4), z, np.nan)   # canopy at least 4 m tall
        ang = np.degrees(np.arctan2(z - steps ** 2 / (2 * H.K_EARTH) - eye[sl, None], steps))
        ang = np.where(np.isnan(ang), -90, ang)
        j = ang.argmax(1)
        best[sl] = ang[np.arange(ang.shape[0]), j]
        dist[sl] = steps[j]
    return best, dist


def tree_skyline(origins, az_true, eye, sky0, trees: pd.DataFrame, tree_idx, only):
    """Raise each street end's skyline `sky0` (deg) by the tree crowns along its sightline.
    `only`: boolean mask of trees that count (evergreens only for leaf-off). Returns
    (skyline deg, blocker label 'tree:<m>:<species>' or '')."""
    n = len(origins)
    back = Transformer.from_crs(H.CRS_M, "EPSG:4326", always_xy=True)
    lon, lat = back.transform(*origins.T)
    L = TREE_KM * 1000
    elon, elat, _ = H.geod.fwd(lon, lat, az_true, np.full(n, L))
    ex, ey = H.to_m.transform(elon, elat)
    ux, uy = (ex - origins[:, 0]) / L, (ey - origins[:, 1]) / L
    rays = shapely.linestrings(np.stack([origins, np.c_[ex, ey]], axis=1))
    ri, ti = tree_idx.query(rays, predicate="dwithin", distance=float(trees.r.max()))
    keep = only[ti]
    ri, ti = ri[keep], ti[keep]
    dx, dy = trees.x.values[ti] - origins[ri, 0], trees.y.values[ti] - origins[ri, 1]
    along = dx * ux[ri] + dy * uy[ri]
    lat_off = np.abs(dx * uy[ri] - dy * ux[ri])
    r = trees.r.values[ti]
    ok = (along > 0) & (lat_off < r)
    ri, ti, along, half = ri[ok], ti[ok], along[ok], np.sqrt(r[ok] ** 2 - lat_off[ok] ** 2)
    near, far = np.maximum(along - half, 1.0), along + half
    e = eye[ri]
    top_ang = np.degrees(np.arctan2(trees.top.values[ti] - e, near))      # pass over: above this
    base_ang = np.degrees(np.arctan2(trees.base.values[ti] - e, far))     # pass under: below this
    sky = sky0.copy()
    who = np.full(n, "", dtype=object)
    order = np.argsort(ri, kind="stable")
    ri, ti, top_ang, base_ang, along = ri[order], ti[order], top_ang[order], base_ang[order], along[order]
    for _ in range(8):   # raising the sun above one crown can put it into the next one
        sun = sky[ri] + H.SUN_R
        hit = (base_ang <= sun) & (sun <= top_ang)
        if not hit.any():
            break
        df = pd.DataFrame({"r": ri[hit], "a": top_ang[hit], "t": ti[hit], "d": along[hit]})
        df = df.sort_values("a").drop_duplicates("r", keep="last")
        sky[df.r.values] = df.a.values
        first = who[df.r.values] == ""     # name the tree that first hides the sun, not the last in a chain
        who[df.r.values[first]] = [f"tree:{d:.0f}:{trees.species.values[t]}"
                                   for d, t in zip(df.d.values[first], df.t.values[first])]
    return sky, who


def day_runs(aligned: np.ndarray, visible: np.ndarray) -> list:
    """[[first, last, visible], ...] runs of aligned days with the same visibility."""
    out, start = [], None
    for d in range(366):
        key = (bool(aligned[d]), bool(visible[d])) if d < 365 else (False, False)
        if start is not None and key != cur:
            out.append([start, d - 1, int(cur[1])])
            start = None
        if start is None and key[0]:
            start, cur = d, key
    return out


def compute(force: bool = False) -> gpd.GeoDataFrame:
    if CACHE.exists() and not force:
        return pd.read_pickle(CACHE)
    s = H.compute()
    boro = gpd.read_file(H.DATA / "boroughs.geojson").to_crs(H.CRS_M)
    grid = H.NYCGrid(boro)
    trees = load_trees(grid)
    tree_idx = shapely.STRtree(shapely.points(trees.x.values, trees.y.values))
    mid = shapely.line_interpolate_point(s.geometry.values, 0.5, normalized=True)
    mxy = shapely.get_coordinates(mid)
    eye = grid.ground_at(mxy[:, 0], mxy[:, 1]) + H.EYE
    can = (s.w1_date.notna() | s.e1_date.notna()).values
    idx = np.where(can)[0]
    every = np.ones(len(trees), bool)
    dsm = H.DSM()
    pmask, pres = park_mask(grid)
    days = np.arange(365)
    season_on = leaf_on(days)

    for end, evening in (("w", True), ("e", False)):
        az = s.west_az.values if end == "w" else (s.west_az.values - 180) % 360
        sky_b = s[f"sky_{end}"].values.astype(float)
        sky_b0 = np.where(np.isnan(sky_b), 0, sky_b)
        sky_off, sky_on = sky_b0.copy(), sky_b0.copy()
        who_off = np.full(len(s), "", dtype=object); who_on = who_off.copy()
        # leaf-on: park woodland first (solid from the ground up), then street trees on top
        pa, pd_ = park_skyline(mxy[idx], az[idx], eye[idx], grid, dsm, pmask, pres)
        start_on = sky_b0.copy()
        park_wins = pa > sky_b0[idx]
        start_on[idx] = np.maximum(sky_b0[idx], pa)
        who_on[idx[park_wins]] = [f"park:{d:.0f}:park trees" for d in pd_[park_wins]]
        for mask, start, sky, who in ((trees.ever.values, sky_b0, sky_off, who_off), (every, start_on, sky_on, who_on)):
            a, w = tree_skyline(mxy[idx], az[idx], eye[idx], start[idx], trees, tree_idx, mask)
            sky[idx] = a
            who[idx] = np.where(w != "", w, who[idx])
        s[f"skyoff_{end}"], s[f"skyon_{end}"] = sky_off, sky_on
        s[f"treeoff_{end}"], s[f"treeon_{end}"] = who_off, who_on
        clear_b = sky_b0 <= H.MAX_SKYLINE
        clear = {"off": sky_off <= H.MAX_SKYLINE, "on": sky_on <= H.MAX_SKYLINE}
        alt = {k: np.where(c, sk, np.where(clear_b, sky_b0, 0)) + H.SUN_R   # blocked: open-horizon dates
               for k, c, sk in (("off", clear["off"], sky_off), ("on", clear["on"], sky_on))}
        dates, wins = {}, {}
        for k in ("off", "on"):
            (d1, d2), (t1, t2) = H.henge_dates(az, alt[k], s.lat.values, s.lon.values, evening)
            dates[k] = [(d1, t1), (d2, t2)]
            wins[k] = H.aligned_windows(az, alt[k], s.lat.values, s.lon.values, evening, can)

        # each crossing judged in its own season
        for c in (0, 1):
            d_off, t_off = dates["off"][c]
            d_on, t_on = dates["on"][c]
            day_off = np.floor(np.nan_to_num(d_off, nan=-9) + 0.5) % 365
            use_on = leaf_on(day_off) & ~np.isnan(d_on)
            d = np.where(use_on, d_on, d_off)
            t = np.where(use_on, t_on, t_off)
            loc = [H.to_local(a, b) for a, b in zip(np.where(can, d, np.nan), t)]
            s[f"{end}{c + 1}_date"] = pd.to_datetime([x[0] for x in loc])
            s[f"{end}{c + 1}_time"] = [x[1] for x in loc]
            ok = np.where(use_on, clear["on"], clear["off"])
            s[f"{end}{c + 1}_ok"] = ok
            # why it's blocked: buildings/terrain first, then trees
            why = np.where(~clear_b, s[f"blk_{end}"].values,
                           np.where(ok, "", np.where(use_on, who_on, who_off)))
            s[f"{end}{c + 1}_why"] = why
            s[f"{end}{c + 1}_leaves"] = use_on

        # day-by-day: aligned in its season's geometry, and visible?
        runs = []
        for i in range(len(s)):
            if not can[i]:
                runs.append([])
                continue
            al_off = np.zeros(365, bool); al_on = np.zeros(365, bool)
            for a, b in wins["off"][i]:
                al_off[a:b + 1] = True
            for a, b in wins["on"][i]:
                al_on[a:b + 1] = True
            aligned = np.where(season_on, al_on, al_off)
            visible = aligned & np.where(season_on, clear["on"][i], clear["off"][i])
            runs.append(day_runs(aligned, visible))
        s[f"{end}_days"] = runs
        s[f"clearb_{end}"] = clear_b
    s.to_pickle(CACHE)
    return s


if __name__ == "__main__":
    import time
    t0 = time.time()
    s = compute(force=True)
    print(f"done in {time.time() - t0:.0f}s")
    h = s[s.w1_date.notna()]
    for end in "we":
        b = h[f"clearb_{end}"].mean()
        print(end, f"clear (buildings only) {b:.2f}  leaf-off {(h[f'skyoff_{end}'] <= H.MAX_SKYLINE).mean():.2f}"
                   f"  leaf-on {(h[f'skyon_{end}'] <= H.MAX_SKYLINE).mean():.2f}")
