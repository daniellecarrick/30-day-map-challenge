"""Day 5 · Senses: Sight — the henge calculation (used by build.py).

Every straight street lines up with the rising or setting sun on a few days a year. For each
street segment this works out:
  1. the street's bearing: segments of the same street that join end to end in a straight line
     form a "run"; the run's bearing (true north, geodesic) is the bearing of the whole straight
     stretch, which is far more precise than one short block;
  2. the skyline at each end: looking straight down the street from the middle of the block
     (eye height 1.7 m), the highest angle anything reaches above flat: NYC buildings from the
     city's footprints (roof heights, burned into a 5 m grid), everything beyond the city (New
     Jersey's Palisades and towers, Long Island) from the Copernicus 30 m surface model; earth
     curvature included;
  3. the henge dates: the days when the sun, sitting just on that skyline (lower edge touching
     it, refraction included), is at the street's bearing. A sunset henge for the west end, a
     sunrise henge for the east end;
  4. whether it is visible: if the skyline at that end is too high (the street ends in a wall
     of buildings or a cliff) the sun is never seen framed low at the end of the street; its
     dates are then the ones it would have with an open horizon.

Cached in data/raw/henge.pkl; delete it to recompute.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Geod, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).parent
DATA = HERE / "data"
RAW = DATA / "raw"
CACHE = RAW / "henge.pkl"

CRS_M = "EPSG:32618"     # UTM 18N, metres
YEAR = 2027              # the next full year of henges
TZ = ZoneInfo("America/New_York")
EYE = 1.7                # m above the street
RAY_KM = 30              # how far to look down the street
K_EARTH = 6_371_000 * 7 / 6   # effective earth radius with standard refraction
SUN_R = 0.27             # sun's radius, degrees
MAX_SKYLINE = 3.0        # deg: a skyline higher than this hides the low sun ("blocked"). Street View
                         # spot checks (data/spotcheck.csv) agree at 3; 2 wrongly hides distant skylines
RUN_ANGLE_TOL = 2.0      # deg: segments of one street join into a run when this straight
GROUND_MIN_M = 150       # m: ignore open ground closer than this (see skyline())
MIN_RUN_M = 150          # runs shorter than this are too short to frame the sun (alleys, stubs)
FT = 0.3048

geod = Geod(ellps="WGS84")
to_m = Transformer.from_crs("EPSG:4326", CRS_M, always_xy=True)


# ---------------------------------------------------------------- sun (NOAA / Meeus) ----------
def _sun(jd):
    """Declination and equation of time (minutes) for Julian day(s) jd."""
    T = (jd - 2451545.0) / 36525
    L0 = np.radians((280.46646 + T * (36000.76983 + T * 0.0003032)) % 360)
    M = np.radians(357.52911 + T * (35999.05029 - 0.0001537 * T))
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    C = np.radians(np.sin(M) * (1.914602 - T * (0.004817 + 0.000014 * T))
                   + np.sin(2 * M) * (0.019993 - 0.000101 * T) + np.sin(3 * M) * 0.000289)
    om = np.radians(125.04 - 1934.136 * T)
    lam = L0 + C - np.radians(0.00569 + 0.00478 * np.sin(om))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = np.radians(eps0 + 0.00256 * np.cos(om))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    eot = 4 * np.degrees(y * np.sin(2 * L0) - 2 * e * np.sin(M) + 4 * e * y * np.sin(M) * np.cos(2 * L0)
                         - 0.5 * y * y * np.sin(4 * L0) - 1.25 * e * e * np.sin(2 * M))
    return dec, eot


def refraction(alt_deg):
    """Bennett's formula: how much the air lifts the sun near the horizon, degrees."""
    return 1 / np.tan(np.radians(alt_deg + 7.31 / (alt_deg + 4.4))) / 60


def sun_at(days, alt_app, lat, lon, evening: bool):
    """Sun azimuth (deg from true north) and UTC time (datetime64) when the sun's centre is at
    apparent altitude `alt_app` on each day. days: int array of day-of-year offsets (0 = Jan 1)
    broadcast against per-street arrays alt_app, lat, lon."""
    jd0 = 2461406.5 + days     # 2027-01-01 00:00 UTC (JD 2461406.5)
    phi = np.radians(lat)
    h_geo = np.radians(alt_app - refraction(alt_app))
    t = 12 - lon / 15 + (4 if evening else -4)     # first guess, hours UTC
    for _ in range(3):
        dec, eot = _sun(jd0 + t / 24)
        cosH = (np.sin(h_geo) - np.sin(phi) * np.sin(dec)) / (np.cos(phi) * np.cos(dec))
        H = np.degrees(np.arccos(np.clip(cosH, -1, 1)))
        t = 12 - lon / 15 - eot / 60 + (H if evening else -H) / 15
    cosA = (np.sin(dec) - np.sin(h_geo) * np.sin(phi)) / (np.cos(h_geo) * np.cos(phi))
    A = np.degrees(np.arccos(np.clip(cosA, -1, 1)))
    return (360 - A if evening else A), t


def henge_dates(bearing, alt_app, lat, lon, evening: bool):
    """The (up to two) days of YEAR when the sun at `alt_app` sits at `bearing`.
    Returns two arrays of fractional day-of-year (NaN when the street never lines up) and
    the local clock time (hours) of each."""
    days = np.arange(-1, 367)[:, None]
    A, t = sun_at(days, alt_app[None], lat[None], lon[None], evening)
    f = A - bearing[None]
    out_d, out_t = [], []
    # the sun's azimuth swings one way Jan->Jun and back Jun->Dec: one crossing each way
    for rising in (True, False):
        s0, s1 = f[:-1], f[1:]
        hit = (s0 < 0) & (s1 >= 0) if rising else (s0 >= 0) & (s1 < 0)
        i = np.where(hit.any(0), hit.argmax(0), -1)
        cols = np.arange(f.shape[1])
        ii = np.clip(i, 0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = s0[ii, cols] / (s0[ii, cols] - s1[ii, cols])
        d = days[ii, 0] + frac
        tt = t[ii, cols] + frac * (t[ii + 1, cols] - t[ii, cols])
        d = np.where(i >= 0, d, np.nan)
        out_d.append(d)
        out_t.append(tt)
    # earlier date first
    swap = np.nan_to_num(out_d[0], nan=1e9) > np.nan_to_num(out_d[1], nan=1e9)
    d1, d2 = np.where(swap, out_d[1], out_d[0]), np.where(swap, out_d[0], out_d[1])
    t1, t2 = np.where(swap, out_t[1], out_t[0]), np.where(swap, out_t[0], out_t[1])
    return (d1, d2), (t1, t2)


def aligned_windows(bearing, alt_app, lat, lon, evening: bool, mask):
    """For each street, the runs of days in YEAR when the sun's centre is within one sun radius
    of the street's bearing as it sits on the skyline: the days the sun fits in the street's
    frame. Returns a list (per street) of [first_day, last_day] day-of-year pairs (0 = Jan 1)."""
    days = np.arange(365)[:, None]
    out = []
    for k in range(0, len(bearing), 4000):
        sl = slice(k, k + 4000)
        A, _ = sun_at(days, alt_app[None, sl], lat[None, sl], lon[None, sl], evening)
        on = np.abs(A - bearing[None, sl]) <= SUN_R
        for j in range(on.shape[1]):
            if not mask[k + j] or not on[:, j].any():
                out.append([])
                continue
            d = np.diff(np.r_[0, on[:, j].astype(int), 0])
            out.append([[int(a), int(b) - 1] for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])])
    return out


def to_local(dayfrac, t_utc_h):
    """(local date, local clock hours) for a fractional day of YEAR and a UTC hour."""
    if np.isnan(dayfrac):
        return None, None
    day = int(np.floor(dayfrac + 0.5))   # the event's day: nearest whole day crossing
    utc = dt.datetime(YEAR, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=day, hours=float(t_utc_h))
    loc = utc.astimezone(TZ)
    return loc.date(), loc.hour + loc.minute / 60


# ---------------------------------------------------------------- streets -> straight runs ----
def load_streets() -> gpd.GeoDataFrame:
    s = pd.concat([gpd.read_file(RAW / f"streets_{o}.geojson") for o in (0, 50000)], ignore_index=True)
    s = s.explode(index_parts=False).reset_index(drop=True)
    coords = [np.asarray(g.coords) for g in s.geometry]
    a = np.array([c[0] for c in coords]); b = np.array([c[-1] for c in coords])
    s["lon0"], s["lat0"], s["lon1"], s["lat1"] = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    s = s.to_crs(CRS_M)
    s = s[s.geometry.length > 0].reset_index(drop=True)
    s["name"] = s.stname_label.fillna(s.full_street_name).fillna("")
    return s


def axis_angle(x0, y0, x1, y1):
    """Undirected direction of a line, degrees in [0, 180)."""
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180


def build_runs(s: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join each street's segments into straight runs; give every segment its run's bearing."""
    xy0 = shapely.get_coordinates(shapely.get_point(s.geometry, 0))
    xy1 = shapely.get_coordinates(shapely.get_point(s.geometry, -1))
    chord = np.hypot(*(xy1 - xy0).T)
    s["straight"] = chord / s.geometry.length > 0.995
    s["ang"] = axis_angle(*xy0.T, *xy1.T)
    # endpoints rounded to 0.5 m identify nodes
    ends = pd.DataFrame({
        "seg": np.r_[np.arange(len(s)), np.arange(len(s))],
        "key": np.r_[[f"{x:.0f},{y:.0f}" for x, y in np.round(xy0 * 2)], [f"{x:.0f},{y:.0f}" for x, y in np.round(xy1 * 2)]],
    })
    ends["name"] = s.name.values[ends.seg]
    pairs = ends.merge(ends, on=["key", "name"])
    pairs = pairs[pairs.seg_x < pairs.seg_y]
    i, j = pairs.seg_x.values, pairs.seg_y.values
    d = np.abs(s.ang.values[i] - s.ang.values[j]) % 180
    d = np.minimum(d, 180 - d)
    ok = (d < RUN_ANGLE_TOL) & s.straight.values[i] & s.straight.values[j]
    n = len(s)
    g = coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    _, s["run"] = connected_components(g, directed=False)

    # each run's axis: principal direction of its vertices; ends = extreme projections
    rows = []
    for run, grp in s.groupby("run"):
        pts = shapely.get_coordinates(grp.geometry.values)
        c = pts.mean(0)
        u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        v = vt[0]
        proj = (pts - c) @ v
        dev = np.abs((pts - c) @ np.array([-v[1], v[0]])).max()
        rows.append((run, *(c + proj.min() * v), *(c + proj.max() * v), proj.max() - proj.min(), dev))
    r = pd.DataFrame(rows, columns=["run", "ax", "ay", "bx", "by", "run_len", "run_dev"]).set_index("run")
    back = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True)
    alon, alat = back.transform(r.ax.values, r.ay.values)
    blon, blat = back.transform(r.bx.values, r.by.values)
    az, _, _ = geod.inv(alon, alat, blon, blat)
    az = az % 360
    r["west_az"] = np.where(az >= 180, az, (az + 180) % 360)   # bearing toward the run's west end
    s = s.join(r[["west_az", "run_len", "run_dev"]], on="run")
    return s


# ---------------------------------------------------------------- skyline down the street ----
def load_buildings() -> gpd.GeoDataFrame:
    parts = []
    for p in sorted(RAW.glob("bld_*.geojson")):
        g = gpd.read_file(p)
        if len(g):
            parts.append(g)
    b = pd.concat(parts, ignore_index=True)
    b["ground_m"] = pd.to_numeric(b.ground_elevation, errors="coerce") * FT
    b["roof_m"] = b.ground_m + pd.to_numeric(b.height_roof, errors="coerce").fillna(0) * FT
    b = b[b.ground_m.notna() & b.geometry.notna()]
    return b.to_crs(CRS_M)[["roof_m", "ground_m", "geometry"]].reset_index(drop=True)


class NYCGrid:
    """Every NYC building footprint burned into a 5 m grid of roof elevations (decimetres above
    sea level), plus a 50 m ground-elevation surface (from building ground elevations, gaps filled
    from the nearest building) for open ground, and a mask of where the grid applies (city land)."""
    RES = 5
    COARSE = 10            # ground grid cell = 10 x RES
    FILE = RAW / "nyc_grid.npz"

    def __init__(self, boro: gpd.GeoDataFrame):
        if self.FILE.exists():
            z = np.load(self.FILE)
            self.roof, self.ground, self.land, (self.x0, self.y1) = z["roof"], z["ground"], z["land"], z["origin"]
            return
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        from scipy.ndimage import distance_transform_edt
        bld = load_buildings()
        x0, y0, x1, y1 = boro.total_bounds + np.array([-200, -200, 200, 200])
        w, h = int((x1 - x0) / self.RES) + 1, int((y1 - y0) / self.RES) + 1
        tf = from_origin(x0, y1, self.RES, self.RES)
        bld = bld.sort_values("roof_m")          # later shapes overwrite: each cell keeps the tallest
        roof = rasterize(zip(bld.geometry.values, np.round(bld.roof_m.values * 10)), out_shape=(h, w),
                         transform=tf, fill=-32768, dtype="int16")
        tfc = from_origin(x0, y1, self.RES * self.COARSE, self.RES * self.COARSE)
        hc, wc = h // self.COARSE + 1, w // self.COARSE + 1
        ground = rasterize(zip(bld.geometry.values, bld.ground_m.values), out_shape=(hc, wc), transform=tfc,
                           fill=np.nan, dtype="float32", all_touched=True)
        _, (ri, ci) = distance_transform_edt(np.isnan(ground), return_indices=True)
        ground = ground[ri, ci]
        land = rasterize(boro.geometry.buffer(40).values, out_shape=(h, w), transform=tf, fill=0,
                         default_value=1, dtype="uint8")
        self.roof, self.ground, self.land, self.x0, self.y1 = roof, ground, land, x0, y1
        np.savez(self.FILE, roof=roof, ground=ground, land=land, origin=np.array([x0, y1]))

    def _rc(self, x, y):
        return ((self.y1 - y) / self.RES).astype(np.int64), ((x - self.x0) / self.RES).astype(np.int64)

    def ground_at(self, x, y):
        r, c = self._rc(x, y)
        return self.ground[np.clip(r // self.COARSE, 0, self.ground.shape[0] - 1),
                           np.clip(c // self.COARSE, 0, self.ground.shape[1] - 1)]

    def sample(self, x, y):
        """(elevation m, is_building, valid): valid = the point is on NYC land inside the grid."""
        r, c = self._rc(x, y)
        inb = (r >= 0) & (r < self.roof.shape[0]) & (c >= 0) & (c < self.roof.shape[1])
        rr, cc = np.where(inb, r, 0), np.where(inb, c, 0)
        valid = inb & (self.land[rr, cc] == 1)
        roof = self.roof[rr, cc]
        is_b = valid & (roof != -32768)
        z = np.where(is_b, roof / 10.0, self.ground_at(x, y))
        return z, is_b, valid


class DSM:
    """Copernicus GLO-30 surface model, sampled at CRS_M points (nearest pixel)."""

    def __init__(self):
        self.tiles = [rasterio.open(p) for p in sorted(RAW.glob("dsm_*.tif"))]
        self.arr = [t.read(1) for t in self.tiles]
        self.back = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True)

    def sample(self, x, y):
        lon, lat = self.back.transform(x, y)
        out = np.full(x.shape, np.nan)
        for t, a in zip(self.tiles, self.arr):
            r, c = rasterio.transform.rowcol(t.transform, lon, lat)
            r, c = np.asarray(r), np.asarray(c)
            ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1]) & np.isnan(out)
            out[ok] = a[r[ok], c[ok]]
        return out


# fine steps near the eye (where a single building matters most), coarser far away
STEPS = np.r_[np.arange(6, 2000, 2.5), np.arange(2000, 8000, 5), np.arange(8000, RAY_KM * 1000, 15)].astype(float)


def skyline(origins: np.ndarray, az_true: np.ndarray, eye_m: np.ndarray, grid: NYCGrid, dsm: DSM):
    """Highest elevation angle (deg) seen looking from each origin (CRS_M xy) along az_true
    (true bearing), and what makes it: "bldg:<metres away>", "ground:<m>" or "dsm:<m>"."""
    n = len(origins)
    back = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True)
    lon, lat = back.transform(*origins.T)
    elon, elat, _ = geod.fwd(lon, lat, az_true, np.full(n, RAY_KM * 1000.0))
    ex, ey = to_m.transform(elon, elat)
    L = RAY_KM * 1000
    ux, uy = (ex - origins[:, 0]) / L, (ey - origins[:, 1]) / L
    drop = STEPS ** 2 / (2 * K_EARTH)
    best = np.full(n, -90.0)
    who = np.full(n, "", dtype=object)
    for k in range(0, n, 500):
        sl = slice(k, k + 500)
        px = origins[sl, 0, None] + ux[sl, None] * STEPS
        py = origins[sl, 1, None] + uy[sl, None] * STEPS
        z, is_b, valid = grid.sample(px, py)
        # the 50 m ground surface is too coarse to trust right next to the eye: open ground only
        # counts from GROUND_MIN_M out (buildings count at any distance)
        z = np.where(valid & ~is_b & (STEPS[None] < GROUND_MIN_M), -1e9, z)
        out = ~valid
        if out.any():
            z[out] = dsm.sample(px[out], py[out])
        z = np.where(np.isnan(z), -1e9, z)
        ang = np.degrees(np.arctan2(z - drop - eye_m[sl, None], STEPS))
        j = ang.argmax(1)
        rows = np.arange(ang.shape[0])
        best[sl] = ang[rows, j]
        src = np.where(is_b[rows, j], "bldg", np.where(valid[rows, j], "ground", "dsm"))
        who[sl] = [f"{a}:{d:.0f}" for a, d in zip(src, STEPS[j])]
    return best, who


# ---------------------------------------------------------------- main ----------------------
def compute(force: bool = False) -> gpd.GeoDataFrame:
    if CACHE.exists() and not force:
        return pd.read_pickle(CACHE)
    s = build_runs(load_streets())
    boro = gpd.read_file(DATA / "boroughs.geojson").to_crs(CRS_M)
    grid = NYCGrid(boro)
    dsm = DSM()

    mid = shapely.line_interpolate_point(s.geometry.values, 0.5, normalized=True)
    mxy = shapely.get_coordinates(mid)
    lon, lat = Transformer.from_crs(CRS_M, "EPSG:4326", always_xy=True).transform(*mxy.T)
    s["lon"], s["lat"] = lon, lat
    eye = grid.ground_at(mxy[:, 0], mxy[:, 1]) + EYE

    # only streets whose bearing the sun can ever reach (with a margin for high skylines)
    can = (s.west_az.between(232, 308)) & (s.run_len >= MIN_RUN_M) & (s.run_dev < 12)
    s["sky_w"] = np.nan; s["sky_e"] = np.nan; s["blk_w"] = ""; s["blk_e"] = ""
    idx = np.where(can)[0]
    for col, blk, az in (("sky_w", "blk_w", s.west_az.values[idx]),
                         ("sky_e", "blk_e", (s.west_az.values[idx] - 180) % 360)):
        a, w = skyline(mxy[idx], az, eye[idx], grid, dsm)
        s.loc[s.index[idx], col] = np.maximum(a, 0)
        s.loc[s.index[idx], blk] = w

    # henge dates: sun's lower edge resting on the street's skyline
    for end, evening, az in (("w", True, s.west_az.values), ("e", False, (s.west_az.values - 180) % 360)):
        sky = s[f"sky_{end}"].values
        # blocked ends: the dates it WOULD line up with an open horizon
        alt = np.where(np.isnan(sky) | (sky > MAX_SKYLINE), 0, sky) + SUN_R
        (d1, d2), (t1, t2) = henge_dates(az, alt, s.lat.values, s.lon.values, evening)
        for k, (d, t) in enumerate(((d1, t1), (d2, t2)), 1):
            loc = [to_local(a, b) for a, b in zip(np.where(can, d, np.nan), t)]
            s[f"{end}{k}_date"] = [x[0] for x in loc]
            s[f"{end}{k}_time"] = [x[1] for x in loc]
        s[f"{end}_win"] = aligned_windows(az, alt, s.lat.values, s.lon.values, evening, can.values)
    s["clear_w"] = s.sky_w <= MAX_SKYLINE
    s["clear_e"] = s.sky_e <= MAX_SKYLINE
    keep = ["physicalid", "name", "boroughcode", "run", "west_az", "run_len", "run_dev", "lon", "lat", "sky_w", "sky_e",
            "blk_w", "blk_e", "clear_w", "clear_e", "w1_date", "w1_time", "w2_date", "w2_time",
            "e1_date", "e1_time", "e2_date", "e2_time", "w_win", "e_win", "geometry"]
    s = s[keep]
    for c in [c for c in keep if c.endswith("_date")]:
        s[c] = pd.to_datetime(s[c])
    s.to_pickle(CACHE)
    return s


if __name__ == "__main__":
    import time
    t = time.time()
    s = compute(force=True)
    print(f"{len(s)} segments in {time.time() - t:.0f}s")
