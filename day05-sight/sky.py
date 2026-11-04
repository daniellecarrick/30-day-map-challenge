"""Day 5 · how much sky can you see from the middle of every NYC block? (used by build.py)

From each block's midpoint (eye 1.7 m above the street), find the horizon in 360 directions out to
3 km, against every building's roof (henge.py's 5 m grid of NYC footprints). Trees and terrain
without buildings on it don't count.

Sky share = 1 - mean(sin(horizon)): the share of the sky dome's AREA not hidden by buildings. (This
is not the cosine-weighted "sky view factor" of climate work.) One degree between rays matters: in
a deep canyon nearly all the sky is in a slit a few degrees wide along the street, so coarse rays
either miss it or overweight it.

Sky circle: 36 values r per 10 degrees of compass, with r^2 = mean(1 - sin(horizon)). Drawn from a
center (straight up) with the rim as the horizon, this is an equal-area projection: the sky shape's
share of the disc IS the block's sky share.

Cached in data/raw/sky.pkl; delete it to recompute.
"""
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

import henge as H

CACHE = H.RAW / "sky.pkl"
NODATA = -32768
AZ = np.radians(np.arange(360) + 0.5)
DX, DY = np.sin(AZ).astype(np.float32), np.cos(AZ).astype(np.float32)
STEPS = np.r_[np.arange(2.5, 200, 2.5), np.arange(200, 800, 5), np.arange(800, 3000, 20)].astype(np.float32)
BIN = 10   # degrees per sky-circle value


def march(x, y, eye, grid):
    """(1 - sin(horizon)) per block and degree, averaged into sky-circle bins, plus the sky share."""
    n = len(x)
    bins = np.zeros((n, 360 // BIN), np.float32)
    h, w = grid.roof.shape
    x0, y1, res = float(grid.x0), float(grid.y1), grid.RES
    x, y, eye = x.astype(np.float32), y.astype(np.float32), eye.astype(np.float32)
    t0 = time.time()
    for k in range(0, n, 100):
        sl = slice(k, k + 100)
        px = x[sl, None, None] + DX[None, :, None] * STEPS[None, None, :]
        py = y[sl, None, None] + DY[None, :, None] * STEPS[None, None, :]
        r = ((y1 - py) / res).astype(np.int32)
        c = ((px - x0) / res).astype(np.int32)
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        r[~ok] = 0; c[~ok] = 0
        tan = (np.where(ok, grid.roof[r, c], NODATA).astype(np.float32) / 10 - eye[sl, None, None]) / STEPS
        top = np.clip(tan.max(2), 0, None)
        open_ = 1 - top / np.sqrt(1 + top ** 2)
        bins[sl] = open_.reshape(open_.shape[0], -1, BIN).mean(2)
        if k % 20000 == 0:
            print(f"  {k:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    return bins


def compute() -> gpd.GeoDataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    s = H.load_streets()
    grid = H.NYCGrid(gpd.read_file(H.DATA / "boroughs.geojson").to_crs(H.CRS_M))
    mid = shapely.get_coordinates(shapely.line_interpolate_point(s.geometry.values, 0.5, normalized=True))
    eye = grid.ground_at(mid[:, 0], mid[:, 1]) + H.EYE
    # midpoint inside a footprint (a street running under a building): no meaningful sky
    r = ((grid.y1 - mid[:, 1]) / grid.RES).astype(int); c = ((mid[:, 0] - grid.x0) / grid.RES).astype(int)
    r = np.clip(r, 0, grid.roof.shape[0] - 1); c = np.clip(c, 0, grid.roof.shape[1] - 1)
    s["inside"] = grid.roof[r, c] / 10.0 > eye + 2
    bins = march(mid[:, 0], mid[:, 1], eye, grid)
    s["sky"] = bins.mean(1)
    s["bins"] = list(np.sqrt(bins))        # r per 10 degrees, 0..1
    s["mx"], s["my"] = mid[:, 0], mid[:, 1]
    s = s[["name", "boroughcode", "sky", "bins", "inside", "mx", "my", "geometry"]]
    s.to_pickle(CACHE)
    return s


if __name__ == "__main__":
    s = compute()
    s = s[~s.inside]
    print(s.groupby("boroughcode").sky.median().round(3).to_dict())
    for n in ["EXCHANGE PL", "NEW ST", "PINE ST", "WALL ST", "W 42 ST", "W 57 ST", "PARK AVE"]:
        g = s[(s.name == n) & (s.boroughcode == "1")]
        print(n, len(g), round(float(g.sky.median()), 3), round(float(g.sky.min()), 3), round(float(g.sky.max()), 3))
