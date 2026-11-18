"""Day 18 · Null — one-state test of the paper-cut look.

Every 2020 Census block with 0 residents (and some land) is cut out of a paper sheet.
The sheet casts a soft shadow onto the table, so the holes read as depth, not color.

Data: Census TIGER/Line 2020 TABBLOCK20 (POP20, ALAND20), data/blocks/tl_2020_SS_tabblock20.zip
Run:  ../.venv/bin/python explore_state.py [STATE_FIPS]
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from matplotlib.colors import to_rgb
from PIL import Image
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

FIPS = sys.argv[1] if len(sys.argv) > 1 else "08"
NAMES = {"08": "Colorado"}
CRS_M = "EPSG:26913"  # UTM 13N, metres

V = S.VINTAGE["color"]


def mix(a, b, t):
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return a + (b - a) * t


PAPER = np.array(to_rgb(V["land"]))
TABLE = mix(V["bg"], V["muted"], 0.38)       # the surface the sheet lies on, seen through holes
WALL = mix(V["land"], V["muted"], 0.45)      # the cut edge (paper thickness)
SHADOW = np.array(to_rgb(V["ink"]))


def load():
    f = DATA / "blocks" / f"tl_2020_{FIPS}_tabblock20.zip"
    gdf = gpd.read_file(f, columns=["GEOID20", "POP20", "HOUSING20", "ALAND20", "AWATER20"])
    return gdf.to_crs(CRS_M)


def stats(gdf):
    land = gdf.ALAND20.sum()
    empty = gdf[(gdf.POP20 == 0) & (gdf.ALAND20 > 0)]
    print(f"blocks: {len(gdf):,}  with land: {(gdf.ALAND20 > 0).sum():,}  empty (pop 0, land>0): {len(empty):,}"
          f" ({len(empty) / (gdf.ALAND20 > 0).sum():.1%})")
    print(f"empty share of land area: {empty.ALAND20.sum() / land:.1%}")
    print(f"empty but with housing units: {(empty.HOUSING20 > 0).sum():,}")
    return empty


def render(gdf, empty, px_w=4800, pad=0.06):
    """Return an RGB array of the paper sheet with empty blocks cut out, at supersampled size."""
    x0, y0, x1, y1 = gdf.total_bounds
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - w * pad, x1 + w * pad, y0 - h * pad * 1.6, y1 + h * pad
    res = (x1 - x0) / px_w
    px_h = int(round((y1 - y0) / res))
    tr = from_origin(x0, y1, res, res)

    state = rasterio.features.rasterize(((g, 1) for g in gdf.geometry), out_shape=(px_h, px_w),
                                        transform=tr, dtype="uint8")
    holes = rasterio.features.rasterize(((g, 1) for g in empty.geometry), out_shape=(px_h, px_w),
                                        transform=tr, dtype="uint8", all_touched=False)
    paper = (state & (1 - holes)).astype(np.float32)

    s = px_w / 4800  # scale effects with resolution
    # Paper thickness: the sheet seen slightly from the south shows a wall strip below its edge
    # (front edge of the sheet; the far wall inside each hole).
    thick = 3 * s
    wall = np.clip(shift(paper, (thick, 0), order=1) - paper, 0, 1)
    # Cast shadow: light from upper left, soft.
    sh = gaussian_filter(shift(paper, (10 * s, 7 * s), order=1), 9 * s)
    sh_tight = gaussian_filter(shift(paper, (3 * s, 2 * s), order=1), 2.5 * s)
    shadow = np.clip(0.38 * sh + 0.45 * sh_tight, 0, 0.75)

    img = TABLE[None, None, :] * (1 - shadow[..., None]) + SHADOW[None, None, :] * shadow[..., None] * 0.35
    img = img * (1 - wall[..., None]) + WALL[None, None, :] * wall[..., None]
    img = img * (1 - paper[..., None]) + PAPER[None, None, :] * paper[..., None]
    return np.clip(img, 0, 1), (x0, x1, y0, y1)


def main():
    S.apply(palette="vintage")
    gdf = load()
    empty = stats(gdf)
    share = empty.ALAND20.sum() / gdf.ALAND20.sum()

    img, ext = render(gdf, empty)
    out_w = 2400
    small = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (out_w, int(img.shape[0] * out_w / img.shape[1])), Image.LANCZOS)

    fig = plt.figure(figsize=(12, 12 * small.height / small.width + 1.4))
    fig.patch.set_facecolor(TABLE)
    ax = fig.add_axes([0, 0.04, 1, 0.86])
    ax.imshow(np.asarray(small), extent=ext, interpolation="lanczos")
    ax.set_axis_off()
    S.title(fig, f"{share:.0%} of {NAMES.get(FIPS, FIPS)}'s land is home to no one", y=0.965)
    fig.text(0.5, 0.905, "Each hole is a 2020 Census block with zero residents", ha="center",
             fontsize=S.SIZE["subtitle"] + 2, color=S.MUTED)
    S.credit(fig, "Blocks with 0 people and some land, cut out  ·  Data: US Census Bureau, 2020 Census "
                  "TIGER/Line blocks  ·  #30DayMapChallenge Day 18: Null", y=0.015)
    fig.savefig(OUT / f"test_{FIPS}.png", dpi=200, facecolor=TABLE)
    print("wrote", OUT / f"test_{FIPS}.png")


if __name__ == "__main__":
    main()
