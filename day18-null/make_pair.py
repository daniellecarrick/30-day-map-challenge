"""Day 18 · Null — the emptiest and fullest states side by side, in the paper-cut look.

Each state fills its own panel (not to the same scale). Uses the stage-1 caches from build.py.
Writes out/pair.png
Run:  ../.venv/bin/python make_pair.py
"""
import pickle

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from PIL import Image

import build as B

S = B.S


def render(f, long_side=3000):
    r = pickle.loads((B.PROC / f"{f}.pkl").read_bytes())
    crs = B.state_crs(f)
    outline = B.STATES.loc[[f]].to_crs(crs).geometry.iloc[0]
    holes = gpd.GeoSeries(r["holes"], crs="EPSG:4326").to_crs(crs)
    tr, shape, ext = B.grid(outline.bounds, long_side, pad=0.05)
    paper = B.supersample(lambda t, s: B.burn([outline], t, s) & (1 - B.burn(holes, t, s)), tr, shape)
    counties = B.COUNTIES[B.COUNTIES.STATEFP == f].to_crs(crs).boundary
    lines = rasterio.features.rasterize(((g, 1) for g in [*counties, outline.boundary]), out_shape=shape,
                                        transform=tr, all_touched=True, dtype="uint8").astype(np.float32)
    img = B.paper_cut(paper, long_side / 4800, lines)
    small = Image.fromarray((img * 255).astype(np.uint8))
    small = small.resize((small.width // 2, small.height // 2), Image.LANCZOS)
    return np.asarray(small), r


def main():
    S.apply(palette="vintage")
    shares = {f: (lambda r: r["empty_land"] / r["land"])(pickle.loads((B.PROC / f"{f}.pkl").read_bytes()))
              for f in B.STATES.index}
    most, least = max(shares, key=shares.get), min(shares, key=shares.get)

    fig = plt.figure(figsize=(16, 9.6))
    fig.patch.set_facecolor(B.TABLE)
    for i, f in enumerate([most, least]):
        img, r = render(f)
        ax = fig.add_axes([0.03 + i * 0.49, 0.07, 0.45, 0.68])
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
        name = B.STATES.loc[f, "NAME"]
        fig.text(0.03 + i * 0.49 + 0.225, 0.80, f"{shares[f]:.0%} of {name}'s land is in empty census blocks",
                 ha="center", fontsize=S.SIZE["panel_title"] + 5, fontweight="bold", color=S.INK,
                 fontfamily=S.HEAD_FONT)
        fig.text(0.03 + i * 0.49 + 0.225, 0.77,
                 "The emptiest state" if f == most else "The fullest state",
                 ha="center", va="top", fontsize=S.SIZE["subtitle"] + 2, color=S.MUTED)
    S.title(fig, "In Nevada, most of the map is missing. In Connecticut, almost none of it is"
            if (most, least) == ("32", "09") else "The emptiest and fullest states", y=0.94)
    S.credit(fig, "Each hole is a 2020 Census block with 0 residents; states not to the same scale  ·  "
                  f"{B.SOURCE}  ·  #30DayMapChallenge {B.DAY}", y=0.02)
    fig.savefig(B.OUT / "pair.png", dpi=200, facecolor=B.TABLE)
    print("wrote", B.OUT / "pair.png", most, least)


if __name__ == "__main__":
    main()
