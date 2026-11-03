"""Day NN · Theme — one-line description of the map.

Copy this folder to dayNN-theme/, then fill in load() and the two output sections.
Writes:
  - out/map.png    (static, for posting)
  - out/map.html   (interactive, self-contained)

Data: <source name and URL>
Run:  ../.venv/bin/python build.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402  (shared series look: fonts, colors, page builder)

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day NN: Theme"
TITLE = "Headline that states the finding"
SOURCE = "Data: <source> (Nov 2026)"
CRS_M = "EPSG:32618"  # metres; pick a local projected CRS for the place you map

# Color: pick ONE job — diverging (above/below a meaningful center), sequential
# (plain magnitude) or categorical (identity). See shared/tokens.json.
CENTER = 10
STOPS = S.diverging_stops(CENTER)
color_for = S.make_scale(STOPS)


def load() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA / "input.geojson").to_crs(CRS_M)
    gdf["color"] = gdf["value"].apply(color_for)
    return gdf


def main():
    S.apply()
    gdf = load()

    # ---- static PNG ----
    fig, ax = plt.subplots(figsize=S.MAP["png_size_in"])
    gdf.plot(ax=ax, color=gdf.color, edgecolor=S.EDGE, linewidth=S.MAP["cell_edge_width"])
    ax.set_axis_off()
    ax.set_aspect("equal")
    S.title(fig, TITLE)
    S.credit(fig, f"{SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])

    # ---- interactive page (geometry stays in CRS_M; the page uses d3.geoIdentity) ----
    g = gdf[["value", "color", "geometry"]].copy()
    g["geometry"] = g.geometry.simplify(3)
    data = {"features": json.loads(g.to_json(drop_id=True)), "stops": STOPS, "center": CENTER,
            "title": TITLE, "day": DAY, "source": SOURCE}
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.png and out/map.html", f"({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
