"""Day 15 PNG: the same view of the globe as it is and turned inside out, side by side.

matplotlib can't draw displaced 3D relief, so the globes are rendered by the page itself
(out/map.html?shot=earth|flip) in headless Chrome, then laid out here with the series title,
callouts, legend and credit. Run after build.py (build.py calls this at the end).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

VIEW = dict(lat=12, lon=138, dist=4.1)   # western Pacific, with the Himalayas on the left
EXAG = 120
SHOT = 1400                              # px, square
FOV = 35                                 # must match the page's camera
R_EARTH = 6371000


def capture(mode: str, dest: Path) -> np.ndarray:
    url = (OUT / "map.html").resolve().as_uri() + \
        f"?shot={mode}&lat={VIEW['lat']}&lon={VIEW['lon']}&dist={VIEW['dist']}&exag={EXAG}"
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", f"--window-size={SHOT},{SHOT}",
                        "--default-background-color=00000000", "--virtual-time-budget=20000",
                        f"--user-data-dir={prof}", f"--screenshot={dest}", url],
                       check=True, capture_output=True, timeout=180)
    return np.asarray(Image.open(dest).convert("RGBA"))


def dir_of(lat, lon):
    """Same as dirOf() in the page (three.js SphereGeometry layout)."""
    phi = np.radians(lon + 180)
    th = np.radians(90 - lat)
    return np.array([-np.cos(phi) * np.sin(th), np.cos(th), np.sin(phi) * np.sin(th)])


def project(lat, lon, radius):
    """Pixel position of a point on the globe in a SHOT x SHOT capture."""
    eye = dir_of(VIEW["lat"], VIEW["lon"]) * VIEW["dist"]
    z = eye / np.linalg.norm(eye)
    x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    p = dir_of(lat, lon) * radius - eye
    cx, cy, cz = p @ x, p @ y, p @ z
    f = 1 / np.tan(np.radians(FOV / 2))
    return ((cx / -cz * f + 1) / 2 * SHOT, (1 - cy / -cz * f) / 2 * SHOT)


def grid_height(lat, lon, png, offset, step):
    h, w = png.shape[:2]
    x = min(w - 1, int((lon + 180) / 360 * w))
    y = min(h - 1, int((90 - lat) / 180 * h))
    return (int(png[y, x, 0]) * 256 + int(png[y, x, 1])) * step - offset


def radius(g, flipped, floor):
    k = EXAG / R_EARTH
    return 1 + k * (-g + floor if flipped else g)


def ramp_cmap(colors):
    """The page's color ramp from 6,500 m deep to 7,000 m high, as a matplotlib colormap."""
    deep, shallow, neutral, mid, high = colors
    lo, hi = -6500, 7000
    pos = lambda v: (v - lo) / (hi - lo)
    eps = 1e-4
    return LinearSegmentedColormap.from_list("ramp", [
        (0, deep), (pos(0) - eps, shallow), (pos(0), neutral), (pos(2500), mid), (1, high)])


def main(meta: dict):
    S.apply()
    with tempfile.TemporaryDirectory() as tmp:
        earth = capture("earth", Path(tmp) / "earth.png")
        flip = capture("flip", Path(tmp) / "flip.png")
    hpng = np.asarray(Image.open(OUT / "height.webp").convert("RGB"))
    feats = {f["name"]: f for f in meta["features"]}

    fig = plt.figure(figsize=S.MAP["png_size_in"])
    axes = [fig.add_axes([0.02, 0.10, 0.47, 0.76]), fig.add_axes([0.51, 0.10, 0.47, 0.76])]
    for ax, img in zip(axes, [earth, flip]):
        ax.imshow(img, interpolation="lanczos")
        ax.set_axis_off()
    S.panel_heading(axes[0], "Earth", f"Real heights, exaggerated {EXAG}×")
    S.panel_heading(axes[1], "Inside out", "Every height flipped around sea level")

    fmt = lambda v: f"{abs(v):,} m"
    def mark(ax, name, flipped, text_xy, heading, detail, ha="left"):
        f = feats[name]
        g = grid_height(f["lat"], f["lon"], hpng, meta["offset"], meta["step"])
        xy = project(f["lat"], f["lon"], radius(g, flipped, meta["floor"]))
        S.callout(ax, xy, text_xy, heading, detail, ha=ha, halo=True)

    ev, md = feats["Mount Everest"]["h"], feats["Challenger Deep, Mariana Trench"]["h"]
    mark(axes[0], "Mount Everest", False, (0.0, 0.76), "Mount Everest", f"{fmt(ev)} high")
    mark(axes[0], "Challenger Deep, Mariana Trench", False, (0.99, 0.10), "Challenger Deep",
         f"{fmt(md)} deep, the bottom of the ocean", ha="right")
    mark(axes[1], "Challenger Deep, Mariana Trench", True, (0.99, 0.10), "Challenger Deep",
         f"now {fmt(md)} high, {-md - ev:,} m taller than Everest", ha="right")
    mark(axes[1], "Mount Everest", True, (0.0, 0.76), "Mount Everest", f"now a pit {fmt(ev)} deep")

    # legend: the page's color ramp, centered on sea level
    lax = fig.add_axes([0.40, 0.085, 0.20, 0.016])
    lax.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=ramp_cmap(meta["diverging"]), extent=[-6500, 7000, 0, 1])
    lax.set_yticks([])
    lax.set_xticks([-6500, 0, 7000], ["6,500 m deep", "Sea level", "7,000 m high"], fontsize=S.SIZE["legend"], color=S.INK)
    lax.tick_params(length=3, color=S.MUTED)
    for s in lax.spines.values():
        s.set_visible(False)
    fig.text(0.5, 0.108, "Height above or below sea level", ha="center", fontsize=S.SIZE["legend"],
             fontweight="semibold", color=S.INK)

    S.title(fig, meta["title"])
    fig.text(0.5, 0.915, "Six ocean trenches are deeper than Everest is tall. Flip every height around sea level "
             "and they become the six highest peaks on Earth, all in the western Pacific.",
             ha="center", fontsize=S.SIZE["subtitle"] + 1, color=S.MUTED)
    S.credit(fig, f"Heights mirrored around sea level; relief exaggerated {EXAG}× (inside out, drawn up from the median "
                  f"ocean floor, {-meta['floor']:,} m). Data: ETOPO1, NOAA NCEI; named depths and heights from published "
                  f"surveys  ·  #30DayMapChallenge {meta['day']}")
    fig.savefig(OUT / "map.png", dpi=S.MAP["png_dpi"])
    plt.close(fig)
    print("wrote out/map.png")


if __name__ == "__main__":
    print("run build.py; it calls make_png.main() with the page data")
