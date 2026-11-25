"""Day 25 · Is this a map? — cities as paint strips.

Every pixel of a cloud-free satellite view (~12 km square around each downtown), sorted by
lightness and cut into seven equal bands. Each chip is the average colour of one band, so
each chip is one seventh of the city.

Imagery: Sentinel-2 cloudless 2020 by EOX IT Services GmbH (contains modified Copernicus
Sentinel data 2020), CC BY-NC-SA 4.0. https://s2maps.eu
Run:  ../.venv/bin/python build.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
TILES = HERE / "data" / "tiles"
TILES.mkdir(parents=True, exist_ok=True)
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 25: Is this a map?"
SOURCE = "Imagery: Sentinel-2 cloudless 2020 by EOX (contains modified Copernicus Sentinel data), CC BY-NC-SA 4.0"
Z = 13
URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
CHIPS = 7
TITLE = "Paint strips from space"

CITIES = [
    ("Dubai", 25.2048, 55.2708), ("Cairo", 30.0444, 31.2357), ("Phoenix", 33.4484, -112.0740),
    ("Las Vegas", 36.1699, -115.1398), ("Mexico City", 19.4326, -99.1332), ("Paris", 48.8566, 2.3522),
    ("Tokyo", 35.6812, 139.7671), ("Singapore", 1.3000, 103.8500), ("Marrakech", 31.6295, -7.9811),
    ("New York", 40.7580, -73.9855), ("Atlanta", 33.7490, -84.3880), ("Seattle", 47.6062, -122.3321),
]


def to_linear(c):
    c = c / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(int)


def lightness(lin):
    """CIE L* (0–100) from linear RGB (N×3)."""
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(Y > 216 / 24389, 116 * np.cbrt(Y) - 16, Y * 24389 / 27)


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return int(x), int(y)


def mosaic(lat, lon, k=1):
    """(2k+1)² tiles around the point, as one RGB array."""
    cx, cy = tile_xy(lat, lon, Z)
    rows = []
    for y in range(cy - k, cy + k + 1):
        row = []
        for x in range(cx - k, cx + k + 1):
            f = TILES / f"{Z}_{x}_{y}.jpg"
            if not f.exists():
                subprocess.run(["curl", "-sS", "--fail", "-m", "60", "-A", "30DayMapChallenge/1.0", "-o", str(f),
                                URL.format(z=Z, x=x, y=y)], check=True)
            row.append(np.asarray(Image.open(f).convert("RGB")))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def strip(img):
    """Seven chips, light → dark: the mean colour (mixed in linear light) of each seventh of the pixels."""
    lin = to_linear(img.reshape(-1, 3).astype(float))
    L = lightness(lin)
    order = np.argsort(-L)
    chips = []
    for band in np.array_split(order, CHIPS):
        rgb = to_srgb(lin[band].mean(axis=0))
        chips.append({"hex": "#%02x%02x%02x" % tuple(rgb), "L": float(np.median(L[band]))})
    return chips, float(np.median(L))


def text_on(L):
    return S.INK if L > 55 else "#ffffff"


def inch_canvas(W, H):
    """A figure laid out in inches: one axes covering it with equal aspect, so cards and images stay square."""
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def draw_card(ax, r, img, x0, top, cw, pad, name_h, chip_h, chip_gap, name_fs, hex_fs=None):
    """One paint card: satellite image, city name, seven chips. Returns the card's height."""
    img_w = cw - 2 * pad
    card_h = pad + img_w + name_h + CHIPS * (chip_h + chip_gap) - chip_gap + pad
    ax.add_patch(Rectangle((x0, top - card_h), cw, card_h, fc="#ffffff", ec="#d9d7d0", lw=0.8))
    y = top - pad
    ax.imshow(img[::3, ::3], extent=(x0 + pad, x0 + pad + img_w, y - img_w, y), interpolation="lanczos", zorder=2)
    y -= img_w
    ax.text(x0 + pad, y - name_h / 2, r["name"], fontfamily=S.HEAD_FONT, fontsize=name_fs,
            color=S.INK, va="center", ha="left", fontweight="bold")
    y -= name_h
    for c in r["chips"]:
        ax.add_patch(Rectangle((x0 + pad, y - chip_h), img_w, chip_h, fc=c["hex"], ec="none"))
        if hex_fs:
            ax.text(x0 + pad + 0.06, y - chip_h + 0.06, c["hex"].upper(), fontsize=hex_fs, color=text_on(c["L"]),
                    va="bottom", ha="left")
        y -= chip_h + chip_gap
    return card_h


def draw_png(res, imgs):
    S.apply()
    W, H = 16, 7.6
    fig, ax = inch_canvas(W, H)
    left, slot = 0.3, (W - 0.6) / len(res)
    cw = slot - 0.18
    for i, r in enumerate(res):
        draw_card(ax, r, imgs[r["name"]], left + i * slot + (slot - cw) / 2, H - 1.25, cw,
                  pad=0.07, name_h=0.36, chip_h=0.58, chip_gap=0.04, name_fs=12.5, hex_fs=7.5)
    S.title(fig, TITLE)
    fig.text(0.5, (H - 0.8) / H, "Every pixel of a cloud-free satellite view of each city (~12 km square), sorted "
             "from light to dark and cut into seven equal parts. Each chip is the average color of one seventh of "
             "the city.", ha="center", fontsize=10.5, color=S.MUTED)
    S.credit(fig, f"Pixels sorted from light to dark, averaged in linear RGB  ·  {SOURCE}  ·  #30DayMapChallenge {DAY}")
    fig.savefig(OUT / "paint_strips.png", dpi=150)
    plt.close(fig)


def draw_linkedin(res, imgs):
    """4:5 portrait (1080x1350) for LinkedIn: the twelve cards in two rows of six, type sized for a phone."""
    S.apply()
    W, H = 5.4, 6.75
    fig, ax = inch_canvas(W, H)
    cols, margin = 6, 0.16
    slot = (W - 2 * margin) / cols
    cw = slot - 0.07
    top = H - 1.1
    for i, r in enumerate(res):
        row, col = divmod(i, cols)
        draw_card(ax, r, imgs[r["name"]], margin + col * slot + (slot - cw) / 2, top - row * 2.66, cw,
                  pad=0.035, name_h=0.22, chip_h=0.204, chip_gap=0.018, name_fs=8.2)
    fig.text(0.5, 0.972, TITLE, ha="center", va="top", fontsize=19, fontweight="bold",
             fontfamily=S.HEAD_FONT, color=S.INK)
    fig.text(0.5, 0.895, "Every pixel of a satellite view of each city, sorted from light to dark\n"
             "and cut into seven equal parts. Each chip is one seventh of the city.",
             ha="center", va="center", fontsize=8.3, color=S.MUTED, linespacing=1.35)
    fig.text(0.5, 0.018, "Imagery: Sentinel-2 cloudless 2020 by EOX (modified Copernicus Sentinel data), CC BY-NC-SA 4.0\n"
             f"#30DayMapChallenge {DAY}  ·  daniellecarrick.github.io/30-day-map-challenge/day25",
             ha="center", va="bottom", fontsize=6.2, color=S.MUTED, linespacing=1.4)
    fig.savefig(OUT / "paint_strips_linkedin.png", dpi=200)
    plt.close(fig)


def main():
    res, imgs = [], {}
    for name, lat, lon in CITIES:
        imgs[name] = mosaic(lat, lon)
        chips, Lmed = strip(imgs[name])
        res.append({"name": name, "lat": lat, "lon": lon, "chips": chips, "L": Lmed})
        print(f"{name:<12} median L {Lmed:4.0f}  " + " ".join(c["hex"] for c in chips))
    res.sort(key=lambda r: -r["L"])
    (OUT / "strips.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    draw_png(res, imgs)
    draw_linkedin(res, imgs)
    title = TITLE
    data = {"title": title, "day": DAY, "tiles": URL, "cities": res,
            "attribution": '<a href="https://s2maps.eu">Sentinel-2 cloudless 2020</a> by EOX',
            "source": 'Imagery: <a href="https://s2maps.eu">Sentinel-2 cloudless 2020</a> by EOX IT Services '
                      '(contains modified Copernicus Sentinel data), CC BY-NC-SA 4.0'}
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "paint_strips.html")
    print(f"wrote out/paint_strips.png and out/paint_strips.html ({size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
