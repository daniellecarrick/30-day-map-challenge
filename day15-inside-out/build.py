"""Day 15 · Inside out: turn the Earth's relief upside down.

Every point's height is mirrored around sea level (h -> -h), so ocean trenches become the
highest mountains and the old continents become sea basins. The page is a three.js globe that
pushes the real relief out of the sphere (exaggerated) and animates between the two Earths.

Writes:
  - out/height.webp elevation as lossless WebP: (metres + 11000) / 4 split into R (high byte) and G (low byte)
  - out/map.html    interactive globe, self-contained (three.js from a CDN)
  - out/map.png     both globes side by side (make_png.py, via headless Chrome)

Data: ETOPO1 global relief, NOAA NCEI, via the CoastWatch ERDDAP (etopo180), 2 arc-minute stride:
  https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.html
Named peaks and deeps use published survey values (the grid smooths extremes).
Run:  ../.venv/bin/python build.py
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import style as S  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DAY = "Day 15: Inside out"
TITLE = "Turn the Earth inside out and six Pacific trenches tower over Everest"
SOURCE = ('Data: <a href="https://www.ncei.noaa.gov/products/etopo-global-relief-model">ETOPO1, NOAA NCEI</a>; '
          "named depths and heights from published surveys")
OFFSET = 11000          # metres added before 16-bit encoding (deepest point is about -10,935)
FACTOR = 5              # 2' grid -> 10' (2160 x 1080 px)
STEP = 4                # metres per stored unit; invisible at any exaggeration the page allows

# Published values (metres). Positive = above sea level. kind: deep | peak | low | ridge
FEATURES = [
    ("Challenger Deep, Mariana Trench", 11.37, 142.59, -10935, "deep"),
    ("Horizon Deep, Tonga Trench", -23.26, -174.73, -10800, "deep"),
    ("Emden Deep, Philippine Trench", 10.42, 126.68, -10540, "deep"),
    ("Scholl Deep, Kermadec Trench", -31.93, -177.30, -10047, "deep"),
    ("Izu–Ogasawara Trench", 29.10, 142.80, -9780, "deep"),
    ("Kuril–Kamchatka Trench", 44.15, 150.50, -9600, "deep"),
    ("Milwaukee Deep, Puerto Rico Trench", 19.77, -66.93, -8376, "deep"),
    ("Meteor Deep, South Sandwich Trench", -55.67, -25.93, -8265, "deep"),
    ("Richards Deep, Atacama Trench", -23.33, -71.33, -8065, "deep"),
    ("Java (Sunda) Trench", -11.17, 118.47, -7290, "deep"),
    ("Mount Everest", 27.99, 86.93, 8849, "peak"),
    ("K2", 35.88, 76.51, 8611, "peak"),
    ("Aconcagua", -32.65, -70.01, 6961, "peak"),
    ("Denali", 63.07, -151.01, 6190, "peak"),
    ("Kilimanjaro", -3.07, 37.35, 5895, "peak"),
    ("Mont Blanc", 45.83, 6.86, 4806, "peak"),
    ("Mauna Kea", 19.82, -155.47, 4207, "peak"),
    ("Dead Sea shore", 31.50, 35.50, -430, "low"),
    ("Mid-Atlantic Ridge", 0.0, -18.0, None, "ridge"),
]
EVEREST = 8849


def load_grid() -> np.ndarray:
    """ETOPO1 at 2', north row first, 5400 x 10800 (drops the duplicate pole row and 180° column)."""
    f = netcdf_file(DATA / "etopo1_2min.nc", "r", mmap=False)
    a = f.variables["altitude"].data.astype(np.float32)[::-1]
    f.close()
    return a[:5400, :10800]


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def encode_height(h: np.ndarray) -> bytes:
    v = np.clip(np.round((h + OFFSET) / STEP), 0, 65535).astype(np.uint16)
    rgb = np.zeros(v.shape + (3,), np.uint8)
    rgb[..., 0] = v >> 8
    rgb[..., 1] = v & 255
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "WEBP", lossless=True, quality=100, method=6)
    return buf.getvalue()


def median_ocean_depth(a: np.ndarray) -> int:
    """Area-weighted median depth of the ocean floor (metres, negative), rounded to 100 m.
    The inside-out globe is drawn with this level as its base radius, so the typical seafloor
    sits flat and only what rises above it (trenches) sticks out."""
    w = np.broadcast_to(np.cos(np.radians(np.linspace(90, -90, a.shape[0])))[:, None], a.shape)
    o = a < 0
    d, wt = a[o], w[o]
    i = np.argsort(d)
    c = np.cumsum(wt[i]) / wt.sum()
    return int(round(float(d[i][np.searchsorted(c, 0.5)]) / 100) * 100)


def main():
    grid = load_grid()
    floor = median_ocean_depth(grid)
    h = block_mean(grid, FACTOR)
    blob = encode_height(h)
    (OUT / "height.webp").write_bytes(blob)

    deeps = [f for f in FEATURES if f[4] == "deep" and -f[3] > EVEREST]
    print(f"heightmap {h.shape[1]}x{h.shape[0]}, {len(blob) / 1e6:.1f} MB; "
          f"{len(deeps)} deeps deeper than Everest is tall: {[d[0] for d in deeps]}")

    tokens = json.loads((S.SHARED / "tokens.json").read_text())
    data = {
        "title": TITLE, "day": DAY, "source": SOURCE,
        "height": "data:image/webp;base64," + base64.b64encode(blob).decode(),
        "offset": OFFSET, "step": STEP,
        "diverging": tokens["scales"]["diverging"],
        "features": [{"name": n, "lat": la, "lon": lo, "h": v, "kind": k} for n, la, lo, v, k in FEATURES],
        "everest": EVEREST,
        "floor": floor,
    }
    print(f"median ocean floor: {floor:,} m")
    size = S.build_page(HERE / "web" / "index.html", data, OUT / "map.html")
    print("wrote out/map.html", f"({size / 1e6:.1f} MB)")

    import make_png  # renders the page's globes in headless Chrome, so it runs after map.html exists
    make_png.main(data)


if __name__ == "__main__":
    main()
