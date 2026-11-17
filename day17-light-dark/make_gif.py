"""Day 17 · social GIF: every Atlantic-coast lighthouse blinking its real signal.

1080 × 1350 (4:5 portrait), 20 s at 12.5 fps. Frames are drawn in Python (no browser): the
same lighthouses, colours and phases logic as the page. A light that is on at any moment of a
frame's 80 ms is drawn lit, so the briefest flashes still show.

Writes out/east_coast.gif and out/preview_card.png (link-preview card, used by publish.py)
Run:  ../.venv/bin/python make_gif.py   (after build.py has downloaded the data)
"""
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box

import build as B

S = B.S
HERE = Path(__file__).parent
OUT = HERE / "out"
FONTS = HERE.parent / "shared" / "fonts"
W, H = 1080, 1350
TOP, BOTTOM = 200, 64                 # header and footer bands, px
FPS, SECONDS = 12.5, 20
SS = 2                                # supersampling for the land layer
CRS = "+proj=tmerc +lat_0=36 +lon_0=-75 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m"   # keeps the coast upright
DARK = B.DARK
# a few well-known lights to label (only those present get a label)
LABELS = ["Portland Head", "Boston", "Cape May", "Cape Hatteras", "Saint Augustine", "Jupiter Inlet"]
LABEL_WEST = {"Boston"}     # inland side: the sea side is crowded with Cape Cod's lights


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def glow_sprite(rgb, r):
    """Additive glow: a hot white core fading through the light's colour to nothing."""
    n = int(r * 2) + 1
    y, x = np.mgrid[:n, :n] - r
    d = np.hypot(x, y) / r
    fade = np.clip(1 - d, 0, 1) ** 1.5                     # round, reaching exactly zero at the edge
    halo = (np.exp(-(d * 3.2) ** 2) + 0.25 * np.exp(-(d * 1.4) ** 2)) * fade
    core = np.exp(-(d * 10) ** 2)
    return halo[..., None] * rgb[None, None] + core[..., None] * 255


def main():
    lh = [r for r in B.lighthouses() if r["region"] == "Atlantic"]
    pts = gpd.GeoSeries(gpd.points_from_xy([r["lon"] for r in lh], [r["lat"] for r in lh]), crs=4326).to_crs(CRS)
    xs, ys = pts.x.values, pts.y.values
    # fit the lighthouses into the map band, with a margin
    mx0, mx1, my0, my1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 60
    scale = min((W - 2 * pad) / (mx1 - mx0), (H - TOP - BOTTOM - 2 * pad) / (my1 - my0))
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
    to_px = lambda x, y: (W / 2 + (x - cx) * scale, TOP + (H - TOP - BOTTOM) / 2 - (y - cy) * scale)
    px = np.array([to_px(x, y) for x, y in zip(xs, ys)])

    # ---- land (drawn once, supersampled) ----
    half_w, half_h = W / 2 / scale, (H - TOP - BOTTOM) / 2 / scale
    view = box(cx - half_w * 1.4, cy - half_h * 1.4, cx + half_w * 1.4, cy + half_h * 1.4)
    # crop in lon/lat first: projecting the whole world into this local projection tears far-off shapes
    area = (-95, 15, -50, 55)
    ne = gpd.read_file(B.DATA / "ne" / "ne_10m_admin_0_countries.zip").clip(area).to_crs(CRS)
    ne["geometry"] = ne.geometry.buffer(0).intersection(view)
    states = gpd.read_file(HERE.parent / "day08-utopia" / "data" / "raw" / "states" / "states.zip").to_crs(4326)
    states = states.clip(area).to_crs(CRS)
    land = Image.new("RGB", (W * SS, H * SS), DARK["bg"])
    dr = ImageDraw.Draw(land)
    sp = lambda x, y: (to_px(x, y)[0] * SS, to_px(x, y)[1] * SS)

    def polys(g):
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                yield p

    for g in ne.geometry:
        for p in polys(g):
            dr.polygon([sp(x, y) for x, y in p.exterior.coords], fill=DARK["ctl"])
            for hole in p.interiors:
                dr.polygon([sp(x, y) for x, y in hole.coords], fill=DARK["bg"])
    for g in states.boundary:
        for ln in getattr(g, "geoms", [g]):
            dr.line([sp(x, y) for x, y in ln.coords], fill=DARK["bg"], width=SS)
    land = land.resize((W, H), Image.LANCZOS)
    # fade the map out behind the header and footer so the text sits on plain dark
    arr = np.asarray(land, float)
    bg = hex_rgb(DARK["bg"])
    yy = np.arange(H)[:, None, None]
    a = np.clip((TOP + 10 - yy) / 70, 0, 1) + np.clip((yy - (H - BOTTOM - 30)) / 50, 0, 1)
    land = Image.fromarray((arr * (1 - np.clip(a, 0, 1)) + bg * np.clip(a, 0, 1)).astype(np.uint8))

    # ---- text (drawn once) ----
    head = ImageFont.truetype(str(FONTS / "Bitter-Bold.ttf"), 50)
    body = ImageFont.truetype(str(FONTS / "Raleway-Regular.ttf"), 25)
    small = ImageFont.truetype(str(FONTS / "Raleway-Regular.ttf"), 19)
    lab = ImageFont.truetype(str(FONTS / "Raleway-SemiBold.ttf"), 19)
    n_sig = len({(r["label"], tuple(r["seq"])) for r in lh})
    td = ImageDraw.Draw(land)
    td.text((W / 2, 62), f"{len(lh)} lighthouses, {n_sig} ways to blink", font=head, fill=DARK["ink"], anchor="mm")
    td.text((W / 2, 118), "Every lighthouse on the Atlantic coast, flashing its official signal", font=body,
            fill=DARK["muted"], anchor="mm")
    td.text((W / 2, 152), "at real speed, so sailors can tell which one they're looking at.", font=body,
            fill=DARK["muted"], anchor="mm")
    td.text((W / 2, H - 40), "Data: NOAA nautical charts · names © OpenStreetMap contributors", font=small,
            fill=DARK["muted"], anchor="mm")
    td.text((W / 2, H - 16), "#30DayMapChallenge Day 17: Light & dark", font=small, fill=DARK["muted"], anchor="mm")
    names = {B.short(r["name"]): i for i, r in enumerate(lh)}
    for nm in LABELS:
        i = names.get(nm)
        if i is None:
            continue
        x, y = px[i]
        # label out to sea (east) unless it would cover neighbouring lights; signal code under it
        west = nm in LABEL_WEST
        dx, anchor = (-26, "rs") if west else (26, "ls")
        td.text((x + dx, y - 2), nm, font=lab, fill=DARK["ink"], anchor=anchor)
        td.text((x + dx, y + 20), lh[i]["label"], font=small, fill=DARK["muted"], anchor=anchor)
    base = np.asarray(land, float)

    # ---- unlit lamps: faint dots, baked into the base ----
    for (x, y), r in zip(px, lh):
        col = hex_rgb(B.LIGHT[r["colours"][0]])
        xi, yi = int(round(x)), int(round(y))
        base[yi - 1:yi + 2, xi - 1:xi + 2] = base[yi - 1:yi + 2, xi - 1:xi + 2] * 0.55 + col * 0.45

    # ---- per-light rhythm, as on/off edges in seconds ----
    sprites = []
    for i, r in enumerate(lh):
        rad = 8 + math.sqrt(r["range"] or 4) * 2.2
        sprites.append(glow_sprite(hex_rgb(B.LIGHT[r["colours"][0]]), rad))
        r["phase"] = ((i * 0.6180339887) % 1) * r["period"]

    def lit_fraction(r, t0, dt, steps=8):
        on = 0
        for k in range(steps):
            tt = (t0 + r["phase"] + dt * k / steps) % r["period"]
            acc = 0.0
            for is_on, d in r["seq"]:
                acc += d
                if tt < acc:
                    on += is_on
                    break
        return on / steps

    dt = 1 / FPS
    n = int(FPS * SECONDS)
    frames = []
    for f in range(n):
        glow = np.zeros_like(base)
        for (x, y), r, spr in zip(px, lh, sprites):
            a = min(1.0, 2 * lit_fraction(r, f * dt, dt))
            if a <= 0:
                continue
            h = spr.shape[0] // 2
            x0, y0 = int(round(x)) - h, int(round(y)) - h
            ys_, xs_ = slice(max(y0, 0), min(y0 + spr.shape[0], H)), slice(max(x0, 0), min(x0 + spr.shape[1], W))
            sub = spr[ys_.start - y0:ys_.stop - y0, xs_.start - x0:xs_.stop - x0]
            glow[ys_, xs_] += sub * a
        img = base + (255 - base) * (1 - np.exp(-glow / 110))      # overlapping glows saturate softly
        frames.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
        if f % 50 == 0:
            print(f"frame {f}/{n}", flush=True)

    # one palette for every frame, so colours don't shimmer between frames
    sample = Image.new("RGB", (W, H * 4))
    for k, f in enumerate(frames[:: max(1, n // 4)][:4]):
        sample.paste(f, (0, H * k))
    pal = sample.quantize(colors=256, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(OUT / "east_coast.gif", save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0,
              optimize=True, disposal=1)
    frames[n // 3].save(OUT / "east_coast_frame.png")
    # link-preview card (1200 x 630): New England to Cape Hatteras from a lit frame
    top = 225
    frames[n // 3].crop((0, top, W, top + round(W * 630 / 1200))).resize((1200, 630), Image.LANCZOS).save(
        OUT / "preview_card.png")
    print("wrote out/east_coast.gif", f"{(OUT / 'east_coast.gif').stat().st_size / 1e6:.1f} MB,", n, "frames,",
          len(lh), "lighthouses")


if __name__ == "__main__":
    main()
