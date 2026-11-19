"""Day 19 · social GIF: the lilac wave rolling north, March to August.

Drawn at 1080 × 1350, saved at 864 × 1080 (4:5 portrait). Same grid, colors and "in bloom now" band as the
page: cells take their date color when lilacs first open and keep it; the 2-week bloom band gets ink edges;
not-yet cells are pale. Ends on the whole-season map (the still map), held for a few seconds.
Override with GIF_STEP (days per frame), GIF_FPS, GIF_COLORS, GIF_SCALE.

Writes out/lilac_wave.gif and out/lilac_wave_frame.png
Run:  ../.venv/bin/python make_gif.py   (after fetch.sh)
"""
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rasterio import features

import build as B

S = B.S
HERE = Path(__file__).parent
OUT = HERE / "out"
FONTS = HERE.parent / "shared" / "fonts"
W, H = 1080, 1350
STEP = int(os.environ.get("GIF_STEP", 2))           # days per frame
FPS = float(os.environ.get("GIF_FPS", 10))
COLORS = int(os.environ.get("GIF_COLORS", 128))
OUT_SCALE = float(os.environ.get("GIF_SCALE", 0.8))
BLOOM_DAYS = 14
FIRST, LAST = 68, 216                                # Mar 9 .. Aug 4
MAP_TOP, MAP_W = 352, 1040
TL_TOP, TL_H = 1035, 120


def hex_rgb(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], float)


def main():
    S.apply()
    A, tr, crs = B.load_stack()
    normal = np.nanmean(A[(B.YEARS >= 1991) & (B.YEARS <= 2020)], 0)
    normal[(~np.isnan(A[(B.YEARS >= 1991) & (B.YEARS <= 2020)])).sum(0) < 10] = np.nan
    (n,), gtr = B.to_albers([normal], tr, crs, normal.shape)
    gh, gw = n.shape
    states = gpd.read_file(f"zip://{B.DATA / 'cb_2020_us_state_500k.zip'}")
    states = states[~states.STUSPS.isin(["AK", "HI", "PR", "VI", "GU", "MP", "AS"])].to_crs(B.CRS)
    land = features.rasterize(((g, 1) for g in states.geometry), out_shape=(gh, gw), transform=gtr, dtype="uint8")
    n[land == 0] = np.nan

    # crop the grid to the lower 48
    rows, cols = np.where(land > 0)
    r0, r1, c0, c1 = rows.min() - 2, rows.max() + 3, cols.min() - 2, cols.max() + 3
    n = n[r0:r1, c0:c1]
    gh, gw = n.shape
    sc = MAP_W / gw
    mw, mh = MAP_W, round(gh * sc)
    mx0 = (W - mw) // 2
    has = ~np.isnan(n)
    v = np.where(has, np.round(n), 0).astype(int)
    warm = has & (v < B.WARM)
    lilac = has & ~warm

    bg, ink, muted = hex_rgb(S.BG), hex_rgb(S.INK), hex_rgb(S.MUTED)
    ctl = hex_rgb(S.C["ctl"])
    warm_c = hex_rgb(S.make_scale([(0, S.CLOSED), (1, S.BG)])(0.5))
    season = S.make_scale(B.SEASON_STOPS)
    lut = np.array([hex_rgb(season(d)) for d in range(256)])

    # state lines + nation outline, drawn once (2x supersampled)
    ss = 2
    lines = Image.new("RGBA", (mw * ss, mh * ss), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    gx0, gy1 = gtr.c + c0 * B.RES, gtr.f - r0 * B.RES

    def rings(geom):
        for poly in getattr(geom, "geoms", [geom]):
            yield poly.exterior
            yield from poly.interiors

    def xy(ring):
        a = np.asarray(ring.coords)
        return [((x - gx0) / B.RES * sc * ss, (gy1 - y) / B.RES * sc * ss) for x, y in a]
    for g in states.geometry.simplify(1500):
        for ring in rings(g):
            ld.line(xy(ring), fill=(255, 255, 255, 235), width=2)
    for ring in rings(states.geometry.union_all().simplify(1500)):
        ld.line(xy(ring), fill=tuple(int(c) for c in ink) + (255,), width=2)
    lines = lines.resize((mw, mh), Image.LANCZOS)

    # share of lilac country in bloom each day (the page's timeline)
    starts = np.bincount(v[lilac], minlength=400).astype(float)
    total = lilac.sum()
    series = {d: starts[d - BLOOM_DAYS + 1:d + 1].sum() / total for d in range(B.D0, B.D1 + 1)}
    peak = max(series.values())

    head = ImageFont.truetype(str(FONTS / "Bitter-Bold.ttf"), 54)
    big = ImageFont.truetype(str(FONTS / "Bitter-Bold.ttf"), 72)
    body = ImageFont.truetype(str(FONTS / "Raleway-Regular.ttf"), 27)
    small = ImageFont.truetype(str(FONTS / "Raleway-Regular.ttf"), 21)
    lab = ImageFont.truetype(str(FONTS / "Raleway-SemiBold.ttf"), 22)
    col = lambda c: tuple(int(x) for x in c)

    def frame(day):
        rgb = np.empty((gh, gw, 3))
        rgb[:] = bg
        rgb[warm] = warm_c
        if day is None:
            rgb[lilac] = lut[v[lilac]]
        else:
            opened = lilac & (v <= day)
            rgb[lilac & ~opened] = ctl
            rgb[opened] = lut[v[opened]]
            band = opened & (day - v < BLOOM_DAYS)
            nb = np.zeros_like(band)               # band cells next to a lilac cell outside the band
            for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                other = np.roll(lilac & ~band, sh, axis=(0, 1))
                nb |= band & other
            rgb[nb] = ink
        m = Image.fromarray(rgb.astype(np.uint8)).resize((mw, mh), Image.BILINEAR)
        m.paste(lines, (0, 0), lines)
        img = Image.new("RGB", (W, H), col(bg))
        img.paste(m, (mx0, MAP_TOP))
        d = ImageDraw.Draw(img)
        d.text((W / 2, 62), "The smell of lilacs moves north", font=head, fill=col(ink), anchor="mm")
        d.text((W / 2, 124), "about 13 miles a day", font=head, fill=col(ink), anchor="mm")
        d.text((W / 2, 186), "When lilacs first open in a typical spring (1991–2020)", font=body, fill=col(muted), anchor="mm")
        if day is None:
            d.text((W / 2, 262), "March to August", font=big, fill=col(ink), anchor="mm")
            d.text((W / 2, 318), "Atlanta in late March, Maine in late May, the Rockies in July",
                   font=body, fill=col(muted), anchor="mm")
        else:
            d.text((W / 2, 262), B.doy_label(day), font=big, fill=col(ink), anchor="mm")
            d.text((W / 2, 318), f"{series[day]:.0%} of lilac country in bloom", font=body, fill=col(muted), anchor="mm")

        # timeline
        x0, x1 = 70, W - 70
        tx = lambda dd: x0 + (dd - B.D0) / (B.D1 - B.D0) * (x1 - x0)
        yb = TL_TOP + TL_H
        ty = lambda s: yb - s / peak * (TL_H - 30)
        pts = [(tx(dd), ty(s)) for dd, s in series.items()]
        d.polygon([(tx(B.D0), yb)] + pts + [(tx(B.D1), yb)], fill=col(hex_rgb(S.SEQUENTIAL_PURPLE[4])))
        d.line([(x0, yb), (x1, yb)], fill=col(muted), width=1)
        d.text((x0, TL_TOP + 2), "Share of lilac country in bloom", font=small, fill=col(muted), anchor="la")
        for dd in (60, 91, 121, 152, 182, 213):
            d.text((tx(dd), yb + 10), B.doy_label(dd), font=small, fill=col(muted),
                   anchor="la" if dd == 60 else "ma")
        if day is not None:
            d.line([(tx(day), TL_TOP + 26), (tx(day), yb)], fill=col(ink), width=3)

        # key
        ky, kx = 1232, 70
        def sw(x, fill, outline=None):
            d.rectangle([x, ky - 11, x + 22, ky + 11], fill=col(fill), outline=col(outline) if outline is not None else None,
                        width=3)
            return x + 32
        kx = sw(kx, lut[121], ink)
        d.text((kx, ky), "In bloom now", font=lab, fill=col(ink), anchor="lm")
        kx += 170
        for k, dd in enumerate((80, 110, 140, 170)):
            d.rectangle([kx + 11 * k, ky - 11, kx + 11 * (k + 1), ky + 11], fill=col(lut[dd]))
        kx += 54
        d.text((kx, ky), "Opened", font=lab, fill=col(ink), anchor="lm")
        kx += 120
        kx = sw(kx, ctl)
        d.text((kx, ky), "Not yet", font=lab, fill=col(ink), anchor="lm")
        kx += 110
        kx = sw(kx, warm_c)
        d.text((kx, ky), "Too warm for lilacs", font=lab, fill=col(ink), anchor="lm")

        d.text((W / 2, H - 52), "Modeled first bloom of a cloned lilac from daily temperatures · Data: USA National Phenology Network",
               font=small, fill=col(muted), anchor="mm")
        d.text((W / 2, H - 24), "#30DayMapChallenge Day 19: Senses: Smell", font=small, fill=col(muted), anchor="mm")
        return img

    days = list(range(FIRST, LAST + 1, STEP))
    frames = [frame(dd) for dd in days] + [frame(None)]
    durations = [int(1000 / FPS)] * len(days) + [3500]
    durations[0] = 900

    sample = Image.new("RGB", (W, H * 4))
    for k, f in enumerate([frames[len(days) // 3], frames[len(days) // 2], frames[2 * len(days) // 3], frames[-1]]):
        sample.paste(f, (0, H * k))
    pal = sample.quantize(colors=COLORS, method=Image.MEDIANCUT)
    size = (round(W * OUT_SCALE), round(H * OUT_SCALE))
    q = [f.resize(size, Image.LANCZOS).quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(OUT / "lilac_wave.gif", save_all=True, append_images=q[1:], duration=durations, loop=0,
              optimize=True, disposal=1)
    frames[days.index(128)].save(OUT / "lilac_wave_frame.png")
    print(f"wrote out/lilac_wave.gif {(OUT / 'lilac_wave.gif').stat().st_size / 1e6:.2f} MB, {len(frames)} frames, "
          f"{sum(durations) / 1000:.1f} s")


if __name__ == "__main__":
    main()
