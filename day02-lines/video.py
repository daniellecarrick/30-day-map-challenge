"""Day 2 · Lines — LinkedIn video of the Citi Bike wind map.

Renders the same flow as the published wind map (15-minute slots of net flow per hex, smoothed into
a field that particles drift along) to a 1080x1350 (4:5) MP4: an average weekday from 5am to
midnight in about 30 seconds, with a clock and the trips timeline.

Writes: out/linkedin_citibike_windmap.mp4 (+ out/linkedin_citibike_windmap_poster.png, one frame)
Needs:  imageio-ffmpeg (bundles ffmpeg):  ../.venv/bin/pip install imageio-ffmpeg
Run:    ../.venv/bin/python video.py          (about 2–3 minutes)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build as B

S = B.S
OUT = B.OUT
FONTS = Path(S.SHARED) / "fonts"

W, H, FPS = 1080, 1350, 30
SS = 2                      # supersampling for the particle layer
MAP_X, MAP_Y, MAP_W = 89, 400, 902   # map area (px); height follows the frame's aspect ratio
START_H, END_H = 5, 24      # hours shown
FRAMES_PER_SLOT = 12        # 15 minutes of day = 0.4 s of video
HOLD = 45                   # frames held at the end
N = 5000                    # particles
SP = 0.12                   # px per frame per (net trip an hour)
FADE = 0.9                  # trail alpha kept each frame
SIGMA_M, WMIN = 280, 0.3    # field smoothing, same as the page

INK, MUTED, BG, CTL = S.INK, S.MUTED, S.BG, S.TOKENS["color"]["light"]["ctl"]
ACCENT = S.TOKENS["color"]["light"]["accent"]
BLUES = S.SEQUENTIAL_BLUE[1:]
hexrgb = lambda c: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def clock(slot):
    m = int(slot) * B.SLOT_MIN
    h, mm = (m // 60) % 24, m % 60
    return f"{h % 12 or 12}:{mm:02d}{'am' if h < 12 else 'pm'}"


def blues(v, ref):
    """Sequential blue for a net-flow value (0..ref), as RGB 0-255."""
    stops = np.array([hexrgb(c) for c in BLUES], float)
    t = np.clip(v / ref, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def basemap(frame, boros, mw, mh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0, y0, x1, y1 = frame
    land = boros.clip((x0, y0, x1, y1))
    fig = plt.figure(figsize=(mw / 100, mh / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    land.plot(ax=ax, color=CTL, edgecolor="none")
    land.boundary.plot(ax=ax, color=INK, linewidth=0.5, alpha=0.4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_axis_off()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    S.apply()
    od, st, boros = B.load()
    _, tide_ = B.tide(st)
    keep, flow, trips = B.slot_flow(st)
    F = flow["wd"]                                   # [slot, hex, (vx, vy)] net trips/hour
    trips = np.array(trips["wd"], float)
    frame = B.windmap_frame()
    x0, y0, x1, y1 = frame
    MW = MAP_W
    MH = round(MW * (y1 - y0) / (x1 - x0))
    mpp = (x1 - x0) / MW                             # metres per map pixel
    print(f"map {MW}x{MH}px, {mpp:.1f} m/px")

    # hex centres in map px
    q = np.array([k[0] for k in keep]); r = np.array([k[1] for k in keep])
    hx, hy = B.hex_center(q, r)
    px, py = (hx - x0) / mpp, (y1 - hy) / mpp

    # ---- field grid: Gaussian weights from each hex to nearby cells (same method as the page) ----
    G = 3
    GW, GH = int(np.ceil(MW / G)), int(np.ceil(MH / G))
    sig, rad = SIGMA_M / mpp, 2.5 * SIGMA_M / mpp
    cells, hexi, ws = [], [], []
    for h in range(len(keep)):
        c0, c1 = max(0, int((px[h] - rad) // G)), min(GW - 1, int((px[h] + rad) // G))
        r0, r1 = max(0, int((py[h] - rad) // G)), min(GH - 1, int((py[h] + rad) // G))
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        d2 = ((cc + 0.5) * G - px[h]) ** 2 + ((rr + 0.5) * G - py[h]) ** 2
        m = d2 <= rad * rad
        cells.append((rr * GW + cc)[m]); ws.append(np.exp(-d2[m] / (2 * sig * sig))); hexi.append(np.full(m.sum(), h))
    cells, hexi, ws = np.concatenate(cells), np.concatenate(hexi), np.concatenate(ws)
    NC = GW * GH
    Wsum = np.bincount(cells, ws, NC)
    valid = np.flatnonzero(Wsum >= WMIN)
    inv = np.where(Wsum >= WMIN, 1 / np.maximum(Wsum, 1e-9), 0)
    fields = {}

    def field(s):
        s %= F.shape[0]
        if s not in fields:
            fields[s] = (np.bincount(cells, ws * F[s, hexi, 0], NC) * inv,
                         np.bincount(cells, ws * F[s, hexi, 1], NC) * inv)
        return fields[s]

    # ---- particles (map px) ----
    rng = np.random.default_rng(2)

    def spawn(n):
        i = valid[rng.integers(0, len(valid), n)]
        return (i % GW + rng.random(n)) * G, (i // GW + rng.random(n)) * G, rng.uniform(40, 110, n)

    P_x, P_y, P_life = spawn(N)
    P_age = rng.uniform(0, 1, N) * P_life

    def advance(t):
        """Move every particle one frame through the field at time t (slots); return segments + speeds."""
        nonlocal P_x, P_y, P_age, P_life
        s0 = int(t); fr = t - s0
        (U0, V0), (U1, V1) = field(s0), field(s0 + 1)
        c = (P_x // G).astype(int); rr = (P_y // G).astype(int)
        inside = (c >= 0) & (c < GW) & (rr >= 0) & (rr < GH)
        i = np.where(inside, rr * GW + c, 0)
        ok = inside & (Wsum[i] >= WMIN) & (P_age < P_life)
        u = U0[i] + (U1[i] - U0[i]) * fr
        v = V0[i] + (V1[i] - V0[i]) * fr
        nx, ny = P_x + u * SP, P_y - v * SP
        sp = np.hypot(u, v)
        seg = ok & (sp >= 1)
        out = (P_x[seg], P_y[seg], nx[seg], ny[seg], sp[seg])
        P_x, P_y, P_age = nx, ny, P_age + 1
        dead = ~ok
        if dead.any():
            a, b, l = spawn(dead.sum())
            P_x[dead], P_y[dead], P_life[dead], P_age[dead] = a, b, l, 0
        return out

    # ---- static chrome ----
    base = basemap(frame, boros, MW * SS, MH * SS).astype(np.float32) / 255
    chrome = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(chrome)
    d.text((64, 52), "Citi Bike's daily tide", font=font("Bitter-Bold.ttf", 62), fill=INK)
    d.text((64, 138), f"Every weekday, bikes flow toward Midtown until about {tide_['turn_clock']},\n"
           "then turn around. Each streak follows the net flow of trips;\nfaster, darker streaks carry more.",
           font=font("Raleway-Regular.ttf", 27), fill=MUTED, spacing=8)
    # timeline: trips started per 15 minutes, 5am -> midnight
    TL_X0, TL_X1, TL_Y0, TL_Y1 = 64, W - 64, 312, 362
    s_first, s_last = START_H * 4, END_H * 4
    nbar = s_last - s_first
    bw = (TL_X1 - TL_X0) / nbar
    tmax = trips[s_first:s_last].max()
    bars = []
    for k in range(nbar):
        hgt = max(1, (TL_Y1 - TL_Y0) * trips[s_first + k] / tmax)
        box = (TL_X0 + k * bw + 0.8, TL_Y1 - hgt, TL_X0 + (k + 1) * bw - 0.8, TL_Y1)
        bars.append(box)
        d.rectangle(box, fill=CTL)
    tick_f = font("Raleway-Regular.ttf", 22)
    for hh in (6, 9, 12, 15, 18, 21):
        x = TL_X0 + (hh - START_H) * 4 * bw
        d.text((x, TL_Y1 + 6), "noon" if hh == 12 else f"{hh % 12}{'am' if hh < 12 else 'pm'}", font=tick_f, fill=MUTED)
    d.text((64, H - 52), "Citi Bike trips, average weekday, Sept 2026 · Data: Citi Bike System Data\n"
           "#30DayMapChallenge Day 2: Lines · daniellecarrick.github.io/30-day-map-challenge",
           font=font("Raleway-Regular.ttf", 18), fill=MUTED, spacing=4)
    clock_f, label_f = font("Bitter-Bold.ttf", 44), font("Raleway-SemiBold.ttf", 24)

    # ---- render ----
    import imageio_ffmpeg
    out = OUT / "linkedin_citibike_windmap.mp4"
    writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2,
                                         output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"])
    writer.send(None)
    trail = np.zeros((MH * SS, MW * SS, 4), np.float32)   # premultiplied RGB + alpha
    BUCK = 6
    bucket_rgb = [tuple(int(c) for c in blues(np.array((b + 0.5) / BUCK * B.REF_HOUR), B.REF_HOUR)) for b in range(BUCK)]
    t = float(s_first)
    for _ in range(90):                                   # warm up the trails before the first frame
        advance(t)
    total = nbar * FRAMES_PER_SLOT + HOLD
    for fi in range(total):
        if fi < nbar * FRAMES_PER_SLOT:
            t = s_first + fi / FRAMES_PER_SLOT
        x0s, y0s, x1s, y1s, sp = advance(min(t, s_last - 1e-3))
        layer = Image.new("RGBA", (MW * SS, MH * SS), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        bk = np.minimum((sp / B.REF_HOUR * BUCK).astype(int), BUCK - 1)
        for a0, b0, a1, b1, k in zip(x0s * SS, y0s * SS, x1s * SS, y1s * SS, bk):
            ld.line((a0, b0, a1, b1), fill=bucket_rgb[k] + (255,), width=3)
        seg = np.asarray(layer, np.float32) / 255
        sa = seg[..., 3:4]
        trail *= FADE
        trail[..., :3] = seg[..., :3] * sa + trail[..., :3] * (1 - sa)
        trail[..., 3:4] = sa + trail[..., 3:4] * (1 - sa)
        comp = base * (1 - trail[..., 3:4]) + trail[..., :3]
        mimg = Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).resize((MW, MH), Image.LANCZOS)
        img = chrome.copy()
        img.paste(mimg, (MAP_X, MAP_Y))
        dd = ImageDraw.Draw(img)
        cur = int(t) if fi < nbar * FRAMES_PER_SLOT else s_last - 1
        dd.rectangle(bars[cur - s_first], fill=ACCENT)
        dd.text((64, 248), clock(cur), font=clock_f, fill=INK)
        dd.text((300, 262), "Average weekday", font=label_f, fill=MUTED)
        writer.send(np.asarray(img).tobytes())
        if fi == 3 * 4 * FRAMES_PER_SLOT + 6:              # ~8:00am: a poster frame for the post preview
            img.save(OUT / "linkedin_citibike_windmap_poster.png")
        if fi % 120 == 0:
            print(f"frame {fi}/{total}  {clock(cur)}", flush=True)
    writer.close()
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB, {total / FPS:.1f} s)")


if __name__ == "__main__":
    main()
