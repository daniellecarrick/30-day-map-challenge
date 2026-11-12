"""Day 11 · LinkedIn GIF (1080x1350): one day of NYC noise complaints, looping.

Districts are colored by their signature sound and brighten in their busy hours; each ripple is a
sample of real complaints that hour, colored by sound. The clock runs 6 am -> 6 am so the loop is seamless.
Run:  ../.venv/bin/python make_gif.py     -> out/day11_linkedin.gif (+ out/day11_linkedin.png, a still)
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402
S = B.S

FPS, SECONDS = 12, 12            # 24 hours in 12 s: half a second per hour
N = FPS * SECONDS
W, H = 1080, 1350
START_HOUR = 6
LIFE, R_MAX = 1.1, 2300          # ripple lifetime (s) and final radius (m)
PER_RIPPLE = 550                # complaint-days per ripple
FRAMES = HERE / "out" / "gif_frames"
FRAMES.mkdir(parents=True, exist_ok=True)

S.apply()
gdf, days, lq, city, hourly, idx = B.load()
res = gdf[gdf.res].reset_index(drop=True)
INK, MUTED = S.INK, S.MUTED
CAT = S.CATEGORICAL
hour_tot = np.array([hourly[idx[cd]].sum(1) for cd in res.boro_cd])      # district x hour
pulse = hour_tot / hour_tot.max(1, keepdims=True)

# ripple events: (time in s, x, y, sound), sampled from each district's real hourly mix
rng = np.random.default_rng(11)
events = []
for k, cd in enumerate(res.boro_cd):
    pts = B.sample_points(res.geometry[k], n=40, seed=int(cd))
    for step in range(24):
        h = (START_HOUR + step) % 24
        mix = hourly[idx[cd], h]
        n = rng.poisson(mix.sum() / PER_RIPPLE)
        for _ in range(n):
            s = rng.choice(8, p=mix / mix.sum())
            x, y = pts[rng.integers(len(pts))]
            events.append((step * SECONDS / 24 + rng.random() * SECONDS / 24, x, y, s))
events = np.array(events)
print(len(events), "ripples")


def frame(i):
    t = i / FPS
    step = t / SECONDS * 24
    h = (START_HOUR + int(step)) % 24
    f = step - int(step)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.955, "Loud music tops noise complaints\nin 55 of 59 NYC districts", fontsize=40,
             fontweight="bold", family=S.HEAD_FONT, va="top", color=INK, linespacing=1.15)
    fig.text(0.07, 0.835, "Each district is colored by the sound it complains about\nmost compared with the rest of the city.",
             fontsize=20, color=MUTED, va="top", linespacing=1.3)

    ax = fig.add_axes([0.04, 0.29, 0.96, 0.505])
    gdf[~gdf.res].plot(ax=ax, color="#e4e3df", edgecolor="white", linewidth=0.4)
    a = 0.3 + 0.7 * (pulse[:, h] * (1 - f) + pulse[:, (h + 1) % 24] * f)
    res.plot(ax=ax, color=[to_rgba(CAT[int(s)], al) for s, al in zip(res.sig, a)], edgecolor="white", linewidth=0.6)
    age = (t - events[:, 0]) % SECONDS
    live = events[age < LIFE]
    ages = age[age < LIFE]
    if len(live):
        circles = [Circle((x, y), 120 + R_MAX * (g / LIFE)) for (_, x, y, _), g in zip(live, ages)]
        col = [to_rgba(CAT[int(s)], 1 - g / LIFE) for s, g in zip(live[:, 3], ages)]
        lw = 3.2 * (1 - ages / LIFE) + 0.8
        halo = [(1, 1, 1, 0.9 * (1 - g / LIFE)) for g in ages]   # white under each ring so it shows on its own color
        ax.add_collection(PatchCollection(circles, facecolor="none", edgecolor=halo, linewidths=lw + 2.4, zorder=5))
        ax.add_collection(PatchCollection([Circle(c.center, c.radius) for c in circles], facecolor="none",
                                          edgecolor=col, linewidths=lw, zorder=6))
    ax.set_axis_off()
    ax.set_aspect("equal")

    # clock, in the empty New Jersey corner
    hr = (START_HOUR + int(step)) % 24
    label = B.clock(hr)
    fig.text(0.07, 0.76, label, fontsize=54, fontweight="bold", family=S.HEAD_FONT, color=INK, va="top")
    fig.text(0.07, 0.655, "each ring = complaints\nthat hour, by sound", fontsize=17, color=MUTED, va="top", linespacing=1.3)
    # day progress bar under the clock
    bx = fig.add_axes([0.07, 0.675, 0.22, 0.008])
    bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.set_axis_off()
    bx.add_patch(Rectangle((0, 0), 1, 1, color="#ecebe6"))
    bx.add_patch(Rectangle((0, 0), t / SECONDS, 1, color=INK))

    # legend: 2 columns x 4 rows, with how many districts each sound defines
    n_sig = res.sig.astype(int).value_counts()
    for k, name in enumerate(B.SOUNDS):
        col, row = k // 4, k % 4
        x, y = 0.07 + col * 0.45, 0.225 - row * 0.042
        fig.patches.append(Rectangle((x, y - 0.012), 0.022, 0.0176, transform=fig.transFigure, color=CAT[k]))
        fig.text(x + 0.034, y - 0.004, f"{name}  ({n_sig.get(k, 0)})", fontsize=19, color=INK, va="center")
    fig.text(0.07, 0.268, "Signature sound  (districts)", fontsize=19, fontweight="semibold", color=INK, va="center")
    fig.text(0.07, 0.035, "Hover any neighborhood to hear it  ·  #30DayMapChallenge Day 11: Sound",
             fontsize=18, color=MUTED, va="center")
    fig.savefig(FRAMES / f"g{i:03d}.png", dpi=100)
    if i == int(N * (16 / 24)):    # 10 pm: busiest hour, used as the still
        fig.savefig(HERE / "out" / "day11_linkedin.png", dpi=100)
    plt.close(fig)


for i in range(N):
    frame(i)
ff = imageio_ffmpeg.get_ffmpeg_exe()
gif = HERE / "out" / "day11_linkedin.gif"
subprocess.run([ff, "-y", "-framerate", str(FPS), "-i", str(FRAMES / "g%03d.png"), "-vf",
                "scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=none",
                "-loop", "0", str(gif)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", gif, f"({gif.stat().st_size / 1e6:.1f} MB)")
