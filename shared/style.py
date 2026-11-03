"""Shared look for every #30DayMapChallenge map (PNG + web).

Use from a day's build.py:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
    import style as S

    S.apply()                                   # Bitter/Raleway + colors for matplotlib
    color = S.make_scale([(0, S.DIVERGING[0]), ...])   # minutes -> hex
    S.credit(fig, "Data: ... · #30DayMapChallenge Day N: Theme")
    S.build_page(HERE / "web/index.html", data, OUT / "map.html")
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from matplotlib import font_manager, rcParams
from matplotlib.colors import to_hex, to_rgb

SHARED = Path(__file__).resolve().parent
TOKENS = json.loads((SHARED / "tokens.json").read_text())

HEAD_FONT = TOKENS["font"]["headline"]   # Bitter: titles, panel headings
BODY_FONT = TOKENS["font"]["body"]       # Raleway: everything else
SIZE = TOKENS["font"]["size"]
C = TOKENS["color"]["light"]          # PNGs are always light
BG, INK, MUTED, EDGE, CLOSED = C["bg"], C["ink"], C["muted"], C["edge"], C["closed"]
DIVERGING = TOKENS["scales"]["diverging"]
SEQUENTIAL_BLUE = TOKENS["scales"]["sequential_blue"]
SEQUENTIAL_AMBER = TOKENS["scales"]["sequential_amber"]
CATEGORICAL = TOKENS["scales"]["categorical_light"]
MAP = TOKENS["map"]


def apply() -> None:
    """Register the series fonts and set matplotlib defaults so every PNG looks the same.
    Body text defaults to Raleway; S.title / S.panel_heading use Bitter."""
    for f in TOKENS["font"]["files"]:
        font_manager.fontManager.addfont(str(SHARED / f))
    rcParams.update({
        "font.family": BODY_FONT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "legend.frameon": False,
    })


def make_scale(stops: list[tuple[float, str]]):
    """Piecewise-linear color scale. stops = [(value, hex), ...] ascending.
    Values below the first stop / above the last are clamped."""
    def color(v: float) -> str:
        if v <= stops[0][0]:
            return stops[0][1]
        if v >= stops[-1][0]:
            return stops[-1][1]
        for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
            if v0 <= v <= v1:
                t = (v - v0) / (v1 - v0)
                return to_hex([a + (b - a) * t for a, b in zip(to_rgb(c0), to_rgb(c1))])
    return color


def diverging_stops(center: float, half_width: float | None = None) -> list[tuple[float, str]]:
    """The series' diverging scale centered on `center` (e.g. an average)."""
    hw = center if half_width is None else half_width
    lo = center - hw
    return [(lo + hw * k / 2, c) for k, c in enumerate(DIVERGING)]


def title(fig, text: str, y: float = 0.975) -> None:
    fig.suptitle(text, x=0.5, y=y, fontsize=SIZE["title"], fontweight="bold", color=INK, fontfamily=HEAD_FONT)


def credit(fig, text: str, y: float = 0.012) -> None:
    fig.text(0.5, y, text, ha="center", fontsize=SIZE["credit"], color=MUTED)


def panel_heading(ax, heading: str, sub: str | None = None) -> None:
    ax.text(0.02, 0.98, heading, transform=ax.transAxes, va="top",
            fontsize=SIZE["panel_title"], fontweight="semibold", color=INK, fontfamily=HEAD_FONT)
    if sub:
        ax.text(0.02, 0.935, sub, transform=ax.transAxes, va="top", fontsize=SIZE["subtitle"], color=MUTED)


def callout(ax, target_xy, text_xy, heading: str, detail: str | None = None,
            ha: str = "left", outline=None) -> None:
    """Annotate one feature: a thin leader line from `target_xy` (data coords) to a label at
    `text_xy` (axes fraction, 0–1). Heading in Raleway semibold, detail in muted.
    `outline`: optional geometry/GeoSeries to trace in ink so the feature stands out."""
    if outline is not None:
        from geopandas import GeoSeries
        gs = outline if isinstance(outline, GeoSeries) else GeoSeries([outline])
        gs.boundary.plot(ax=ax, color=INK, linewidth=1.1, zorder=6)
    ax.annotate(heading, xy=target_xy, xycoords="data", xytext=text_xy, textcoords="axes fraction",
                ha=ha, va="bottom", fontsize=SIZE["annotation"] + 0.5, fontweight="semibold", color=INK,
                zorder=7, arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=3, shrinkB=1,
                                          connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    if detail:
        ax.annotate(detail, xy=text_xy, xycoords="axes fraction", xytext=(0, -2), textcoords="offset points",
                    ha=ha, va="top", fontsize=SIZE["annotation"], color=MUTED, zorder=7)


def build_page(template: Path, data: dict, out: Path) -> int:
    """Write one self-contained HTML file: inlines shared/theme.css, shared/components.js
    and the page data (window.DATA). In the template, reference them as
      <link rel="stylesheet" href="../../shared/theme.css">
      <script src="../../shared/components.js"></script>
      <script src="data.js"></script>
    so the template also works when opened from disk during development."""
    page = Path(template).read_text()
    css = (SHARED / "theme.css").read_text()
    js = (SHARED / "components.js").read_text()
    blob = json.dumps(data, separators=(",", ":"))
    blob = re.sub(r"(\d{3,}\.\d)\d+", r"\1", blob)  # trim long coordinates (metres -> 0.1 m)
    page = re.sub(r'<link rel="stylesheet" href="[./]*shared/theme\.css">', lambda _: f"<style>\n{css}</style>", page)
    page = re.sub(r'<script src="[./]*shared/components\.js"></script>', lambda _: f"<script>\n{js}</script>", page)
    page = page.replace('<script src="data.js"></script>', f"<script>window.DATA = {blob};</script>")
    Path(out).write_text(page)
    return len(page)
