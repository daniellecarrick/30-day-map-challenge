# 30 Day Map Challenge 2026

One folder per day (`day03-polygons/`, `day04-clusters/`, …). Each day makes:
- `out/*.png`: static map for posting (Python: geopandas + matplotlib)
- `out/*.html`: interactive map, ONE self-contained file (D3), built by `build.py`

Prompts: https://30daymapchallenge.com/

## Setup
- Python env: `.venv/` at the repo root. Run a day with `../.venv/bin/python build.py` from its folder.
- New day: copy `_template/` to `dayNN-theme/` and edit `build.py` + `web/index.html`.
- Raw downloads go in `dayNN/data/`. Outputs go in `dayNN/out/`.

## Publishing (GitHub Pages)
Live site: https://daniellecarrick.github.io/30-day-map-challenge/ (served from `docs/` on `main`).
1. Build the day: `../.venv/bin/python build.py` in its folder.
2. Make sure the day has a `meta.json` (day, theme, title, description, html, png, published).
3. From the repo root: `.venv/bin/python publish.py`. This copies the page to `docs/dayNN/index.html` with link-preview tags, makes `preview.png` (1200×630) and `thumb.png`, and rebuilds the gallery `docs/index.html`.
4. Commit and push. The site updates in about a minute.
`publish.py` also writes:
- `docs/cover.png`: a 6×5 grid of all 30 days (thumbnails for finished days, empty tiles with the prompt for upcoming ones). Use it as the portfolio case-study hero image.
- `drafts/week-N.md`: a Markdown blog draft per week (days 1–7, 8–14, 15–21, 22–30), pre-filled from each day's `meta.json`, using live Pages URLs for images. It never overwrites an existing draft: if one is out of date, the fresh version goes to `week-N.new.md`. `drafts/` is gitignored.
- Prompts for all 30 days live in `shared/prompts.json`.
`day*/out/` is gitignored; `docs/` holds the published copies.

## Shared look (keep every day consistent)
Everything visual comes from `shared/`. Don't hard-code colors, fonts or control styles in a day's folder.
- `shared/tokens.json` is the source of truth: colors, scales, font, sizes. `shared/theme.css` mirrors it as CSS variables, so **change both together**.
- `shared/style.py`: `S.apply()` (Bitter/Raleway + colors for matplotlib), `S.make_scale`, `S.diverging_stops(center)`, `S.title`, `S.credit`, `S.panel_heading`, `S.callout` (leader-line annotation for a standout feature), `S.build_page` (inlines theme.css, components.js and data into one HTML file).
- `shared/components.js` (`window.MapKit`): `tooltip()`, `segmented()` toggle, `legend()` drawn inside the map SVG.
- Fonts: **Bitter** for headlines (titles, panel headings), **Raleway** for body text, labels, legends and controls. TTFs in `shared/fonts/` for PNGs; Google Fonts link for web. The Raleway TTFs were modified so their digits are lining (even-height) by default; Raleway's stock digits are old-style. If you re-download them, re-run `.venv/bin/pyftfeatfreeze -f lnum` on each file. Web pages get lining digits from `font-variant-numeric: lining-nums` in theme.css.

## Style rules
- **Minimal maps.** No basemap, no north arrow or scale bar unless they matter. Thin white edges between polygons, thin ink boundary lines.
- **Color by job.** Diverging (blue → neutral → amber → brown) for above/below a meaningful center (state the center, e.g. "11 avg"). Sequential (one hue, light → dark) for magnitude. Categorical in the fixed `tokens.json` order, never cycled.
- **Headline states the finding** ("On average, the nearest public bathroom is an 11-minute walk"), not the topic. Check that the number is literally true for the measure shown.
- **Interactive page:** fills the viewport with no page scroll on desktop; scrolls on phones. Legend goes inside the map's emptiest area. Alternate views use a segmented toggle, not side-by-side panels. Hover tooltip on every mark.
- **Footer/credit:** how it was calculated (one line) · data source with link · `#30DayMapChallenge Day N: Theme`.
- **Call out extremes** (best/worst, highest/lowest) on the PNG with `S.callout`: label in empty space, short leader line, feature outlined in ink.
- Closed / unavailable features: grey dots (`--closed`), excluded from the analysis.
- **Dark mode:** pages follow `prefers-color-scheme` via theme.css. PNGs are always light.
