# Day 2 experiments

The published Day 2 page is the particle wind map: `../web/windmap.html`, built to
`../out/citibike_windmap.html` and published via `../meta.json`.

Earlier interactive versions:

- `citibike_three_windows.html` (snapshot): the first published Day 2 page (Nov 3, 2026). A toggle between
  three 3-hour windows: weekday 7–10am, weekday 4–7pm, weekend 1–4pm.
- `citibike_allday_3hour.html` (snapshot): the first all-day animation. Each frame was a rolling 3-hour window,
  which read as confusing.
- Hourly arrows: `../web/index.html`, still built by `build.py` to `../out/citibike_flow.html` (not published).
  One arrow per hexagon, one hour at a time, with play/pause and a timeline. Its headline numbers feed the wind map.

The snapshots are self-contained (open them in a browser) and not rebuilt; their source is in git history.
