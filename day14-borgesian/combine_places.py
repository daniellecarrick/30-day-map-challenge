"""Combine the category files into data/places.csv (one row per mapped place).

category: phantom (phantoms.csv), lost (lost_lands.csv), mistake (mistakes.csv),
hoax (hoaxes.csv). atlantis.csv stays separate: it is the Atlantis callout layer.
Rows without coordinates are dropped; rows without a year are kept (year blank),
so the page can decide how to show them on the slider.
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"
COLS = ["category", "name", "article", "year", "year_precision",
        "removed_year", "removed_note", "lat", "lon", "coord_note", "note"]

# In the phantom list but better told in another category.
MOVED = {"Atlantis", "Island of California", "Mount Penglai"}


def main():
    ph = pd.read_csv(DATA / "phantoms.csv")
    ph = ph[~ph.name.isin(MOVED)]
    ph = pd.DataFrame({
        "category": "phantom", "name": ph.name, "article": ph.article,
        "year": ph.found_year, "year_precision": ph.found_precision,
        "removed_year": ph.removed_year, "removed_note": ph.removed_evidence,
        "lat": ph.lat, "lon": ph.lon, "coord_note": ph.coord_source, "note": ph.notes,
    })
    parts = [ph]
    for cat, f in [("lost", "lost_lands.csv"), ("mistake", "mistakes.csv"), ("hoax", "hoaxes.csv")]:
        d = pd.read_csv(DATA / f).rename(columns={"first_year": "year", "first_precision": "year_precision"})
        d.insert(0, "category", cat)
        parts.append(d[COLS])
    all_ = pd.concat(parts, ignore_index=True)
    out = all_.dropna(subset=["lat", "lon"]).copy()
    for c in ("year", "removed_year"):
        out[c] = out[c].astype("Int64")
    out.to_csv(DATA / "places.csv", index=False)

    at = pd.read_csv(DATA / "atlantis.csv")
    print(f"{len(out)} places mapped ({len(all_) - len(out)} dropped, no coordinates)")
    print(out.groupby("category").agg(n=("name", "size"), with_year=("year", "count"),
                                      with_removed=("removed_year", "count")).to_string())
    print(f"years {out.year.min()} to {out.year.max()}")
    print(f"Atlantis: {len(at)} proposed sites, {at.year.min()}–{at.year.max()}")


if __name__ == "__main__":
    main()
