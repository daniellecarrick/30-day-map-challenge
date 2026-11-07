"""Count what stands today in the path of each unbuilt Moses expressway.

Path = a corridor around each traced route in data/unbuilt_routes.geojson, as wide as that plan's
documented right-of-way (width_m in routes.py): LOMEX 100 m (250-350 ft), Mid-Manhattan 30 m (100 ft),
Cross-Brooklyn 37 m (121 ft). The Bushwick's width is not documented; it uses the Cross-Brooklyn's.

  buildings  NYC building footprints (feature_code 2100 = building) that overlap the corridor by > 5 m²
  people     2020 Census block population, area-weighted by the share of each block inside the corridor
  homes      2020 Census housing units, same weighting

Writes data/path_stats.json, data/path_buildings.geojson (the buildings, for drawing) and
data/context_lomex.geojson (other buildings around LOMEX, grey context for the PNG zoom).
Run after routes.py and fetch.py:  ../.venv/bin/python analyze.py
"""
import json
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).parent
DATA = HERE / "data"
CRS_M = "EPSG:32618"
MIN_OVERLAP = 5  # m²; ignores footprints that only graze the corridor edge
PLAN_DEAD = 1971  # all four were killed by 1971


def main():
    routes = gpd.read_file(DATA / "unbuilt_routes.geojson").to_crs(CRS_M)
    corr = routes.copy()
    corr["geometry"] = routes.buffer(routes.width_m / 2)

    fp = gpd.read_file(DATA / "raw" / "footprints.geojson").to_crs(CRS_M)
    fp = fp[fp.feature_code.astype(str) == "2100"].copy()
    fp["construction_year"] = gpd.pd.to_numeric(fp.construction_year, errors="coerce")
    fp["height_roof"] = gpd.pd.to_numeric(fp.height_roof, errors="coerce")
    blocks = gpd.read_file(DATA / "raw" / "blocks" / "nyc_blocks.gpkg").to_crs(CRS_M)
    blocks["barea"] = blocks.area

    stats, hits = [], []
    for _, r in corr.iterrows():
        c = r.geometry
        cand = fp[fp.intersects(c)].copy()
        cand["overlap"] = cand.intersection(c).area
        b = cand[cand.overlap > MIN_OVERLAP].copy()
        b["route"] = r["name"]
        hits.append(b)
        blk = blocks[blocks.intersects(c)].copy()
        w = blk.intersection(c).area / blk.barea
        yrs = b.construction_year
        stats.append({
            "name": r["name"],
            "buildings": int(len(b)),
            "buildings_pre1971": int((yrs < PLAN_DEAD).sum()),
            "buildings_post1971": int((yrs >= PLAN_DEAD).sum()),
            "people": int(round((blk.POP20 * w).sum())),
            "homes": int(round((blk.HOUSING20 * w).sum())),
            "length_mi": r["length_mi"],
            "width_m": int(r["width_m"]),
        })
    allb = gpd.pd.concat(hits)
    # a building near where two routes meet counts once in the total
    uniq = allb.drop_duplicates("bin")
    union = corr.union_all()
    blk = blocks[blocks.intersects(union)].copy()
    w = blk.intersection(union).area / blk.barea
    total = {
        "buildings": int(len(uniq)),
        "buildings_pre1971": int((uniq.construction_year < PLAN_DEAD).sum()),
        "people": int(round((blk.POP20 * w).sum())),
        "homes": int(round((blk.HOUSING20 * w).sum())),
    }
    (DATA / "path_stats.json").write_text(json.dumps({"routes": stats, "total": total}, indent=2))
    out = allb[["bin", "route", "construction_year", "height_roof", "geometry"]].copy()
    out.to_crs(4326).to_file(DATA / "path_buildings.geojson", driver="GeoJSON")
    # grey context for the PNG's lower Manhattan zoom: the other buildings around LOMEX
    lomex = corr[corr.name == "Lower Manhattan Expressway"].geometry.iloc[0]
    ctx = fp[fp.within(lomex.buffer(220)) & ~fp.bin.isin(out.bin)][["geometry"]].copy()
    ctx["geometry"] = ctx.geometry.simplify(1)
    ctx.to_crs(4326).to_file(DATA / "context_lomex.geojson", driver="GeoJSON")

    for s in stats:
        print(f"{s['name']:<28} {s['length_mi']:>4} mi  {s['buildings']:>5} bldgs "
              f"({s['buildings_pre1971']} built before 1971)  {s['people']:>6} people  {s['homes']:>6} homes")
    print("TOTAL", total)


if __name__ == "__main__":
    main()
