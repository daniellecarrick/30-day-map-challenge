"""Count what stands today in the path of each unbuilt Moses expressway.

Path = a corridor around each traced route in data/unbuilt_routes.geojson, as wide as that plan's
documented right-of-way (width_m in routes.py): LOMEX 100 m (250-350 ft), Mid-Manhattan 30 m (100 ft),
Cross-Brooklyn 37 m (121 ft). The Bushwick's width is not documented; it uses the Cross-Brooklyn's.

  buildings  NYC building footprints (feature_code 2100 = building) that overlap the corridor by > 5 m²
  people     people living in those buildings: each building's homes (MapPLUTO residential units on its
             tax lot, split among the lot's buildings by floor volume) x its census tract's people per
             home (2020 POP20 / HOUSING20, by tract). Area-weighting blocks overcounts here, because these paths
             run along streets, medians and rail lines where nobody lives.

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


def residents(fp, blocks):
    """Add `people` to each footprint: its share of its lot's homes x its census tract's people per home.
    (Tract, not block: single blocks can disagree with PLUTO on how many homes they hold.)"""
    units = json.loads((DATA / "pluto_units.json").read_text())
    fp = fp.copy()
    fp["bbl"] = gpd.pd.to_numeric(fp.mappluto_bbl, errors="coerce").fillna(0).astype("int64").astype(str)
    vol = fp.area * fp.height_roof.fillna(10).clip(lower=3)  # floor volume; low or unknown heights count as one storey
    fp["homes"] = fp.bbl.map(units).fillna(0) * vol / vol.groupby(fp.bbl).transform("sum")
    pts = gpd.GeoDataFrame(geometry=fp.representative_point(), crs=fp.crs)
    tract = blocks.assign(tract=blocks.GEOID20.str[:11]).groupby("tract")[["POP20", "HOUSING20"]].sum()
    per_home = (tract.POP20 / tract.HOUSING20).where(tract.HOUSING20 > 0)
    blk = gpd.sjoin(pts, blocks[["GEOID20", "geometry"]], how="left", predicate="within")
    blk = blk[~blk.index.duplicated()]
    ratio = blk.GEOID20.str[:11].map(per_home)
    fallback = blocks.POP20.sum() / blocks.HOUSING20.sum()  # citywide, for a building outside any tract with homes
    fp["people"] = fp.homes * ratio.fillna(fallback)
    return fp


def main():
    routes = gpd.read_file(DATA / "unbuilt_routes.geojson").to_crs(CRS_M)
    corr = routes.copy()
    corr["geometry"] = routes.buffer(routes.width_m / 2)

    fp = gpd.read_file(DATA / "raw" / "footprints.geojson").to_crs(CRS_M)
    fp = fp[fp.feature_code.astype(str) == "2100"].copy()
    fp["construction_year"] = gpd.pd.to_numeric(fp.construction_year, errors="coerce")
    fp["height_roof"] = gpd.pd.to_numeric(fp.height_roof, errors="coerce")
    blocks = gpd.read_file(DATA / "raw" / "blocks" / "nyc_blocks.gpkg").to_crs(CRS_M)
    fp = residents(fp, blocks)

    stats, hits = [], []
    for _, r in corr.iterrows():
        c = r.geometry
        cand = fp[fp.intersects(c)].copy()
        cand["overlap"] = cand.intersection(c).area
        b = cand[cand.overlap > MIN_OVERLAP].copy()
        b["route"] = r["name"]
        hits.append(b)
        yrs = b.construction_year
        stats.append({
            "name": r["name"],
            "buildings": int(len(b)),
            "buildings_pre1971": int((yrs < PLAN_DEAD).sum()),
            "buildings_post1971": int((yrs >= PLAN_DEAD).sum()),
            "people": int(round(b.people.sum())),
            "length_mi": r["length_mi"],
            "width_m": int(r["width_m"]),
        })
    allb = gpd.pd.concat(hits)
    # a building near where two routes meet counts once in the total
    uniq = allb.drop_duplicates("bin")
    total = {
        "buildings": int(len(uniq)),
        "buildings_pre1971": int((uniq.construction_year < PLAN_DEAD).sum()),
        "people": int(round(uniq.people.sum())),
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
              f"({s['buildings_pre1971']} built before 1971)  {s['people']:>6} people")
    print("TOTAL", total)


if __name__ == "__main__":
    main()
