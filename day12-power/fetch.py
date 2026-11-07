"""Download the raw inputs for Day 12 (Power): Moses's built and unbuilt expressways.

Writes into data/:
  osm_built.json      OpenStreetMap motorway ways for the built Moses expressways
  osm_guides.json     named streets, rail lines and bridges used to trace the unbuilt routes
  raw/footprints.geojson  NYC building footprints near the unbuilt routes (needs data/unbuilt_routes.geojson,
                      made by routes.py; run this script again after routes.py)
  raw/blocks/ny.zip   2020 Census TIGER blocks for New York State (POP20, HOUSING20), ~330 MB, gitignored

Run:  ../.venv/bin/python fetch.py
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from pathlib import Path

DATA = Path(__file__).parent / "data"
UA = "30daymap-build"
BBOX = "40.49,-74.27,40.92,-73.68"  # NYC

BUILT = ["Cross Bronx Expressway", "Brooklyn-Queens Expressway", "Gowanus Expressway", "Prospect Expressway",
         "Major Deegan Expressway", "Bruckner Expressway", "Sheridan Boulevard", "Sheridan Expressway",
         "Long Island Expressway", "Van Wyck Expressway", "Clearview Expressway", "Staten Island Expressway"]

GUIDE_STREETS = ["Canal Street", "West Street", "Washington Street", "Greenwich Street", "Hudson Street", "Varick Street",
                 "Watts Street", "Broome Street", "Avenue of the Americas", "Centre Street", "Bowery", "Chrystie Street",
                 "Forsyth Street", "Delancey Street", "Suffolk Street", "Clinton Street", "Elizabeth Street",
                 "West 30th Street", "East 30th Street", "12th Avenue", "11th Avenue", "8th Avenue", "5th Avenue",
                 "Lexington Avenue", "1st Avenue", "FDR Drive",
                 "Broadway", "Flushing Avenue", "Wyckoff Avenue", "Force Tube Avenue", "Atlantic Avenue",
                 "North Conduit Boulevard", "South Conduit Boulevard", "North Conduit Avenue", "South Conduit Avenue",
                 "Cross Bay Boulevard", "Jamaica Avenue", "Highland Boulevard", "Cooper Avenue", "Cypress Avenue",
                 "Myrtle Avenue", "Metropolitan Avenue", "Driggs Avenue", "Bedford Avenue", "Havemeyer Street",
                 "7th Avenue", "65th Street", "Fort Hamilton Parkway", "Flatlands Avenue", "Schenck Avenue",
                 "Fountain Avenue", "Van Sinderen Avenue", "Linden Boulevard", "Ralph Avenue", "Pennsylvania Avenue",
                 "Nostrand Avenue", "Ocean Parkway", "Glenwood Road", "Albany Avenue", "Belt Parkway"]


def curl_json(url: str, data: str | None = None) -> dict:
    """curl rather than urllib: this machine's Python SSL can't handshake with some of these servers."""
    cmd = ["curl", "-sS", "--fail", "-m", "300", "-A", UA]
    if data is not None:
        cmd += ["--data-urlencode", f"data={data}"]
    for wait in (0, 10, 20, 40, 80, 160):
        time.sleep(wait)
        r = subprocess.run(cmd + [url], capture_output=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
        print("  retrying after error", r.returncode, flush=True)
    raise SystemExit(f"could not fetch {url}")


def overpass(q: str) -> dict:
    return curl_json("https://overpass-api.de/api/interpreter", q)


def main():
    names = "|".join(BUILT)
    built = overpass(f'[out:json][timeout:200];way["highway"~"motorway"]["name"~"^({names})$"]({BBOX});out geom tags;')
    (DATA / "osm_built.json").write_text(json.dumps(built))
    print("built ways:", len(built["elements"]))

    # in chunks (one big query drops the connection); `out geom` keeps node ids so streets can be intersected
    bbox = "40.56,-74.05,40.80,-73.76"
    queries = [f'way["highway"]["name"~"^({"|".join(GUIDE_STREETS[i:i + 12])})$"]({bbox});'
               for i in range(0, len(GUIDE_STREETS), 12)]
    queries.append(f'(way["railway"]["name"~"Bay Ridge"]({bbox}); way["name"~"^(Manhattan Bridge|Williamsburg Bridge)$"]({bbox}); '
                   f'way["name"~"^(Cemetery of the Evergreens|Highland Park)$"]({bbox}););')
    guides = {"elements": []}
    for q in queries:
        guides["elements"] += overpass(f"[out:json][timeout:200];{q}out geom;")["elements"]
    (DATA / "osm_guides.json").write_text(json.dumps(guides))
    print("guide ways:", len(guides["elements"]))

    blocks = DATA / "raw" / "blocks" / "ny.zip"
    if not blocks.exists():
        blocks.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "-o", str(blocks),
                        "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip"], check=True)

    routes = DATA / "unbuilt_routes.geojson"
    if routes.exists():
        fetch_footprints(routes)


def fetch_footprints(routes: Path) -> None:
    """Building footprints (NYC Open Data 5zhs-2jue) inside a 300 m box around each unbuilt route.
    The analysis buffer is much narrower; this just keeps the download small."""
    import geopandas as gpd
    r = gpd.read_file(routes).to_crs("EPSG:2263")
    feats = []
    for _, row in r.iterrows():
        poly = gpd.GeoSeries([row.geometry.buffer(1000)], crs="EPSG:2263").to_crs(4326).iloc[0].simplify(0.0003)
        wkt = poly.wkt
        offset = 0
        while True:
            q = urllib.parse.urlencode({"$where": f"within_polygon(the_geom, '{wkt}')", "$limit": 50000,
                                        "$offset": offset, "$order": "bin"})
            page = curl_json(f"https://data.cityofnewyork.us/resource/5zhs-2jue.geojson?{q}")["features"]
            feats += page
            print(row["name"], offset, len(page), flush=True)
            if len(page) < 50000:
                break
            offset += 50000
    seen, uniq = set(), []
    for f in feats:
        b = f["properties"].get("bin"), f["properties"].get("doitt_id")
        if b not in seen:
            seen.add(b)
            uniq.append(f)
    (DATA / "raw" / "footprints.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": uniq}))
    print("footprints:", len(uniq))


if __name__ == "__main__":
    main()
