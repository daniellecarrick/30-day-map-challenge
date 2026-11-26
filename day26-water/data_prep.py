"""Download the raw data for Day 26 (Water) into data/.

1. OpenStreetMap: Midtown Manhattan buildings with their 3D parts -> data/midtown_osm.json
   OSM maps the big towers as several `building:part` pieces, each with its own `height` and
   `min_height`, so their setbacks are kept (a plain footprint x roof height would overstate the
   Empire State Building's volume almost fourfold).
2. NYC Open Data (ia2d-e54m): the city's water use per year, 1979 onward -> data/consumption.json

Files already downloaded are skipped; pass --refresh to download again.
Run:  ../.venv/bin/python data_prep.py
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
REFRESH = "--refresh" in sys.argv
UA = {"User-Agent": "map-research-script/1.0"}
BBOX = "40.7440,-73.9960,40.7680,-73.9650"   # 33rd to 59th St, 8th Ave to 2nd Ave (roughly)


def get(url, data=None):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=300) as r:
        return r.read()


# ---- 1. Midtown buildings ----
osm_file = DATA / "midtown_osm.json"
if REFRESH or not osm_file.exists():
    query = f"""[out:json][timeout:240];
(
  way["building"]({BBOX});
  way["building:part"]({BBOX});
  relation["building"]({BBOX});
  relation["building:part"]({BBOX});
);
out geom tags;"""
    osm = json.loads(get("https://overpass-api.de/api/interpreter", urllib.parse.urlencode({"data": query}).encode()))
    if not osm.get("elements"):
        sys.exit(f"Overpass returned nothing (remark: {osm.get('remark')}); try again later")
    osm_file.write_text(json.dumps(osm))
    print(f"  {len(osm['elements']):,} OSM elements -> {osm_file.name}")

# ---- 2. yearly water use ----
use_file = DATA / "consumption.json"
if REFRESH or not use_file.exists():
    rows = json.loads(get("https://data.cityofnewyork.us/resource/ia2d-e54m.json?$limit=500"))
    use_file.write_text(json.dumps(rows))
    print(f"  {len(rows)} years of water use -> {use_file.name}")
