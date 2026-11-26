"""Download the raw data for Day 26 (Water) into data/.

1. NYC DEP Reservoir Operations (NYC Open Data, zkky-n5j3): daily storage, elevation and
   diversions for the six Catskill/Delaware reservoirs, 1982 onward -> data/reservoir_ops.csv
2. OpenStreetMap: aqueduct/tunnel pipelines and the system's reservoirs -> data/osm_water.geojson

Run once:  ../.venv/bin/python data_prep.py   (re-run to refresh)
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "map-research-script/1.0"}


def get(url, data=None):
    req = urllib.request.Request(url, data=data, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


# ---- 1. reservoir operations (skipped if already downloaded; pass --refresh to redo) ----
import sys
REFRESH = "--refresh" in sys.argv
BASE = "https://data.cityofnewyork.us/resource/zkky-n5j3.json"
rows, off, page = [], 0, 50000
while REFRESH or not (DATA / "reservoir_ops.csv").exists():
    q = urllib.parse.urlencode({
        "$where": "parameter in ('Storage_derived','Elevation','Diversion')",
        "$order": "datetime,reservoir,parameter", "$limit": page, "$offset": off})
    chunk = json.loads(get(f"{BASE}?{q}"))
    rows += chunk
    print(f"  reservoir ops: {len(rows):,} rows")
    if len(chunk) < page:
        break
    off += page
if rows:
  df = pd.DataFrame(rows)
  df["date"] = df.datetime.str[:10]
  df["value"] = pd.to_numeric(df.value, errors="coerce")
  df[["date", "reservoir", "parameter", "value", "units"]].to_csv(DATA / "reservoir_ops.csv", index=False)

# ---- 2. OpenStreetMap: aqueducts, tunnels, reservoirs ----
RES = ("Ashokan|Rondout|Neversink|Pepacton|Cannonsville|Schoharie|Kensico|Hillview|"
       "New Croton|West Branch|Croton Falls|Cross River|Boyd|Middle Branch|East Branch|"
       "Titicus|Amawalk|Muscoot|Bog Brook|Diverting|Lake Gilead|Lake Gleneida|Kirk Lake|Jerome Park")
QUERY = f"""[out:json][timeout:180];
(
  way["man_made"="pipeline"]["name"~"Aqueduct|Tunnel|Water Tunnel",i](40.4,-75.6,42.6,-73.4);
  way["name"~"(Delaware|Catskill|Croton|Shandaken|Neversink|East Delaware|West Delaware).*(Aqueduct|Tunnel)",i](40.4,-75.6,42.6,-73.4);
  wr["natural"="water"]["name"~"^(Upper |Lower )?({RES})( Reservoir)?( /.*)?$"](40.8,-75.6,42.6,-73.4);
);
out geom tags;"""
osm = json.loads(get("https://overpass-api.de/api/interpreter",
                     urllib.parse.urlencode({"data": QUERY}).encode()))
def is_res(t):
    return t.get("natural") == "water" or t.get("landuse") == "reservoir" or t.get("water") == "reservoir"


if not osm.get("elements"):
    sys.exit(f"Overpass returned nothing (remark: {osm.get('remark')}); try again later")
feats = []
for e in osm["elements"]:
    t = e.get("tags", {})
    if e["type"] == "way" and "geometry" in e:
        coords = [[p["lon"], p["lat"]] for p in e["geometry"]]
        closed = coords[0] == coords[-1] and is_res(t)
        geom = {"type": "Polygon", "coordinates": [coords]} if closed else {"type": "LineString", "coordinates": coords}
    elif e["type"] == "relation":
        outers = [[[p["lon"], p["lat"]] for p in m["geometry"]] for m in e.get("members", [])
                  if m.get("role") == "outer" and "geometry" in m]
        if not outers:
            continue
        geom = {"type": "MultiLineString", "coordinates": outers}  # stitched into polygons in build.py
    else:
        continue
    feats.append({"type": "Feature", "geometry": geom,
                  "properties": {"osm": f"{e['type']}/{e['id']}", "name": t.get("name"),
                                 "kind": "reservoir" if is_res(t) else ("trail" if "highway" in t else "conduit"),
                                 "tunnel": t.get("tunnel") or t.get("location")}})
# ---- 3. big reservoirs mapped as multipolygon relations: fetch from the OSM API ----
# (Overpass often times out on these; IDs looked up by name on Nominatim, Nov 2026)
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, mapping
from shapely.ops import polygonize, unary_union

RELATIONS = {2499379: "Kensico Reservoir", 3289973: "Lower Ashokan Reservoir",
             6281964: "Upper Ashokan Reservoir", 4057387: "Pepacton Reservoir",
             19704108: "Cannonsville Reservoir", 8548269: "Schoharie Reservoir",
             12603325: "New Croton Reservoir"}
have = {f["properties"]["osm"] for f in feats}
for rid, name in RELATIONS.items():
    if f"relation/{rid}" in have:
        continue
    root = ET.fromstring(get(f"https://api.openstreetmap.org/api/0.6/relation/{rid}/full"))
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    ways = {w.get("id"): [nodes[nd.get("ref")] for nd in w.iter("nd")] for w in root.iter("way")}
    rel = root.find("relation")
    lines = [LineString(ways[m.get("ref")]) for m in rel.iter("member")
             if m.get("type") == "way" and m.get("role") == "outer"]
    inner = [LineString(ways[m.get("ref")]) for m in rel.iter("member")
             if m.get("type") == "way" and m.get("role") == "inner"]
    shape = unary_union(list(polygonize(unary_union(lines))))
    if inner:
        shape = shape.difference(unary_union(list(polygonize(unary_union(inner)))))
    feats.append({"type": "Feature", "geometry": mapping(shape),
                  "properties": {"osm": f"relation/{rid}", "name": name, "kind": "reservoir", "tunnel": None}})
    print(f"  {name}: {shape.area * 1e4:.1f} (deg² ×1e4)")
    time.sleep(1)

(DATA / "osm_water.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
print(f"  osm: {len(feats)} features")
