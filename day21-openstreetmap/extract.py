"""Day 21 · OpenStreetMap: pull the ten folk tags (folk.py) out of the New York State extract.

Reads data/raw/new-york-latest.osm.pbf (get it with ./fetch.sh), keeps features inside the five boroughs,
and writes data/folk_tags.geojson: one point per feature (ways become the mean of their vertices),
with its folk key, OSM type/id, edit date, and the tags a tooltip needs.

Run:  ../.venv/bin/python extract.py   (about 3 minutes)
"""
import json
from pathlib import Path

import osmium
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from folk import FOLK, SHOW

HERE = Path(__file__).parent
PBF = HERE / "data" / "raw" / "new-york-latest.osm.pbf"
OUT = HERE / "data" / "folk_tags.geojson"

boros = json.loads((HERE / "data" / "boroughs.geojson").read_text())["features"]
BORO = [(f["properties"]["boroname"], prep(shape(f["geometry"]))) for f in boros]
W, S, E, N = unary_union([shape(f["geometry"]) for f in boros]).bounds

feats = []


def keep(kind, obj, lon, lat):
    if not (W <= lon <= E and S <= lat <= N):
        return
    t = {k: v for k, v in obj.tags}
    hits = [f["key"] for f in FOLK if f["test"](t)]
    if not hits:
        return
    p = Point(lon, lat)
    boro = next((b for b, g in BORO if g.contains(p)), None)
    if boro is None:
        return
    for key in hits:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                      "properties": {"folk": key, "osm": f"{kind}/{obj.id}", "boro": boro,
                                     "edited": obj.timestamp.strftime("%Y-%m-%d"),
                                     **{k: t[k] for k in SHOW if k in t}}})


class Handler(osmium.SimpleHandler):
    def node(self, n):
        if len(n.tags) and n.location.valid():
            keep("node", n, n.location.lon, n.location.lat)

    def way(self, w):
        if not len(w.tags):
            return
        pts = [(nd.location.lon, nd.location.lat) for nd in w.nodes if nd.location.valid()]
        if pts:
            keep("way", w, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


Handler().apply_file(str(PBF), locations=True, idx="flex_mem")
OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False))
counts = {f["key"]: sum(x["properties"]["folk"] == f["key"] for x in feats) for f in FOLK}
print(f"wrote {OUT.name}: {len(feats)} features", counts)
