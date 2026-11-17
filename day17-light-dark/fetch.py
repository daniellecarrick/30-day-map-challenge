"""Day 17 · download lights and lighthouse names. Cached; re-run to fill in anything that failed.

NOAA ENC Direct, every chart scale (general → harbour): Light_point (rhythms) and named
Landmark_point (lighthouse structures). OpenStreetMap: named man_made=lighthouse.
The NOAA server answers "service not found" now and then, so every page is retried.

Run: ../.venv/bin/python fetch.py
"""
import json
import subprocess
import time
from pathlib import Path

RAW = Path(__file__).parent / "data" / "noaa"
RAW.mkdir(parents=True, exist_ok=True)
ENC = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect"
LIGHTS = {"general": "enc_general/MapServer/10", "coastal": "enc_coastal/MapServer/10",
          "approach": "enc_approach/MapServer/13", "harbour": "enc_harbour/MapServer/11"}
MARKS = {"general": "enc_general/MapServer/21", "coastal": "enc_coastal/MapServer/24",
         "approach": "enc_approach/MapServer/28", "harbour": "enc_harbour/MapServer/26"}
LIGHT_FIELDS = "OBJECTID,CATLIT,COLOUR,LITCHR,SIGGRP,SIGPER,SIGSEQ,VALNMR,OBJNAM,HEIGHT"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


def get(url):
    for wait in (0, 2, 5, 10, 20, 40, 60):
        time.sleep(wait)
        r = subprocess.run(["curl", "-sS", "-m", "180", url], capture_output=True)
        try:
            d = json.loads(r.stdout or b"{}")
        except json.JSONDecodeError:
            continue
        if "features" in d:
            return d
    raise RuntimeError(f"gave up: {url[:120]}")


def esri(path, where, fields, out):
    f = RAW / out
    if f.exists():
        print("have", out)
        return
    feats, off = [], 0
    while True:
        d = get(f"{ENC}/{path}/query?where={where}&outFields={fields}&f=json&outSR=4326"
                f"&resultOffset={off}&resultRecordCount=1000&orderByFields=OBJECTID")
        feats += d["features"]
        off += len(d["features"])
        if not d.get("exceededTransferLimit") or not d["features"]:
            break
    f.write_text(json.dumps(feats))
    print(out, len(feats), flush=True)


def osm():
    f = RAW / "osm_lighthouses.json"
    if f.exists():
        print("have osm")
        return
    # US + territories, in a few boxes so each query stays small
    boxes = ["24,-98,50,-66", "24,-125,50,-98", "50,-180,72,-129", "18,-161,23,-154", "17,-68,19,-64"]
    els = []
    for b in boxes:
        q = f'[out:json][timeout:180];nwr["man_made"="lighthouse"]({b});out center tags;'
        for k in range(6):
            r = subprocess.run(["curl", "-sS", "--fail", "-m", "240", "-A", "30DayMapChallenge/1.0",
                                "--data-urlencode", f"data={q}", OVERPASS[k % 3]], capture_output=True)
            if r.returncode == 0:
                els += json.loads(r.stdout)["elements"]
                break
            time.sleep(10 * (k + 1))
        else:
            raise RuntimeError(f"OSM box {b} failed")
    f.write_text(json.dumps(els))
    print("osm", len(els))


jobs = [(p, "OBJNAM+IS+NOT+NULL", "OBJNAM", f"landmarks_{k}.json") for k, p in MARKS.items()]
# approach-scale lights reject a SIGSEQ filter, so ask for everything there (build.py skips rows without a rhythm)
jobs += [(p, "1%3D1" if k == "approach" else "SIGSEQ+IS+NOT+NULL", LIGHT_FIELDS if k != "approach" else "*",
          f"lights_{k}.json") for k, p in LIGHTS.items()]
# fixed lights (LITCHR 1: steady, never blinking) have no SIGSEQ, so fetch them separately
jobs += [(LIGHTS[k], "LITCHR%3D1", LIGHT_FIELDS, f"lights_{k}_fixed.json") for k in ("coastal", "harbour")]
failed = []
for job in jobs:
    try:
        esri(*job)
    except RuntimeError as e:      # one stubborn layer shouldn't stop the rest; re-run to retry it
        print("FAILED", job[3], e, flush=True)
        failed.append(job[3])
try:
    osm()
except RuntimeError as e:
    print("FAILED osm", e)
    failed.append("osm")
print("still missing:", failed or "nothing")
