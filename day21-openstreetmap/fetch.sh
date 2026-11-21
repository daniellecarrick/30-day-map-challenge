#!/bin/sh
# Day 21 · OpenStreetMap: download the New York State extract (~500 MB) from Geofabrik into data/raw/
# (gitignored), then run extract.py to boil it down to data/folk_tags.geojson.
# OSM data © OpenStreetMap contributors, ODbL. https://download.geofabrik.de/north-america/us/new-york.html
set -e
cd "$(dirname "$0")/data"
mkdir -p raw
curl -sSL -A "map-challenge-script/1.0" -o raw/new-york-latest.osm.pbf \
  "https://download.geofabrik.de/north-america/us/new-york/new-york-latest.osm.pbf"
