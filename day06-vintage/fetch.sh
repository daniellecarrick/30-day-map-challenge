#!/bin/sh
# Raw downloads for Day 6 (NYC Open Data, Socrata API). Run from day06-vintage/.
set -e
B=https://data.cityofnewyork.us/resource
mkdir -p data/raw
# Building Footprints (5zhs-2jue): attributes for every structure, then shapes for buildings only.
curl -sf --retry 3 --max-time 600 -o data/raw/buildings.csv -G "$B/5zhs-2jue.csv" \
  --data-urlencode "\$select=bin,doitt_id,base_bbl,mappluto_bbl,construction_year,feature_code,height_roof,shape_area,last_status_type,name" \
  --data-urlencode "\$limit=1200000"
curl -sf --retry 3 --max-time 600 -o data/raw/buildings_geom.csv -G "$B/5zhs-2jue.csv" \
  --data-urlencode "\$select=doitt_id,the_geom" --data-urlencode "\$where=feature_code=2100" \
  --data-urlencode "\$limit=1000000"
# 2020 Neighborhood Tabulation Areas (9nt8-h7nd).
curl -sf --retry 3 --max-time 120 -o data/raw/nta2020.geojson "$B/9nt8-h7nd.geojson?\$limit=500"
