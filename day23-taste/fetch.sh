#!/bin/sh
# Day 23 · Taste: download raw data into data/
#  - restaurants.csv: one row per restaurant (CAMIS) with cuisine + coords + last inspection date,
#    from NYC DOHMH Restaurant Inspection Results (https://data.cityofnewyork.us/d/43nn-pn8j)
#  - nta2020.geojson: 2020 Neighborhood Tabulation Areas (https://data.cityofnewyork.us/d/9nt8-h7nd)
set -e
cd "$(dirname "$0")/data"
curl -sS -o restaurants.csv "https://data.cityofnewyork.us/resource/43nn-pn8j.csv?\$select=camis,dba,boro,zipcode,cuisine_description,latitude,longitude,nta,max(inspection_date)%20as%20last_insp&\$group=camis,dba,boro,zipcode,cuisine_description,latitude,longitude,nta&\$limit=100000"
curl -sS -o nta2020.geojson "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?\$limit=500"
