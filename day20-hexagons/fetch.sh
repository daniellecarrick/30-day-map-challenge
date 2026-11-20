#!/bin/sh
# Living street trees (tpstructure = Full) from NYC Parks Forestry Tree Points (hn5i-inap).
set -e
cd "$(dirname "$0")/data"
curl -s -o trees.csv "https://data.cityofnewyork.us/resource/hn5i-inap.csv?\$select=objectid,genusspecies,dbh,tpcondition,location&\$where=tpstructure='Full'&\$limit=2000000"
cp ../../day04-clusters/data/boroughs.geojson boroughs.geojson
wc -l trees.csv
