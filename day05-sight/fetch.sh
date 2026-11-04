#!/bin/sh
# Raw downloads for Day 5 (NYC Open Data, Socrata API). Run from day05-sight/.
set -e
B=https://data.cityofnewyork.us/resource
mkdir -p data/raw
# Street centerlines (CSCL, inkn-q76z): rw_type 1 = ordinary streets, all boroughs.
for off in 0 50000; do
  [ -s data/raw/streets_$off.geojson ] && continue
  curl -sf --retry 3 --max-time 300 -o data/raw/streets_$off.geojson -G "$B/inkn-q76z.geojson" \
    --data-urlencode "\$select=the_geom,physicalid,full_street_name,stname_label,boroughcode,streetwidth" \
    --data-urlencode "\$where=rw_type='1'" --data-urlencode "\$order=objectid" \
    --data-urlencode "\$limit=50000" --data-urlencode "\$offset=$off"
done
# Building footprints (5zhs-2jue): every building in the city (~1.08M), in pages of 50k.
for off in $(seq -f %.0f 0 50000 1050000); do
  [ -s data/raw/bld_$off.geojson ] && continue
  curl -sf --retry 3 --max-time 300 -o data/raw/bld_$off.geojson -G "$B/5zhs-2jue.geojson" \
    --data-urlencode "\$select=the_geom,bin,name,base_bbl,ground_elevation,height_roof" \
    --data-urlencode "\$order=objectid" --data-urlencode "\$limit=50000" --data-urlencode "\$offset=$off"
done
# Living trees (Forestry Tree Points, hn5i-inap): street trees plus inventoried park trees.
for off in 0 300000 600000; do
  [ -s data/raw/trees_$off.csv ] && continue
  curl -sf --retry 3 --max-time 600 -o data/raw/trees_$off.csv -G "$B/hn5i-inap.csv" \
    --data-urlencode "\$select=objectid,dbh,genusspecies,location" \
    --data-urlencode "\$where=tpstructure='Full'" --data-urlencode "\$order=objectid" \
    --data-urlencode "\$limit=300000" --data-urlencode "\$offset=$off"
done
# Park boundaries (NYC Parks Properties, enfh-gkve): where the surface model's canopy counts as woodland.
[ -s data/raw/parks.geojson ] || curl -sf --retry 3 --max-time 600 -o data/raw/parks.geojson -G "$B/enfh-gkve.geojson" \
  --data-urlencode "\$select=multipolygon,signname,typecategory" --data-urlencode "\$limit=50000"
# Copernicus GLO-30 DSM (surface model: includes buildings and terrain), for everything outside
# Manhattan, e.g. the New Jersey Palisades and the Jersey City skyline. Public AWS bucket.
for t in N40_00_W074_00 N40_00_W075_00 N41_00_W074_00 N41_00_W075_00; do
  [ -s data/raw/dsm_$t.tif ] && continue
  curl -sf --retry 3 --max-time 300 -o data/raw/dsm_$t.tif \
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_${t}_DEM/Copernicus_DSM_COG_10_${t}_DEM.tif"
done
