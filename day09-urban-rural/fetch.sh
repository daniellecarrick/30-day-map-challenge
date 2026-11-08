#!/bin/sh
# Day 9 inputs. The Census API now needs a key, so stream the bulk ACS
# summary-file tables and keep only the NY county rows (GEO_ID 0500000US36…).
set -e
cd "$(dirname "$0")/data"
mkdir -p acs
BASE=https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData
for t in b01003 b25003 b08201 b19001 b08301 b05002 b25024 c24030 b21001 b08303; do
  curl -s "$BASE/acsdt5y2023-$t.dat" | grep -E "^GEO_ID|^0500000US36" > acs/$t.dat &
done
wait
# votes2024_nysboe.csv: certified results from https://results.elections.ny.gov/contest/5591 (copied by hand; the site blocks scripts)
curl -sL -o usrc2020_group_detail.xlsx https://www.usreligioncensus.org/sites/default/files/2023-06/2020_USRC_Group_Detail.xlsx
curl -sL -o counties.zip https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip
