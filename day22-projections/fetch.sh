#!/bin/sh
# Day 22 inputs: county presidential results (2020, 2024), state populations, 20m boundaries.
set -e
cd "$(dirname "$0")/data"
R=https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/master
curl -sL -o votes2020.csv $R/2020_US_County_Level_Presidential_Results.csv
curl -sL -o votes2024.csv $R/2024_US_County_Level_Presidential_Results.csv
curl -sL -o pop2024.csv https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv
for f in cb_2020_us_county_20m cb_2020_us_state_20m cb_2023_us_county_20m; do
  curl -sL -o $f.zip https://www2.census.gov/geo/tiger/GENZ$(echo $f | cut -c4-7)/shp/$f.zip; done
