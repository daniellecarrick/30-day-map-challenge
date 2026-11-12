#!/bin/sh
# Day 11 inputs: every 2025 NYC 311 noise complaint (one row each) + community district boundaries.
# Rows are needed (not counts) so repeat callers can be collapsed: one address files 115k complaints a year.
set -e
cd "$(dirname "$0")/data"; mkdir -p raw
B="https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
W="created_date between '2025-01-01T00:00:00' and '2025-12-31T23:59:59' AND starts_with(complaint_type,'Noise')"
SEL="created_date,complaint_type,descriptor,community_board,incident_address,latitude,longitude"
for off in 0 200000 400000 600000 800000; do
  curl -sG "$B" --data-urlencode "\$select=$SEL" --data-urlencode "\$where=$W" \
    --data-urlencode "\$order=unique_key" --data-urlencode "\$limit=200000" --data-urlencode "\$offset=$off" \
    -o "noise_part_$off.csv" &
done
wait
{ head -1 noise_part_0.csv; for f in noise_part_*.csv; do tail -n +2 "$f"; done; } | gzip > raw/noise_2025.csv.gz
rm noise_part_*.csv
curl -sL -o cd.geojson "https://data.cityofnewyork.us/api/geospatial/5crt-au7u?method=export&format=GeoJSON"

# Sound clips (Wikimedia Commons; licenses in sounds.py). sounds.py trims and encodes them.
mkdir -p raw/audio
UA="30DayMapChallenge-build/1.0"
get() { curl -sL -A "$UA" -o "raw/audio/$2" "https://commons.wikimedia.org/wiki/Special:Redirect/file/$1"; sleep 2; }
get "A_dog_making_noises_and_barking.flac" dog.flac
get "Helicopter_flying_overhead.wav" heli.wav
get "Car_Horn.wav" horn.wav
get "442697-SBssa-Crowd_Talking_003.wav" talk.wav
