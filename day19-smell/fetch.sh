#!/bin/sh
# Day 19 downloads (run from day19-smell/). States: copy of day18-null/data/cb_2020_us_state_500k.zip.
set -e
mkdir -p data/rasters
# Yearly modeled lilac first-bloom maps, 1981–2025 (USA-NPN Extended Spring Index, PRISM, 4 km)
seq 1981 2025 | xargs -P 6 -I{} sh -c 'curl -s --max-time 110 -o data/rasters/lilac_bloom_{}.tif \
  "https://geoserver.usanpn.org/geoserver/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=si-x__lilac_bloom_prism&format=image/geotiff&subset=time(%22{}-01-01T07:00:00.000Z%22)"'
# Nature's Notebook: first open flowers per plant per year, common lilac (36) + cloned lilac (35)
curl -s --max-time 280 -o data/npn_lilac_individual.json \
  "https://services.usanpn.org/npn_portal/observations/getSummarizedData.json?request_src=mapchallenge&start_date=2009-01-01&end_date=2025-12-31&species_id%5B0%5D=35&species_id%5B1%5D=36&phenophase_id%5B0%5D=205&phenophase_id%5B1%5D=501"
