#!/bin/sh
# Day 28 raw downloads. Run from day28-feeling/.
set -e
cd data/raw
# CDC PLACES 2025 release (BRFSS 2023), census tracts, NYC's five counties (36005 Bronx, 36047 Kings, 36061 New York, 36081 Queens, 36085 Richmond)
curl -s -o places_nyc.json "https://data.cdc.gov/resource/cwsq-ngmh.json?\$select=locationname,countyname,measureid,data_value,low_confidence_limit,high_confidence_limit,totalpopulation&\$where=stateabbr='NY'%20AND%20countyfips%20in('36005','36047','36061','36081','36085')%20AND%20measureid%20in('LONELINESS','EMOTIONSPT','MHLTH','DEPRESSION')&\$limit=50000"
curl -s -o tracts_ny.zip https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_36_tract_500k.zip
curl -s -o states.zip https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip
curl -s -o gnis.zip https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/DomesticNames/DomesticNames_National_Text.zip
curl -s -o nyc_tracts_nta.json 'https://data.cityofnewyork.us/resource/63ge-mke6.json?$select=geoid,ntaname,boroname&$limit=5000'  # NYC DCP 2020 tracts -> neighborhood (NTA) names
