#!/bin/sh
# Downloads for Day 27 (run from anywhere): 2020 Census blocks and ACS 2020-24 block-group tables.
set -e
cd "$(dirname "$0")/data"
# 2020 Census blocks: POP20, HOUSING20, internal points and shapes (~330 MB; gitignored)
curl -sf -o tabblock20.zip https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_36_tabblock20.zip
# ACS 2020-24 5-year: total population (B01001) and household income (B19001), NYC block groups only
for t in b01001 b19001; do
  curl -sf https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-$t.dat \
    | grep -E '^GEO_ID|^1500000US36(005|047|061|081|085)' > acs_$t.dat
done
