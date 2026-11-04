# Day 23 · Senses: Taste — working notes

**Question:** where do restaurants of each cuisine cluster? Does Chinatown have the most Chinese food? Is Cobble Hill a coffee mecca?

## Data
- NYC DOHMH Restaurant Inspection Results (`43nn-pn8j`): every inspected food place has a `cuisine_description` (~90 categories), lat/lon and a 2010 NTA code. `fetch.sh` pulls one row per restaurant (CAMIS) with its latest inspection date.
- 2020 NTAs (`9nt8-h7nd`), joined spatially (the dataset's own NTA field uses 2010 codes).
- Cleaning: drop records with no coords and anything not inspected since 2023-01-01 (likely closed). 31,422 → **26,646** restaurants.
- Scripts: `experiments/explore.py data` (counts + share by NTA, writes `data/_joined.csv`), then `experiments/radius.py data/_joined.csv` (500 m neighborhood around every restaurant).

## Early findings (Nov 2026 pull)
**Chinese (8.4% of NYC restaurants):**
- By count, Chinatown is #2: Flushing 259, **Chinatown–Two Bridges 113**, Sunset Park (Central) 106.
- By share, Chinatown is #4: Flushing 52%, Sunset Park Central 46%, Sunset Park East 46%, **Chinatown 36%**.
- Within 500 m, Sunset Park blocks hit 67% Chinese, the highest anywhere.
- Headline candidate: *"Manhattan's Chinatown isn't NYC's Chinese-food capital: Flushing has more than twice as many Chinese restaurants."*

**Coffee/Tea (8.1%; ~17% are chains such as Starbucks and Pret):**
- Cobble Hill is lumped into "Carroll Gardens–Cobble Hill–Gowanus–Red Hook": 37 of 284 (13%), so above average but not the top.
- A 500 m circle on Court St / Atlantic: 20 of 145 places are coffee (**14%, about 1.7× the city average**).
- Highest coffee share (500 m, ≥60 places): Morningside Heights 27%, Williamsburg 24%, Greenwich Village 21%. Most by count: SoHo / Chinatown / LES (~70 within 500 m).
- So Cobble Hill is coffee-rich but not the mecca. Williamsburg and the Village beat it.

**Other strong clusters worth a panel:** Korean (Murray Hill, Flushing 46%; K-town), Caribbean (East Flatbush–Rugby 54%), Jewish/Kosher (Midwood 51%, Borough Park 48%), Latin American (Corona / North Corona ~33%), Mexican (Port Richmond SI 33%).

## Caveats
- Cuisine is self-reported at permit time. "American" (17%) and "Coffee/Tea" are catch-alls, and the categories mix food (Chinese) with formats (Pizza, Donuts).
- 2020 NTAs are big and lump neighborhoods (Cobble Hill). Use a 500 m kernel or hexes for fine clusters, not NTA polygons.
- Count favours Midtown for everything (it's dense). **Share / location quotient** (local share ÷ city share) is the honest measure of a cluster. Hide low-n areas (e.g. <60 places within the radius).

## Map idea
- **Measure:** location quotient per cuisine on a ~400 m hex grid (or a 500 m kernel), sequential palette, shown only where n ≥ threshold. Grey hexes = too few restaurants.
- **Interactive:** segmented toggle for 6–8 cuisines (Chinese, Coffee, Korean, Caribbean, Kosher, Latin American, Italian, Mexican). Each view has its own finding-style headline and a callout on the peak. Restaurant dots on hover.
- **PNG:** small multiples (2×3 or 2×4) of the same city, one cuisine per panel, each panel's peak labelled with `S.callout`. Title: "Every cuisine has a capital, and it's rarely where you'd guess."
- Alternative: one "dominant over-represented cuisine per hex" categorical map (the city's flavor map). Busy but striking; needs the categorical order from tokens.json and ≤8 groups.

## Status (2026-11-03)
- `build.py` writes `out/nyc_cuisine_capitals.png` and `out/nyc_cuisine_capitals.html`.
- **Page:** every cuisine with 100+ restaurants except Coffee/Tea and "Other" (37). A dropdown (`MapKit.dropdown`, new in shared/) groups them by how clustered they are. A side panel shows the share of NYC restaurants, the most concentrated spot, the namesake neighborhood, 3 runners-up in other neighborhoods, a "how clustered" meter, and a context note from `context.json` (edit there).
- **PNG:** 8 panels: Chinese, Italian, Korean, Eastern European, Caribbean, Kosher, Latin American, Mexican.
- **Measure:** 400 m hexes, each scored on the restaurants within 500 m. Uncolored when under 1.5× or fewer than 30 restaurants. A peak needs 40+ restaurants nearby. Namesakes are scored at their best hex within 500 m of the landmark corner. "Clustered" = share of a cuisine's restaurants whose other neighbors within 500 m are 3×+ the city share for it.
- **Hex names:** the neighborhood most of a hex's nearby restaurants are in (keeps park and stadium names off busy blocks).
- **Data quirks called out in the context notes:** American peaks at Yankee Stadium concessions; Donuts at the Staten Island Ferry Dunkin' stands.
- No `meta.json` yet, on purpose: publish.py would put the day live. Add it on Nov 23.
- Known issue: in dark mode the darkest blue (10×+) is low-contrast on the dark background.
