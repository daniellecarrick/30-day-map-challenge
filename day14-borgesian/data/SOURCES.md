# Day 14 data sources

Goal: one world map of places that were on maps but not in the world, in 4 categories,
with a slider on the year each first appeared on maps (removal year in the tooltip, when known).
Atlantis gets a callout as the extreme (most locations claimed for one place).

## 1. Phantom islands (backbone, ~120 rows)
- `phantom_islands.wikitext`: raw wikitext of Wikipedia "Phantom island" (its list section, ~123 rows).
  Columns: Name | Date of alleged discovery | Notes. Coordinates and the year it was disproved
  are only in the notes text, so they need parsing plus hand-fixing.
- `wikidata_phantoms.csv`: Wikidata items that are instances of phantom island (Q129264),
  phantom settlement (Q7180622) or trap street (Q1506107). About half have coordinates (P625),
  none have dates. Use it to fill in coordinates and to link to the en.wikipedia article.

## 2. Atlantis (callout)
- `atlantis_hypotheses.wikitext`: Wikipedia "Hypothetical locations of Atlantis". About 25 proposed
  sites with who proposed them and when. No coordinates, so geocode by hand (centroid of the
  named sea or region).

## 3. Mapmaking mistakes (hand-curated, ~12)
Wikipedia "Category:Cartographic errors" (Category:Phantom geographical features):
Mountains of Kong (1798–1889), Island of California, Terra Australis, Sea of the West,
Lake Chimay, Lake Uniamési, Dragon's Tail, Peary Channel, Anson Archipelago, Sandy Island.
Plus Strait of Anián, Lake Parime, Rio San Buenaventura, Mountains of the Moon.
Get dates from each article (first map, last map / disproved).

## 4. Hoaxes and paper towns (hand-curated, ~10)
Wikipedia "Fictitious entry" and "Phantom settlement": Agloe NY (1930s), Beatosu and Goblu
(1978 Michigan state map), Argleton (Google Maps), Mount Richard (1970s), Ripton,
Listenbourg, Llandegley International Airport. Plus Poyais (MacGregor, 1820s).

## Mythical lands (optional 5th group, or fold into phantoms)
El Dorado, Lemuria, Mu, Shangri-La, Garden of Eden, Camelot, Thule, Hy-Brasil, Antillia.
Several already appear in the phantom-island list.

## Base map
Natural Earth land/coastlines (already used on day 13).

## Dates, if Wikipedia is thin
David Rumsey Map Collection (davidrumsey.com) search gives the earliest and latest maps
that show a feature. Use it to check, not as the main source.

Downloaded 2026-11-13. Wikipedia text is CC BY-SA, so credit it in the footer.

## Parsed so far (parse_phantoms.py -> data/phantoms.csv)
Run: `../.venv/bin/python parse_phantoms.py`. Coordinates come from, in order: the Wikipedia
list notes, Wikidata, the island's own article (cached in data/articles.json), then
data/phantoms_fixes.csv (hand fixes; "-" clears a wrong value). Hand-placed points say
"hand-approx" in coord_source and are where old maps drew them, roughly.
- 122 islands, 104 with coordinates (79 also with a found year)
- 92 with a found year, but only 25 with a removed/doubted year, 16 with both
  -> DECIDED: slider uses the first-appeared year only. Removed/doubted year goes in the tooltip when known.
- Still unplaced (18): Candyn, Isle Grande, Great Ireland, Groclant, Juan de Lisboa, Kantia,
  Kianida, María de Lajara, Matsyn, Mokupāpapa, Petermannland, Rocabarraigh, Royllo,
  Sefton Reef, Torca, Tuanaki, Walker Island, Yosemite Rock.

## Other categories (hand-built from Wikipedia article leads, 2026-11-13)
- `lost_lands.csv` (10): Atlantis, Penglai, Eden, Camelot, Saguenay, Quivira, El Dorado, Lemuria, Mu, Shangri-La.
- `mistakes.csv` (12): Mountains of the Moon, Dragon's Tail, Lake Chimay, Strait of Anián, Lake Parime,
  Island of California, Terra Australis, Sea of the West, Buenaventura River, Mountains of Kong,
  Lake Uniamési, Peary Channel.
- `hoaxes.csv` (8): Poyais, Agloe, Mount Richard, Beatosu and Goblu, Ripton, Llandegley airport,
  Argleton, Listenbourg.
- `atlantis.csv` (27): proposed Atlantis sites with proposer and year (1552–2021), for the callout.
Dates first filled from memory, then checked against Wikipedia on 2026-11-15:
- Confirmed: Terra Australis gone 1775 (Cook's 2nd voyage 1772–75), Strait of Anián 1562 (Gastaldi),
  Sea of the West 1752 (Buache) to 1790 (Meares, last original depiction), Buenaventura 1776
  (Domínguez–Escalante) to 1844 (Frémont 1843–44), Mu 1864 (Brasseur), Lemuria 1864 (Sclater),
  Great Britain 1946 (Comyns Beaumont, Riddle of Prehistoric Britain), North Sea 1953 (Spanuth,
  Das enträtselte Atlantis; de.wikipedia), Helike 1985 (Giovannini).
- Corrected: Island of California removal 1747 -> 1776 (article: settled by Anza 1774–76; the 1747
  decree isn't in it); Lake Chimay 1550–1790 -> 1554 (Ramusio map) to 1751 (last new map); Dragon's
  Tail 1489 -> c. 833 (al-Khwārizmī); Camelot 1180 -> c. 1175 (Chrétien, 1170s); El Dorado 1530 ->
  1539 (earliest written reference); Thera 1909 -> 1913 (Frost's cited paper); Cyprus 2003 -> 2006
  (Sarmast's cited book); Mount Richard 1971 -> c. 1970 ("early 1970s").
- Cleared (no source gives one): Strait of Anián removal year, Mount Richard removal year,
  Richat Structure year (only undated "fringe claims").
Most coordinates in these files are hand-approx (see coord_note).

## Combined (combine_places.py -> data/places.csv)
131 mapped places: 101 phantom, 10 lost, 12 mistake, 8 hoax. Atlantis, Island of California and
Penglai were moved out of the phantom list into their better category.
