"""Turn data/parsed.csv into the Day 8 dataset: data/communities.csv.

Steps: drop rows that aren't US communities, merge rows that appear in two lists, classify each
community as religious or secular, parse founding and ending years, and place it on the map using
the coordinates of its Wikipedia article (or of the town / county it links to).

Ending year = when the community stopped as a communal experiment, as the source lists give it.
"Currently active" communities have no end year and are counted as still running in 2026.
Run: ../.venv/bin/python clean.py
"""
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
COORDS = DATA / "raw" / "coords.json"
NOW = 2026

# Main-list rows that aren't US communities.
DROP = {"Province of Carolina", "Province of Pennsylvania", "Province of Georgia", "Topolobampo Colony",
        "Equality Colony@1900"}  # the 1900 row repeats the 1897 Equality Colony
# Main-list rows that repeat a row in the Owenite or Fourierist list (that list's row is kept).
DUPES = {"New Harmony@1825", "Nashoba", "Brook Farm", "North American Phalanx", "Northampton Association",
         "Skaneateles Community", "Sodus Bay Phalanx", "Wisconsin Phalanx", "Clermont Phalanx",
         "Prairie Home Community", "Alphadelphia Association", "Spring Farm Colony", "Raritan Bay Union",
         "Reunion Colony", "Silkville"}
# The Fourierist list gives no end year for Hopedale; the main list's row (1842–1868) is kept instead.
DUPES_SPECIAL = {("fourierist", "Hopedale Community")}

# Main-list classification: name prefix -> (basis, movement). Owenite and Fourierist rows are
# secular; Shaker rows are religious. "mixed" marks a call that could go either way; the headline
# is checked with these counted both ways.
MAIN = {
    "Ephrata Cloister": ("religious", "Seventh Day Baptist"),
    "Harmony@1805": ("religious", "Harmony Society"),
    "New Harmony, Indiana": ("religious", "Harmony Society"),
    "Zoar": ("religious", "Separatist (Zoarite)"),
    "Old Economy Village": ("religious", "Harmony Society"),
    "United Order": ("religious", "Latter Day Saint"),
    "New Philadelphia Colony": ("religious", "Count de Leon's followers"),
    "Oberlin Colony": ("religious", "Christian"),
    "Community of United Christians": ("religious", "Methodist"),
    "Hopedale Community": ("religious", "Practical Christian"),
    "Fruitlands": ("secular", "Transcendentalist", "mixed"),
    "Fruit Hills": ("secular", "Owenite / anarchist"),
    "Kristeen Community": ("secular", "Universal Inquiry and Reform"),
    "Bishop Hill Colony": ("religious", "Swedish Pietist"),
    "Utopia": ("secular", "Individualist anarchist"),
    "Oneida Community": ("religious", "Perfectionist"),
    "Icarians": ("secular", "Icarian"),
    "Amana Colonies": ("religious", "Inspirationist"),
    "Modern Times": ("secular", "Individualist anarchist"),
    "Aurora Colony": ("religious", "Christian (Keil)"),
    "Free Lovers at Davis House": ("secular", "Free love / Spiritualist", "mixed"),
    "Octagon City": ("secular", "Vegetarian"),
    "Workingmen's Co-operative Colony": ("secular", "Chartist"),
    "Union Colony": ("secular", "Temperance colony"),
    "Progressive Colony": ("secular", "Communist", "mixed"),
    "Zion Valley": ("religious", "Bickertonite"),
    "Danish Socialist Colony": ("secular", "Socialist"),
    "Esperanza": ("secular", "Communist"),
    "Rugby": ("secular", "Cooperative colony"),
    "Am Olam": ("secular", "Jewish agrarian socialist", "mixed"),
    "Shalam Colony": ("religious", "Faithist"),
    "Kaweah Colony": ("secular", "Socialist"),
    "Ruskin Colony": ("secular", "Socialist"),
    "Glennis Cooperative": ("secular", "Bellamyist"),
    "Altruria": ("secular", "Christian socialist", "mixed"),
    "Fairhope Single Tax": ("secular", "Georgist (single tax)"),
    "Koreshan Unity": ("religious", "Koreshan"),
    "Harmony@1895": ("secular", "Socialist"),
    "Home Colony": ("secular", "Anarchist"),
    "Nucla": ("secular", "Socialist"),
    "Equality Colony": ("secular", "Bellamyist"),
    "Bellamy Cooperative Colony": ("secular", "Bellamyist"),
    "Freedom": ("secular", "Labor exchange"),
    "Arden Village": ("secular", "Georgist (single tax)"),
    "Zion, Illinois": ("religious", "Christian Catholic Church"),
    "Freeland Association": ("secular", "Socialist"),
    "Helicon Home Colony": ("secular", "Socialist"),
    "Post": ("secular", "Planned town"),
    "Free Acres": ("secular", "Georgist (single tax)"),
    "Llano del Rio": ("secular", "Socialist"),
    "New Llano": ("secular", "Socialist"),
    "Holy City": ("religious", "Riker's Holy City"),
    "Jersey Homesteads": ("secular", "New Deal cooperative"),
    "Druid Heights": ("secular", "Bohemian"),
    "Kerista Commune": ("religious", "Kerista", "mixed"),
    "Padanaram Settlement": ("religious", "Christian"),
    "Twin Oaks": ("secular", "Egalitarian commune"),
    "The Farm": ("religious", "Spiritual (Gaskin)", "mixed"),
    "East Wind Community": ("secular", "Egalitarian commune"),
    "Uranian Phalanstery": ("secular", "Artists' collective"),
    "Acorn Community Farm": ("secular", "Egalitarian commune"),
}
MOVEMENT = {"owenite": "Owenite", "fourierist": "Fourierist", "shaker": "Shaker"}

# Fixes where the source text has no plain year. Each gives the year the source text states.
YEARS = {
    "Amana Colonies": (1855, 1932),          # "1850s": moved from New York to Iowa in 1855
    "Kerista Commune": (1956, 1991),         # Old Tribe founded 1956
    "Koreshan Unity": (1894, 1982),          # "last new member admitted in 1940 (died 1982)"
    "Padanaram Settlement": (1966, 2001),    # "privatized soon after the death of the founder in 2001"
    "New Harmony@1825": (1825, 1827),
}

# Where a community's own article has no coordinates and its location link is too vague.
PLACE = {
    "Am Olam": "Douglas County, Oregon",
    "Amana Colonies": "Amana, Iowa",
    "Bellamy Cooperative Colony": "Lincoln County, Oregon",
    "Ephrata Cloister": "Ephrata, Pennsylvania",
    "Icarians": "Nauvoo, Illinois",
    "Koreshan Unity": "Estero, Florida",
    "Kerista Commune": "Haight-Ashbury",
    "New Philadelphia Colony": "Monaca, Pennsylvania",
    "Shalam Colony": "Doña Ana County, New Mexico",
    "Post": "Garza County, Texas",
    "Equality Colony": "Bow, Washington",
    "Ruskin Colony": "Dickson County, Tennessee",
    "United Order": "Independence, Missouri",
    "Uranian Phalanstery": "Lower East Side",
    "Hopedale Community": "Hopedale, Massachusetts",
    "Northampton Association": "Florence, Massachusetts",
    "North Union Shaker Village": "Shaker Heights, Ohio",
    "Watervliet Shaker Village (Ohio)": "Kettering, Ohio",
    "Fruit Hills": "Warren County, Ohio",
    "Kristeen Community": "Marshall County, Indiana",
    "Spring Farm Colony": "Sheboygan County, Wisconsin",
    "Free Lovers at Davis House": "Berlin Heights, Ohio",
    "Workingmen's Co-operative Colony": "Bourbon County, Kansas",
    "Progressive Colony": "Cedar Vale, Kansas",
    "Danish Socialist Colony": "Hays, Kansas",
    "Esperanza": "Neosho County, Kansas",
    "Glennis Cooperative": "Skagit County, Washington",
    "Gorham Shaker Village": "Gorham, Maine",
    "Narcoosee Shaker Village": "Narcoossee, Florida",
    "New Canaan Shaker Village": "New Canaan, Connecticut",
    "Philadelphia Shakers": "Philadelphia",
    "Savoy Shaker Village": "Savoy, Massachusetts",
    "White Oak Shaker Village": "White Oak, Georgia",
    "Prairie Home Community": "Logan County, Ohio",
    "Zanesfield Phalanx": "Zanesfield, Ohio",
    "Community of United Christians": "Berea, Ohio",
}

# Narcoossee is in the source table but its row doesn't parse cleanly; values from that row.
EXTRA = [{"source": "shaker", "name": "Narcoosee Shaker Village", "city": "Narcoossee", "state": "Florida",
          "dates": "1895–1924", "name_link": "", "location": "", "location_link": "", "city_link": "",
          "start": "", "end": "", "notes": ""}]


def key(r):
    return f'{r["name"]}@{year(r["start"])}'


def match(table, r):
    """Look up a row by 'Name@year' first, then by name prefix."""
    k = key(r)
    if k in table:
        return table[k]
    for name, v in table.items():
        if "@" not in name and r["name"].startswith(name):
            return v
    return None


def year(s):
    m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", s or "")
    return int(m.group(1)) if m else None


def note(r):
    """The source table's notes for a row, tidied. Shaker rows have no notes column, so they get
    the village's spiritual name from that table instead."""
    if r["source"] == "shaker":
        sp = r.get("spiritual", "").strip()
        return f"Shaker spiritual name: \u201c{sp}\u201d." if sp and sp.lower() != "none" else ""
    t = re.sub(r"\s*\|\}.*$", "", r["notes"]).strip()      # stray table-end markup
    t = re.sub(r"\s+([.,;])", r"\1", t)
    return t


def same(a, b):
    """Is dropped row `a` the same community as kept row `b`? Same article, or same leading words."""
    import unicodedata
    plain = lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    if a["name_link"] and plain(a["name_link"]) == plain(b["name_link"]) and year(a["start"]) == year(b["start"]):
        return True
    wa, wb = a["name"].lower().split()[:2], b["name"].lower().split()[:2]
    return wa == wb


def lookup_coords(titles):
    cache = json.loads(COORDS.read_text()) if COORDS.exists() else {}
    todo = sorted(t for t in set(titles) if t and t not in cache)
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        q = urllib.parse.urlencode({"action": "query", "prop": "coordinates", "titles": "|".join(batch),
                                    "redirects": 1, "format": "json", "colimit": "max"})
        req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q,
                                     headers={"User-Agent": "30DayMapChallenge-build/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=60))["query"]
        to = {t: t for t in batch}
        for step in ("normalized", "redirects"):
            for x in d.get(step, []):
                for t, cur in to.items():
                    if cur == x["from"]:
                        to[t] = x["to"]
        got = {p["title"]: p.get("coordinates") for p in d["pages"].values()}
        for t in batch:
            c = got.get(to[t])
            cache[t] = [c[0]["lat"], c[0]["lon"]] if c else None
    COORDS.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def main():
    rows = list(csv.DictReader(open(DATA / "parsed.csv"))) + EXTRA
    out, spare_notes = [], []   # notes from dropped duplicate rows, used when the kept row has none
    for r in rows:
        src = r["source"]
        if src == "main" and match({k: 1 for k in DROP}, r):
            continue
        if (src == "main" and match({k: 1 for k in DUPES}, r)) or (src, r["name"]) in DUPES_SPECIAL:
            spare_notes.append(r)
            continue
        if src == "main":
            cls = match(MAIN, r)
            if cls is None:
                raise SystemExit(f"unclassified main-list row: {r['name']}")
            basis, movement, *flag = cls
        else:
            basis = "religious" if src == "shaker" else "secular"
            movement, flag = MOVEMENT[src], []
            if src == "fourierist" and r["name"].startswith("Brook Farm"):
                flag = ["mixed"]  # Transcendentalist before it turned Fourierist
        if src == "shaker":
            a, _, b = r["dates"].partition("–")
            start, end_txt = year(a), b
        else:
            start, end_txt = year(r["start"]), r["end"]
        active = bool(re.search(r"present|currently active", end_txt, flags=re.I))
        end = None if active else year(end_txt)
        fix = match(YEARS, r)
        if fix:
            start, end = fix
        place = r["location"] or ", ".join(x for x in (r["city"], r["state"]) if x)
        out.append({
            "name": re.sub(r"\s*\(.*?\)\s*$", "", r["name"]).strip(),
            "basis": basis, "movement": movement, "mixed": int(bool(flag)),
            "start": start, "end": end, "active": int(active),
            "place": place, "source": src,
            "notes": note(r), "_row": r,
            "_titles": [match(PLACE, r), r["name_link"], r["city_link"], r["location_link"]],
        })

    for o in out:
        r = o.pop("_row")
        if not o["notes"]:
            twin = next((d for d in spare_notes if same(d, r)), None)
            o["notes"] = note(twin) if twin else ""

    cache = lookup_coords(t for o in out for t in o["_titles"])
    for o in out:
        titles = [t for t in o.pop("_titles") if t]
        hit = next((t for t in titles if cache.get(t)), None)
        o["lat"], o["lon"] = (cache[hit] if hit else (None, None))
        o["located_by"] = hit or ""
        known_end = NOW if o["active"] else o["end"]
        o["years"] = (known_end - o["start"]) if (o["start"] and known_end) else ""

    keys = ["name", "basis", "movement", "mixed", "start", "end", "active", "years", "place",
            "lat", "lon", "located_by", "source", "notes"]
    with open(DATA / "communities.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(out, key=lambda o: (o["start"] or 0, o["name"])))
    print(len(out), "communities ->", "data/communities.csv")
    for o in out:
        if o["lat"] is None:
            print("  NOT LOCATED:", o["name"], "|", o["place"])
        if o["years"] == "":
            print("  no lifespan:", o["name"], o["start"], o["end"])


if __name__ == "__main__":
    main()
