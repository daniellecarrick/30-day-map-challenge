"""Parse Wikipedia's list of phantom islands into data/phantoms.csv.

Inputs (in data/):
  phantom_islands.wikitext  raw wikitext of en.wikipedia "Phantom island"
  wikidata_phantoms.csv     Wikidata phantom islands with coordinates + enwiki link
Output:
  data/phantoms.csv         one row per island: name, article, found_year,
                            removed_year (+ evidence), lat, lon, coord_source, notes
Hand fixes belong in data/phantoms_fixes.csv (same columns, non-empty cells win).
"""
import csv
import re
import urllib.parse
from pathlib import Path

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- wikitext helpers
LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def strip_refs(s):
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    return s


def plain(s):
    """Wikitext -> readable text (links, templates, quotes removed)."""
    s = strip_refs(s)
    s = re.sub(r"\{\{nowrap\|(.*?)\}\}", r"\1", s)
    s = re.sub(r"\{\{sort\|[^|]*\|(.*?)\}\}", r"\1", s)
    s = re.sub(r"\{\{circa\|(.*?)\}\}", r"c. \1", s)
    s = re.sub(r"\{\{coord\|[^}]*\}\}", "", s)
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = LINK.sub(lambda m: m.group(2) or m.group(1), s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def rows(text):
    table = text.split('{| class="sortable wikitable"', 1)[1].split("\n|}", 1)[0]
    for chunk in table.split("\n|-")[1:]:
        lines = [l for l in chunk.strip("\n").split("\n") if l.strip()]
        if not lines:
            continue
        # Cells are either "| a || b || c" on one line or one "|cell" per line.
        joined = "\n".join(lines)
        cells = re.split(r"\|\||\n\|", "\n" + joined)
        cells = [c.lstrip("|").strip() for c in cells if c.strip() != ""]
        # Cells that are blank in the source get dropped above; pad.
        while len(cells) < 3:
            cells.append("")
        yield cells[0], cells[1], "\n".join(cells[2:])


# ---------------------------------------------------------------- field parsers
def parse_found(raw):
    m = re.search(r"\{\{sort\|(-?\d+)\|", raw)
    if m:
        key = int(m.group(1))
        if "century" in raw:
            # {{sort|1600|17th century}} -> 1600. Odd keys like {{sort|0714|...}}
            # aren't years: leave blank for phantoms_fixes.csv.
            return (key, "century") if key >= 1000 else (None, "")
        if "Middle Ages" in raw:
            return key, "century"
        return key, "exact" if "circa" not in raw else "circa"
    t = plain(raw)
    if not t or t.lower().startswith("unknown"):
        return None, ""
    m = re.search(r"(\d{1,4})\s*BC", t)
    if m:
        return -int(m.group(1)), "circa"
    m = re.search(r"(\d{1,2})(?:th|st|nd|rd) century", t)
    if m and not re.search(r"\b\d{3,4}\b", t):
        return (int(m.group(1)) - 1) * 100 + 50, "century"
    m = re.search(r"\b(\d{3,4})s?\b", t)
    if m:
        return int(m.group(1)), "circa" if ("c." in t or "s" in m.group(0)) else "exact"
    return None, ""


def dms(deg, mins, hemi):
    v = float(deg) + (float(mins) / 60 if mins else 0)
    return -v if hemi in "SW" else v


def parse_coord_template(raw):
    m = re.search(r"\{\{coord\|([^}]*)\}\}", raw)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split("|") if p.strip() and "=" not in p]
    nums, hemis = [], []
    for p in parts:
        if p in ("N", "S", "E", "W"):
            hemis.append((len(nums), p))
        else:
            try:
                nums.append(float(p))
            except ValueError:
                pass
    if len(hemis) < 2:  # decimal form {{coord|lat|lon}}
        return (nums[0], nums[1]) if len(nums) >= 2 else None

    def val(lo, hi, h):
        n = nums[lo:hi]
        v = sum(x / 60**i for i, x in enumerate(n))
        return -v if h in "SW" else v

    (i1, h1), (i2, h2) = hemis[0], hemis[1]
    return val(0, i1, h1), val(i1, i2, h2)


DEG = r"(\d{1,3}(?:\.\d+)?)\s*°\s*(?:(\d{1,2}(?:\.\d+)?)\s*[′'’])?\s*(?:\d+\s*[″\"])?\s*"
LATLON = re.compile(DEG + r"([NS])[,;]?\s*(?:and\s*)?" + DEG + r"([EW])")


def parse_coord_text(t):
    m = LATLON.search(t)
    if not m:
        return None
    return dms(m.group(1), m.group(2), m.group(3)), dms(m.group(4), m.group(5), m.group(6))


# "Noted as doubtful in 1875", "Could not be found in 1838", "Doubtful by 1817",
# "on maps until 2012", "removed from charts in 1875", "disproved in 1902"
GONE = re.compile(
    r"(doubtful|could not (?:be )?f(?:i|ou)nd|not found|removed|disproved|disproven|"
    r"until|vanished|deleted|expunged|searched for|no trace|failed to find|"
    r"proven not to exist|shown not to exist|non-?existen)[^.]{0,60}?\b(1[0-9]{3}|20[0-2][0-9])\b",
    re.I,
)


def parse_removed(t):
    m = GONE.search(t)
    if not m:
        return None, ""
    a = max(0, m.start() - 20)
    return int(m.group(2)), t[a : m.end() + 10].strip()


# ---------------------------------------------------------------- Wikidata coords
def wikidata_by_title():
    out = {}
    with open(DATA / "wikidata_phantoms.csv") as f:
        for r in csv.DictReader(f):
            if not r.get("article") or not r["c"].startswith("Point("):
                continue
            title = urllib.parse.unquote(r["article"].rsplit("/wiki/", 1)[1]).replace("_", " ")
            lon, lat = map(float, r["c"][6:-1].split())
            out[title.lower()] = (lat, lon, r["i"].rsplit("/", 1)[1])
    return out


# ---------------------------------------------------------------- article coords
ARTICLES = DATA / "articles.json"  # cache: {title: wikitext}


def article_texts(titles):
    """Fetch (once) the wikitext of each linked article; cached in data/articles.json."""
    import json
    import urllib.request

    cache = json.loads(ARTICLES.read_text()) if ARTICLES.exists() else {}
    todo = [t for t in titles if t and t not in cache]
    for i in range(0, len(todo), 40):
        batch = todo[i : i + 40]
        q = urllib.parse.urlencode({
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json", "formatversion": 2,
            "redirects": 1, "titles": "|".join(t.split("#")[0] for t in batch),
        })
        req = urllib.request.Request(
            "https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "map-research/0.1"})
        res = json.load(urllib.request.urlopen(req))["query"]
        alias = {}
        for k in ("normalized", "redirects"):
            for r in res.get(k, []):
                alias[r["to"]] = alias.get(r["from"], r["from"])
        for p in res["pages"]:
            txt = p.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
            src = alias.get(p["title"], p["title"])
            src = alias.get(src, src)
            for t in batch:
                if t.split("#")[0] in (src, p["title"]):
                    cache[t] = txt
        for t in batch:
            cache.setdefault(t, "")
    ARTICLES.write_text(json.dumps(cache))
    return cache


INFOBOX_LATLON = re.compile(
    r"\|\s*lat_?d(?:eg)?\s*=\s*(-?[\d.]+).*?(?:\|\s*lat_?m(?:in)?\s*=\s*([\d.]*))?.*?"
    r"\|\s*lat_?(?:NS|dir)\s*=\s*([NS]).*?\|\s*long?_?d(?:eg)?\s*=\s*(-?[\d.]+).*?"
    r"(?:\|\s*long?_?m(?:in)?\s*=\s*([\d.]*))?.*?\|\s*long?_?(?:EW|dir)\s*=\s*([EW])",
    re.S,
)


def coord_from_article(txt):
    if not txt:
        return None
    c = parse_coord_template(txt)
    if c:
        return c
    m = INFOBOX_LATLON.search(txt[:6000])
    if m:
        return dms(m.group(1), m.group(2), m.group(3)), dms(m.group(4), m.group(5), m.group(6))
    return parse_coord_text(plain(txt[:8000]))


# ---------------------------------------------------------------- main
COLS = ["name", "article", "found_year", "found_precision", "removed_year",
        "removed_evidence", "lat", "lon", "coord_source", "wikidata", "notes"]


def main():
    text = (DATA / "phantom_islands.wikitext").read_text()
    wd = wikidata_by_title()
    out = []
    for name_raw, date_raw, notes_raw in rows(text):
        link = LINK.search(name_raw)
        article = link.group(1).strip() if link else ""
        name = plain(name_raw)
        notes = plain(notes_raw)
        found, prec = parse_found(date_raw)
        removed, evidence = parse_removed(notes)
        lat = lon = None
        src = qid = ""
        c = parse_coord_template(notes_raw) or parse_coord_text(notes)
        if c:
            lat, lon, src = c[0], c[1], "wikipedia"
        if article and article.lower() in wd:
            wlat, wlon, qid = wd[article.lower()]
            if lat is None:
                lat, lon, src = wlat, wlon, "wikidata"
        out.append(dict(
            name=name, article=article, found_year=found if found is not None else "",
            found_precision=prec, removed_year=removed or "", removed_evidence=evidence,
            lat="" if lat is None else round(lat, 4), lon="" if lon is None else round(lon, 4),
            coord_source=src, wikidata=qid, notes=notes,
        ))

    texts = article_texts([r["article"] for r in out if r["lat"] == "" and r["article"]])
    for r in out:
        if r["lat"] == "" and r["article"]:
            c = coord_from_article(texts.get(r["article"], ""))
            if c and -90 <= c[0] <= 90 and -180 <= c[1] <= 180:
                r["lat"], r["lon"], r["coord_source"] = round(c[0], 4), round(c[1], 4), "article"

    fixes = DATA / "phantoms_fixes.csv"
    if fixes.exists():
        with open(fixes) as f:
            fx = {r["name"]: r for r in csv.DictReader(f)}
        for r in out:
            for k, v in fx.get(r["name"], {}).items():
                if v.strip():
                    r[k] = "" if v.strip() == "-" else v  # "-" clears a wrong value

    with open(DATA / "phantoms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)

    n = len(out)
    has = lambda k: sum(1 for r in out if r[k] != "")
    print(f"{n} islands")
    print(f"  found year:   {has('found_year')}")
    print(f"  removed year: {has('removed_year')}")
    print(f"  coordinates:  {has('lat')}  "
          f"(wikipedia {sum(r['coord_source']=='wikipedia' for r in out)}, "
          f"wikidata {sum(r['coord_source']=='wikidata' for r in out)})")
    both = sum(1 for r in out if r["lat"] != "" and r["found_year"] != "")
    print(f"  mappable with a found year: {both}")


if __name__ == "__main__":
    main()
