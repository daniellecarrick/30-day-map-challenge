"""Fetch one freely licensed photo per tree species from Wikipedia / Wikimedia Commons.

For each species that is the most common tree in some hexagon (data/needed_taxa.json, written by
build.py; cultivars already folded into their species), looks up the
Wikipedia article's lead image, keeps it only if it is on Commons under a free license, and saves
a small JPEG plus its credit. Writes data/photos/<slug>.jpg and data/photos.json
({taxon: {file, title, artist, license, url}}). Re-running skips taxa already in photos.json.

Run:  ../.venv/bin/python fetch_photos.py
"""
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
DATA = HERE / "data"
PHOTOS = DATA / "photos"
PHOTOS.mkdir(exist_ok=True)
UA = {"User-Agent": "30DayMapChallenge-build/1.0 (static map; python urllib)"}
FETCH_W = 500  # a standard Wikimedia thumbnail width (others get rate-limited)
WIDTH = 360  # px; shown at 180 CSS px in the tooltip (2x for sharp screens)
FREE = re.compile(r"^(CC0|CC BY|CC-BY|Public domain|PD|GFDL|Attribution)", re.I)
# a few names Wikipedia files under another title
ALIASES = {"Platanus x acerifolia": "Platanus × hispanica", "Tilia x europaea": "Tilia × europaea",
           "Acer x freemanii": "Acer × freemanii", "Prunus x yedoensis": "Prunus × yedoensis",
           "Gleditsia triacanthos var. inermis": "Gleditsia triacanthos", "Aesculus sp.": "Aesculus",
           "Acer ginnala": "Acer tataricum subsp. ginnala", "Platanus x acerfolia": "Platanus × hispanica",
           "Morus": "Morus alba", "Acer": "Maple"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def lookup(name: str):
    title = ALIASES.get(name, name)
    s = json.loads(get("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"))))
    src = (s.get("originalimage") or {}).get("source")
    if not src:
        return None
    parts = src.split("?")[0].split("/")
    fname = urllib.parse.unquote(parts[-2] if "thumb" in parts else parts[-1])  # .../thumb/a/ab/NAME/960px-NAME
    q = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url|extmetadata"
         f"&iiurlwidth={FETCH_W}&titles=" + urllib.parse.quote("File:" + fname))
    pages = json.loads(get(q))["query"]["pages"]
    page = next(iter(pages.values()))
    if "imageinfo" not in page:  # not on Commons (local, possibly non-free file)
        return None
    ii = page["imageinfo"][0]
    meta = ii.get("extmetadata", {})
    lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
    if not FREE.match(lic):
        return None
    img = Image.open(io.BytesIO(get(ii["thumburl"]))).convert("RGB")
    img.thumbnail((WIDTH, WIDTH))
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    img.save(PHOTOS / f"{slug}.jpg", quality=72, optimize=True, progressive=True)
    artist = strip_html(meta.get("Artist", {}).get("value")) or "Unknown"
    return {"file": f"{slug}.jpg", "article": s.get("title"), "artist": artist[:80], "license": lic,
            "url": ii["descriptionurl"]}


def main():
    names = [n for n in json.loads((DATA / "needed_taxa.json").read_text()) if n.lower() != "unknown"]
    out_f = DATA / "photos.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    for n in names:
        if n in out:
            continue
        try:
            out[n] = lookup(n)
        except urllib.error.HTTPError as e:
            if e.code == 404:          # no article under this name: remember that
                out[n] = None
            else:                      # rate limit or server hiccup: try again next run
                print("  failed", n, e)
                time.sleep(30)
                continue
        print(("ok   " if out[n] else "none ") + n)
        out_f.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        time.sleep(3)
    print(sum(1 for n in names if out.get(n)), "of", len(names), "species have a photo")


if __name__ == "__main__":
    main()
