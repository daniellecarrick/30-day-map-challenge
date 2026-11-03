"""Publish finished days to docs/ (served by GitHub Pages) and rebuild the gallery.

For every dayNN-*/meta.json it:
  - copies the interactive page to docs/dayNN/index.html, adding link-preview tags
    (Open Graph + Twitter card) and a small "All maps" link
  - copies the PNG to docs/dayNN/map.png and makes preview.png (1200x630, for social cards)
    and thumb.png (for the gallery)
Then writes docs/index.html, the gallery home page.

Run from the repo root after building a day:   .venv/bin/python publish.py
Then commit and push docs/ to update the live site.
"""
import json
import shutil
from html import escape
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
BASE_URL = "https://daniellecarrick.github.io/30-day-map-challenge"
REPO_URL = "https://github.com/daniellecarrick/30-day-map-challenge"
SITE_TITLE = "30 Day Map Challenge 2026"
SITE_BLURB = "One map a day through November, made for the #30DayMapChallenge."
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Bitter:wght@600;700'
         '&family=Raleway:wght@400;600;700&display=swap" rel="stylesheet">')


def meta_tags(title: str, description: str, url: str, image: str) -> str:
    t, d = escape(title, quote=True), escape(description, quote=True)
    return "\n".join([
        f'<meta name="description" content="{d}">',
        f'<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{t}">',
        f'<meta property="og:description" content="{d}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{image}">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        '<meta name="twitter:card" content="summary_large_image">',
        f'<meta name="twitter:title" content="{t}">',
        f'<meta name="twitter:description" content="{d}">',
        f'<meta name="twitter:image" content="{image}">',
    ])


def social_image(src: Path, dest: Path, size=(1200, 630)) -> None:
    """Fit the PNG into a 1200x630 card on a white background (the size link previews expect)."""
    im = Image.open(src).convert("RGB")
    im.thumbnail(size, Image.LANCZOS)
    card = Image.new("RGB", size, "white")
    card.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    card.save(dest, optimize=True)


def thumbnail(src: Path, dest: Path, width=800) -> None:
    im = Image.open(src).convert("RGB")
    im.thumbnail((width, width), Image.LANCZOS)
    im.save(dest, optimize=True)


HOME_LINK = '<a class="home-link" href="../">&larr; All maps</a>'  # styled in shared/theme.css


def publish_day(folder: Path) -> dict:
    m = json.loads((folder / "meta.json").read_text())
    slug = f"day{m['day']:02d}"
    out = DOCS / slug
    out.mkdir(parents=True, exist_ok=True)
    url, image = f"{BASE_URL}/{slug}/", f"{BASE_URL}/{slug}/preview.png"

    page = (folder / m["html"]).read_text()
    head_extra = meta_tags(m["title"], m["description"], url, image)
    page = page.replace("</head>", f"{head_extra}\n</head>", 1)
    page = page.replace("<body>", f"<body>\n{HOME_LINK}", 1)
    (out / "index.html").write_text(page)

    png = folder / m["png"]
    shutil.copyfile(png, out / "map.png")
    social_image(png, out / "preview.png")
    thumbnail(png, out / "thumb.png")
    print(f"published {slug}: {url}")
    return {**m, "slug": slug}


def gallery(days: list[dict]) -> None:
    cards = []
    for m in sorted(days, key=lambda d: d["day"]):
        cards.append(f"""
    <article class="card">
      <a class="thumb" href="{m['slug']}/"><img src="{m['slug']}/thumb.png" alt="{escape(m['title'], quote=True)}" loading="lazy"></a>
      <div class="body">
        <p class="kicker">Day {m['day']} · {escape(m['theme'])}</p>
        <h2><a href="{m['slug']}/">{escape(m['title'])}</a></h2>
        <p class="desc">{escape(m['description'])}</p>
        <p class="links"><a href="{m['slug']}/">Interactive map</a><a href="{m['slug']}/map.png">PNG</a></p>
      </div>
    </article>""")
    latest = max(days, key=lambda d: d["day"])
    css = (ROOT / "shared" / "theme.css").read_text()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{SITE_TITLE}</title>
{meta_tags(SITE_TITLE, SITE_BLURB, BASE_URL + "/", f"{BASE_URL}/{latest['slug']}/preview.png")}
{FONTS}
<style>
{css}
.gallery {{ max-width: 1100px; margin: 0 auto; padding: 40px 16px 32px; }}
.gallery > header {{ text-align: center; margin-bottom: 28px; }}
.gallery h1 {{ font-family: var(--font-head); font-size: clamp(26px, 4vw, 36px); line-height: 1.15; margin: 0 0 8px; }}
.gallery .lede {{ margin-bottom: 4px; }}
.progress {{ color: var(--muted); font-size: 13px; margin: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: 20px; }}
.card {{ border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent); border-radius: 10px; overflow: hidden;
         background: var(--bg); display: flex; flex-direction: column; }}
.card .thumb {{ display: block; background: #fff; aspect-ratio: 5 / 3; }}
.card .thumb img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.card .body {{ padding: 14px 16px 16px; }}
.kicker {{ margin: 0 0 4px; font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.card h2 {{ font-family: var(--font-head); font-size: 18px; line-height: 1.25; margin: 0 0 6px; }}
.card h2 a {{ color: var(--ink); text-decoration: none; }}
.card h2 a:hover {{ text-decoration: underline; }}
.desc {{ margin: 0 0 10px; color: var(--muted); font-size: 13.5px; }}
.links {{ margin: 0; display: flex; gap: 14px; font-size: 13.5px; font-weight: 600; }}
.links a {{ color: var(--accent); text-decoration: none; }}
.links a:hover {{ text-decoration: underline; }}
.gallery > footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 28px; }}
.gallery > footer a {{ color: inherit; }}
</style>
</head>
<body>
<main class="gallery">
  <header>
    <h1>{SITE_TITLE}</h1>
    <p class="lede">{SITE_BLURB}</p>
    <p class="progress">{len(days)} of 30 days</p>
  </header>
  <section class="grid">{''.join(cards)}
  </section>
  <footer>Prompts from <a href="https://30daymapchallenge.com/">30daymapchallenge.com</a> · Code and data on <a href="{REPO_URL}">GitHub</a></footer>
</main>
</body>
</html>
"""
    (DOCS / "index.html").write_text(html)
    print(f"wrote gallery: {BASE_URL}/ ({len(days)} days)")


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")  # serve files as-is (no Jekyll processing)
    days = [publish_day(m.parent) for m in sorted(ROOT.glob("day*/meta.json"))]
    gallery(days)


if __name__ == "__main__":
    main()
