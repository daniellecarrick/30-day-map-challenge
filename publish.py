"""Publish finished days to docs/ (served by GitHub Pages) and rebuild the gallery.

For every dayNN-*/meta.json it:
  - copies the interactive page to docs/dayNN/index.html, adding link-preview tags
    (Open Graph + Twitter card) and a small "All maps" link
  - copies the PNG to docs/dayNN/map.png and makes preview.png (1200x630, for social cards)
    and thumb.png (for the gallery)
Then writes docs/index.html (gallery home page), docs/cover.png (30-day grid for the
portfolio) and drafts/week-N.md (weekly blog drafts, gitignored).

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
.progress {{ color: var(--muted); font-size: 14px; margin: 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: 20px; }}
.card {{ border: 1px solid color-mix(in srgb, var(--ink) 12%, transparent); border-radius: 10px; overflow: hidden;
         background: var(--bg); display: flex; flex-direction: column; }}
.card .thumb {{ display: block; background: #fff; aspect-ratio: 5 / 3; }}
.card .thumb img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.card .body {{ padding: 14px 16px 16px; }}
.kicker {{ margin: 0 0 4px; font-size: 14px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.card h2 {{ font-family: var(--font-head); font-size: 18px; line-height: 1.25; margin: 0; }}
.card h2 a {{ color: var(--ink); text-decoration: none; }}
.card h2 a:hover {{ text-decoration: underline; }}
.gallery > footer {{ text-align: center; color: var(--muted); font-size: 14px; margin-top: 28px; }}
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


PROMPTS = {int(k): v for k, v in json.loads((ROOT / "shared" / "prompts.json").read_text()).items()
           if not k.startswith("_")}
TOKENS = json.loads((ROOT / "shared" / "tokens.json").read_text())
INK, MUTED = TOKENS["color"]["light"]["ink"], TOKENS["color"]["light"]["muted"]
EMPTY = TOKENS["color"]["light"]["ctl"]
FONT_DIR = ROOT / "shared" / "fonts"
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 30)]


def cover(days: list[dict]) -> None:
    """docs/cover.png: all 30 days as a 6x5 grid. Finished days show their map thumbnail,
    upcoming days an empty tile with the prompt. Use as the portfolio case-study hero."""
    from PIL import ImageDraw, ImageFont
    done = {d["day"]: d for d in days}
    cols, rows, tw, th, lab, gap, mx, top = 6, 5, 360, 216, 58, 24, 60, 190
    W = mx * 2 + cols * tw + (cols - 1) * gap
    H = top + rows * (th + lab) + (rows - 1) * gap + 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(str(FONT_DIR / "Bitter-Bold.ttf"), 56)
    f_sub = ImageFont.truetype(str(FONT_DIR / "Raleway-Regular.ttf"), 26)
    f_lab = ImageFont.truetype(str(FONT_DIR / "Raleway-SemiBold.ttf"), 21)
    f_theme = ImageFont.truetype(str(FONT_DIR / "Raleway-Regular.ttf"), 20)
    f_day = ImageFont.truetype(str(FONT_DIR / "Bitter-SemiBold.ttf"), 30)
    d.text((mx, 60), SITE_TITLE, font=f_title, fill=INK)
    d.text((mx, 132), f"{len(days)} of 30 maps  ·  daniellecarrick.github.io/30-day-map-challenge",
           font=f_sub, fill=MUTED)
    for n in range(1, 31):
        r, c = divmod(n - 1, cols)
        x, y = mx + c * (tw + gap), top + r * (th + lab + gap)
        if n in done:
            thumb = Image.open(DOCS / done[n]["slug"] / "thumb.png").convert("RGB")
            thumb.thumbnail((tw, th), Image.LANCZOS)
            img.paste(thumb, (x + (tw - thumb.width) // 2, y + (th - thumb.height) // 2))
            d.rectangle([x, y, x + tw - 1, y + th - 1], outline="#e2e1dc", width=1)
            d.text((x, y + th + 12), f"Day {n} · {PROMPTS[n]}", font=f_lab, fill=INK)
        else:
            d.rounded_rectangle([x, y, x + tw - 1, y + th - 1], radius=10, fill=EMPTY)
            d.text((x + tw / 2, y + th / 2 - 14), f"Day {n}", font=f_day, fill="#b9b8b2", anchor="mm")
            d.text((x + tw / 2, y + th / 2 + 22), PROMPTS[n], font=f_theme, fill="#a3a29c", anchor="mm")
            d.text((x, y + th + 12), f"Day {n} · {PROMPTS[n]}", font=f_lab, fill="#b9b8b2")
    img.save(DOCS / "cover.png", optimize=True)
    print(f"wrote cover: {BASE_URL}/cover.png")


def weekly_drafts(days: list[dict]) -> None:
    """drafts/week-N.md: a Markdown blog draft per week, pre-filled from each day's meta.json.
    Images and links use the live GitHub Pages URLs so the draft works on any blog platform.
    Never overwrites an existing draft: if week-N.md exists and is out of date, the fresh
    version is written to week-N.new.md so you can copy new sections across."""
    out = ROOT / "drafts"
    out.mkdir(exist_ok=True)
    done = {d["day"]: d for d in days}
    for w, (a, b) in enumerate(WEEKS, start=1):
        in_week = [done[n] for n in range(a, b + 1) if n in done]
        if not in_week:
            continue
        lines = [
            f"# 30 Day Map Challenge, week {w}: days {a}–{b}",
            "",
            f"<!-- {len(in_week)} of {b - a + 1} days finished. Lead with your favorite map of the week. -->",
            "",
            "_Intro: what this week's prompts were, and one sentence on the theme that tied your maps together._",
            "",
        ]
        for n in range(a, b + 1):
            if n not in done:
                lines += [f"## Day {n} · {PROMPTS[n]}", "", "_Not published yet._", ""]
                continue
            m = done[n]
            url = f"{BASE_URL}/{m['slug']}/"
            lines += [
                f"## Day {n} · {m['theme']}: {m['title']}",
                "",
                f"[![{m['title']}]({url}map.png)]({url})",
                "",
                m["description"],
                "",
                f"**[Open the interactive map →]({url})**",
                "",
                "_What I tried, what surprised me, what I'd do differently._",
                "",
            ]
        lines += [
            "## What I learned this week",
            "",
            "_One or two takeaways about tools, data or design._",
            "",
            "---",
            "",
            f"All maps: [{BASE_URL.replace('https://', '')}]({BASE_URL}/) · "
            f"Code: [GitHub]({REPO_URL}) · Prompts: [30daymapchallenge.com](https://30daymapchallenge.com/)",
            "",
            "#30DayMapChallenge",
            "",
        ]
        text = "\n".join(lines)
        dest = out / f"week-{w}.md"
        if dest.exists():
            if dest.read_text() == text:
                continue
            dest = out / f"week-{w}.new.md"  # never overwrite a draft you may have edited
        dest.write_text(text)
        print(f"wrote draft: {dest.relative_to(ROOT)}")


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")  # serve files as-is (no Jekyll processing)
    days = [publish_day(m.parent) for m in sorted(ROOT.glob("day*/meta.json"))]
    gallery(days)
    cover(days)
    weekly_drafts(days)


if __name__ == "__main__":
    main()
