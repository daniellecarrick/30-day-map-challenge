"""Parse the downloaded Wikipedia tables into one raw table: data/parsed.csv.

One row per table row, with the source list, name, the article each name/location links to,
founding and ending text (unparsed), and notes. clean.py turns this into the dataset.
Run: ../.venv/bin/python parse.py
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

# list file -> source label, and which header names mean what
SOURCES = {
    "List_of_American_utopian_communities": "main",
    "List_of_Owenite_communities_in_the_United_States": "owenite",
    "List_of_Fourierist_Associations_in_the_United_States": "fourierist",
    "Shaker_communities": "shaker",
}
COLS = {
    "name": ["name", "site"],
    "location": ["location"],
    "city": ["city"],
    "state": ["state"],
    "start": ["founding date", "launched"],
    "end": ["ending date", "terminated"],
    "dates": ["dates"],
    "notes": ["notes", "comments"],
    "spiritual": ["spiritual name"],
}


def strip_refs(s: str) -> str:
    s = re.sub(r"<ref[^>/]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    for _ in range(4):  # nested templates ({{#tag:ref|...{{sfn}}...}})
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s


def first_link(s: str) -> str:
    m = re.search(r"\[\[([^|\]#]+)", strip_refs(s))
    return m.group(1).strip() if m and not m.group(1).startswith("File:") else ""


def text(s: str) -> str:
    s = strip_refs(s)
    s = re.sub(r"\[\[File:[^\]]*\]\]", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", s)
    s = re.sub(r"'''?", "", s)
    s = re.sub(r"<br\s*/?>", "; ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())


def cell_value(c: str) -> str:
    # drop "align=..." style attributes before the content pipe
    m = re.match(r'^\s*(?:[a-z-]+=(?:"[^"]*"|[^\s|]+)\s*)+\|(?!\|)(.*)$', c, flags=re.S)
    return m.group(1) if m else c


def split_cells(block: str, marker: str) -> list[str]:
    """Cells of one table row; cells start a line with `marker` or follow `marker*2` inline."""
    cells, cur = [], None
    for line in block.split("\n"):
        if line.startswith(marker) and not line.startswith(("|}", "|-", "{|")):
            if cur is not None:
                cells.append(cur)
            parts = re.split(re.escape(marker * 2), line[1:])
            cells.extend(parts[:-1])
            cur = parts[-1]
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        cells.append(cur)
    return [cell_value(c) for c in cells]


def tables(txt: str):
    for m in re.finditer(r"^\{\|.*?^\|\}", txt, flags=re.S | re.M):
        rows = re.split(r"^\|-.*$", strip_refs(m.group(0)), flags=re.M)
        header = None
        for r in rows[1:]:
            if r.strip().startswith("!"):
                header = [text(h).lower() for h in split_cells(r.strip(), "!")]
                continue
            if header:
                yield header, split_cells(r.strip("\n"), "|")


def main():
    out = []
    for fname, src in SOURCES.items():
        txt = (RAW / f"{fname}.wiki").read_text()
        for header, cells in tables(txt):
            if src == "shaker" and len(cells) == len(header) + 1 and not cells[0].strip():
                cells = cells[1:]  # "|| ||" rows with no image carry one extra empty cell
            if src == "shaker" and len(cells) == len(header) - 1:
                cells = [""] + cells  # rows that start "|  ||" lost the empty image cell
            row = {"source": src}
            for key, names in COLS.items():
                idx = next((i for i, h in enumerate(header) if h in names), None)
                raw = cells[idx] if idx is not None and idx < len(cells) else ""
                row[key] = text(raw)
                if key in ("name", "location", "city"):
                    row[key + "_link"] = first_link(raw)
            if row["name"]:
                out.append(row)
    keys = ["source", "name", "name_link", "location", "location_link", "city", "city_link", "state",
            "start", "end", "dates", "notes", "spiritual"]
    with open(HERE / "data" / "parsed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print(len(out), "rows ->", "data/parsed.csv")
    for s in SOURCES.values():
        print(" ", s, sum(r["source"] == s for r in out))


if __name__ == "__main__":
    main()
