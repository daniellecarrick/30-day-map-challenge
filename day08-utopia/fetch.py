"""Download the Wikipedia tables Day 8 is built from (wikitext, into data/raw/).

Run: ../.venv/bin/python fetch.py
"""
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "30DayMapChallenge-build/1.0"}

PAGES = [
    "List_of_American_utopian_communities",
    "List_of_Owenite_communities_in_the_United_States",
    "List_of_Fourierist_Associations_in_the_United_States",
    "Shaker_communities",
]


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


if __name__ == "__main__":
    for p in PAGES:
        url = f"https://en.wikipedia.org/w/index.php?title={urllib.parse.quote(p)}&action=raw"
        (RAW / f"{p}.wiki").write_bytes(get(url))
        print("fetched", p)
