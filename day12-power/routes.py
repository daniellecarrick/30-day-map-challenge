"""Trace Moses's four unbuilt expressways onto today's streets.

The routes were never built, so there is no official geometry. Each route below is a chain of legs
taken from the written route descriptions (sources in SOURCES), snapped to today's OpenStreetMap streets
and rail lines in data/osm_guides.json:
  ("along", name, A, B)   follow the named street/railway from A to B
  ("mid", nameA, nameB, (A, B), (A2, B2))  midline between two carriageways (a median the road would use)
  ("line", A, B)          straight connector where the route left the street grid (approximate)
A / B are ("x", street1, street2) = their intersection, ("near", street, A) = closest point on a street,
("end", street, "south") = its southernmost vertex, or (lon, lat).
Writes data/unbuilt_routes.geojson (EPSG:4326). Run:  ../.venv/bin/python routes.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring, transform
from pyproj import Transformer

HERE = Path(__file__).parent
DATA = HERE / "data"
CRS_M = "EPSG:32618"
to_m = Transformer.from_crs(4326, CRS_M, always_xy=True).transform
to_ll = Transformer.from_crs(CRS_M, 4326, always_xy=True).transform

NYCROADS = "http://www.nycroads.com/roads/"
# OSM maps the cemetery as a relation; this is a point near its middle (approximate, the route "cut through" it)
EVERGREENS = (-73.9010, 40.6840)
# Section 8 of the 1966 plan "must cross over Belt Parkway, over 84th Street": the Belt Parkway in Howard Beach
BELT_84TH = (-73.8585, 40.6650)
ROUTES = [
    {
        "name": "Lower Manhattan Expressway", "short": "LOMEX", "route": "I-78 / I-478", "borough": "Manhattan",
        "years": "1941–1971", "fate": "Demapped 1969, killed 1971",
        "plan": "1955 Joint Study elevated route",
        "source": NYCROADS + "lower-manhattan/",
        "trace": "West St along Canal St two blocks, over the Holland Tunnel plaza, along Watts St and Broome St, "
                 "meeting Delancey St at Suffolk St to the Williamsburg Bridge; spur down the Bowery to the Manhattan Bridge.",
        "legs": [
            [("along", "Canal Street", ("x", "Canal Street", "West Street"), ("x", "Canal Street", "Washington Street")),
             ("line", ("x", "Canal Street", "Washington Street"), ("x", "Watts Street", "Greenwich Street")),
             ("along", "Watts Street", ("x", "Watts Street", "Greenwich Street"), ("x", "Watts Street", "Broome Street")),
             ("along", "Broome Street", ("x", "Watts Street", "Broome Street"), ("x", "Broome Street", "Chrystie Street")),
             ("line", ("x", "Broome Street", "Chrystie Street"), ("x", "Delancey Street", "Suffolk Street")),
             ("along", "Delancey Street", ("x", "Delancey Street", "Suffolk Street"), ("x", "Delancey Street", "Clinton Street"))],
            [("along", "Bowery", ("x", "Broome Street", "Bowery"), ("x", "Canal Street", "Bowery")),
             ("line", ("x", "Canal Street", "Bowery"), ("x", "Canal Street", "Forsyth Street"))],
        ],
    },
    {
        "name": "Mid-Manhattan Expressway", "short": "Mid-Manhattan", "route": "I-495", "borough": "Manhattan",
        "years": "1929–1971", "fate": "Killed 1971",
        "plan": "1955 Joint Study elevated route",
        "source": NYCROADS + "mid-manhattan/",
        "trace": "Along 30th St from the West Side Highway (12th Ave) to the FDR Drive, with ramps to the Lincoln "
                 "and Queens-Midtown tunnels. Drawn on the 30th St centerline; the plan used a strip just south of it.",
        "legs": [
            [("along", "West 30th Street", ("x", "West 30th Street", "12th Avenue"), ("x", "West 30th Street", "5th Avenue")),
             ("along", "East 30th Street", ("x", "East 30th Street", "5th Avenue"), ("x", "East 30th Street", "1st Avenue")),
             ("line", ("x", "East 30th Street", "1st Avenue"), ("near", "FDR Drive", ("x", "East 30th Street", "1st Avenue")))],
        ],
    },
    {
        "name": "Bushwick Expressway", "short": "Bushwick", "route": "I-78", "borough": "Brooklyn / Queens",
        "years": "1955–1971", "fate": "Removed from city map 1969, killed 1971",
        "plan": "1963 recommended (middle) route",
        "source": NYCROADS + "bushwick/",
        "trace": "From the Williamsburg Bridge along Broadway to Flushing Ave, east on Flushing Ave to Wyckoff Ave, "
                 "southeast through the Cemetery of the Evergreens and Highland Park, down Force Tube Ave and the "
                 "Conduit Blvd median to the Belt Parkway–Nassau Expressway interchange.",
        "legs": [
            [("along", "Broadway", ("x", "Broadway", "Bedford Avenue"), ("x", "Broadway", "Flushing Avenue")),
             ("along", "Flushing Avenue", ("x", "Broadway", "Flushing Avenue"), ("x", "Flushing Avenue", "Wyckoff Avenue")),
             ("along", "Wyckoff Avenue", ("x", "Flushing Avenue", "Wyckoff Avenue"), ("x", "Wyckoff Avenue", "Myrtle Avenue")),
             ("line", ("x", "Wyckoff Avenue", "Myrtle Avenue"), EVERGREENS),
             ("line", EVERGREENS, ("x", "Force Tube Avenue", "Jamaica Avenue")),
             ("along", "Force Tube Avenue", ("x", "Force Tube Avenue", "Jamaica Avenue"), ("end", "Force Tube Avenue", "south")),
             ("line", ("end", "Force Tube Avenue", "south"), ("near", "North Conduit Boulevard", ("end", "Force Tube Avenue", "south"))),
             ("mid", "North Conduit Boulevard", "South Conduit Boulevard",
              (("near", "North Conduit Boulevard", ("end", "Force Tube Avenue", "south")), ("end", "North Conduit Boulevard", "east")),
              (("near", "South Conduit Boulevard", ("end", "Force Tube Avenue", "south")), ("end", "South Conduit Boulevard", "east"))),
             ("mid", "North Conduit Avenue", "South Conduit Avenue",
              (("end", "North Conduit Avenue", "west"), ("x", "North Conduit Avenue", "Cross Bay Boulevard")),
              (("end", "South Conduit Avenue", "west"), ("x", "South Conduit Avenue", "Cross Bay Boulevard")))],
        ],
    },
    {
        "name": "Cross-Brooklyn Expressway", "short": "Cross-Brooklyn", "route": "I-78 / I-878", "borough": "Brooklyn",
        "years": "1941–1971", "fate": "Withdrawn 1969, demapped 1971",
        "plan": "1966 TBTA–Blauvelt plan, Brooklyn section",
        "source": NYCROADS + "cross-brooklyn/",
        "trace": "From the Gowanus Expressway at 65th St along the LIRR Bay Ridge Branch to East New York, then "
                 "along Flatlands Ave (Schenck to Fountain Ave), then across the Spring Creek marshes to the Belt Parkway at 84th St.",
        "legs": [
            [("line", ("x", "7th Avenue", "65th Street"), ("near", "RAIL", ("x", "7th Avenue", "65th Street"))),
             ("along", "RAIL", ("near", "RAIL", ("x", "7th Avenue", "65th Street")),
              ("near", "RAIL", ("x", "Van Sinderen Avenue", "Linden Boulevard"))),
             ("line", ("near", "RAIL", ("x", "Van Sinderen Avenue", "Linden Boulevard")), ("x", "Flatlands Avenue", "Schenck Avenue")),
             ("along", "Flatlands Avenue", ("x", "Flatlands Avenue", "Schenck Avenue"), ("x", "Flatlands Avenue", "Fountain Avenue")),
             ("line", ("x", "Flatlands Avenue", "Fountain Avenue"), ("near", "Belt Parkway", BELT_84TH))],
        ],
    },
]


class Guides:
    def __init__(self, path: Path):
        els = json.loads(path.read_text())["elements"]
        self.by_name: dict[str, list] = {}
        self.nodes: dict[str, set] = {}
        self.polys: dict[str, list] = {}
        for e in els:
            if e["type"] != "way" or "geometry" not in e:
                continue
            tags = e.get("tags", {})
            name = "RAIL" if tags.get("railway") == "rail" and "Bay Ridge" in tags.get("name", "") else tags.get("name")
            line = LineString([(g["lon"], g["lat"]) for g in e["geometry"]])
            if tags.get("leisure") == "park" or tags.get("landuse") == "cemetery" or tags.get("amenity") == "grave_yard":
                self.polys.setdefault(name, []).append(line)
            self.by_name.setdefault(name, []).append(transform(to_m, line))
            self.nodes.setdefault(name, set()).update(e.get("nodes", []))
        self._pos = {}
        for e in els:
            for nid, g in zip(e.get("nodes", []), e.get("geometry", [])):
                self._pos[nid] = transform(to_m, Point(g["lon"], g["lat"]))

    def point(self, spec) -> Point:
        kind = spec[0]
        if kind == "x":
            a, b = spec[1], spec[2]
            shared = self.nodes.get(a, set()) & self.nodes.get(b, set())
            if shared:
                pts = [self._pos[n] for n in shared]
                return Point(np.mean([p.x for p in pts]), np.mean([p.y for p in pts]))
            # no shared node (e.g. one of them is a bridge or a dual carriageway): closest approach
            la, lb = MultiLineString(self.by_name[a]), MultiLineString(self.by_name[b])
            if la.distance(lb) > 40:
                raise ValueError(f"{a} and {b} do not meet ({la.distance(lb):.0f} m apart)")
            from shapely.ops import nearest_points
            pa, pb = nearest_points(la, lb)
            return Point((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
        if kind == "end":  # the street's extreme vertex in a compass direction
            pts = [Point(c) for l in self.by_name[spec[1]] for c in l.coords]
            key = {"south": lambda p: p.y, "north": lambda p: -p.y, "west": lambda p: p.x, "east": lambda p: -p.x}[spec[2]]
            return min(pts, key=key)
        if kind == "near":
            target = self.point(spec[2])
            lines = MultiLineString(self.by_name[spec[1]])
            from shapely.ops import nearest_points
            return nearest_points(lines, target)[0]
        if kind == "centroid":
            from shapely.geometry import Polygon
            rings = [Polygon(r.coords) for r in self.polys[spec[1]] if r.is_ring]
            big = max(rings, key=lambda p: transform(to_m, p).area)
            return transform(to_m, big.centroid)
        return transform(to_m, Point(spec))

    def along(self, name: str, p: Point, q: Point) -> LineString:
        """Shortest path along every way with this name (one-way pairs, gaps of < 90 m bridged)."""
        import networkx as nx
        from scipy.spatial import cKDTree
        G = nx.Graph()
        for line in self.by_name[name]:
            c = [tuple(np.round(xy, 1)) for xy in line.coords]
            for u, v in zip(c, c[1:]):
                G.add_edge(u, v, w=float(np.hypot(u[0] - v[0], u[1] - v[1])))
        nodes = list(G.nodes)
        tree = cKDTree(nodes)
        for i, j in tree.query_pairs(90):
            if not G.has_edge(nodes[i], nodes[j]):
                G.add_edge(nodes[i], nodes[j], w=float(np.hypot(*np.subtract(nodes[i], nodes[j]))) * 1.5)
        (dp, ip), (dq, iq) = tree.query((p.x, p.y)), tree.query((q.x, q.y))
        if dp > 80 or dq > 80:
            raise ValueError(f"{name}: route points are {dp:.0f} / {dq:.0f} m from the street")
        return LineString(nx.shortest_path(G, nodes[ip], nodes[iq], weight="w"))

    def mid(self, na: str, nb: str, pqa, pqb) -> LineString:
        la = self.along(na, self.point(pqa[0]), self.point(pqa[1]))
        lb = self.along(nb, self.point(pqb[0]), self.point(pqb[1]))
        t = np.linspace(0, 1, 120)
        pa = [la.interpolate(x, normalized=True) for x in t]
        pb = [lb.interpolate(x, normalized=True) for x in t]
        return LineString([((u.x + v.x) / 2, (u.y + v.y) / 2) for u, v in zip(pa, pb)])

    def leg(self, spec) -> LineString:
        kind = spec[0]
        if kind == "along":
            return self.along(spec[1], self.point(spec[2]), self.point(spec[3]))
        if kind == "mid":
            return self.mid(*spec[1:])
        return LineString([self.point(spec[1]), self.point(spec[2])])


def chain(parts: list[LineString]) -> LineString:
    coords = []
    for p in parts:
        c = list(p.coords)
        if coords and Point(coords[-1]).distance(Point(c[-1])) < Point(coords[-1]).distance(Point(c[0])):
            c = c[::-1]
        coords += c
    return LineString(coords)


def main():
    g = Guides(DATA / "osm_guides.json")
    rows = []
    for r in ROUTES:
        branches = [chain([g.leg(s) for s in legs]) for legs in r["legs"]]
        geom = MultiLineString(branches) if len(branches) > 1 else branches[0]
        props = {k: v for k, v in r.items() if k != "legs"}
        props["length_mi"] = round(geom.length / 1609.344, 1)
        rows.append({**props, "geometry": transform(to_ll, geom)})
        print(f"{r['name']:<30} {props['length_mi']:>5} mi")
    gpd.GeoDataFrame(rows, crs=4326).to_file(DATA / "unbuilt_routes.geojson", driver="GeoJSON")


if __name__ == "__main__":
    main()
