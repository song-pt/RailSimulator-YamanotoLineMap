"""Shared projection and OSM helpers. Local metric frame: x east, y north, origin at Tokyo station."""
import json, math
LAT0, LON0 = 35.6812, 139.7671
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110574.0

def proj(lon, lat):
    return ((lon - LON0) * KX, (lat - LAT0) * KY)

def unproj(x, y):
    return (x / KX + LON0, y / KY + LAT0)

class OSM:
    def __init__(self, path):
        d = json.load(open(path))
        self.nodes = {int(k): v for k, v in d["nodes"].items()}
        self.ways = {int(k): v for k, v in d["ways"].items()}
        self.rels = {int(k): v for k, v in d["relations"].items()}
    def xy(self, n):
        lon, lat, _ = self.nodes[n]
        return proj(lon, lat)
    def way_xy(self, wid):
        return [self.xy(n) for n in self.ways[wid]["nodes"] if n in self.nodes]
    def route_chain(self, rid):
        """Ordered polyline of a route relation's track members (those present in the extract)."""
        pts = []
        for typ, ref, role in self.rels[rid]["members"]:
            if typ != "way" or role != "" or ref not in self.ways:
                continue
            nodes = self.ways[ref]["nodes"]
            if pts and pts[-1][0] != nodes[0]:
                if pts[-1][0] == nodes[-1]:
                    nodes = nodes[::-1]
                elif len(pts) == len(self.ways[prev]["nodes"]) and pts[0][0] in (nodes[0], nodes[-1]):
                    pts = pts[::-1]
                    if pts[-1][0] == nodes[-1]:
                        nodes = nodes[::-1]
            for n in nodes:
                if not pts or pts[-1][0] != n:
                    pts.append((n, ref))
            prev = ref
        return pts
