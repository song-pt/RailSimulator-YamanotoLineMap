"""Fetch every dataset for the whole Yamanote loop corridor (OSM, GSI DEM, PLATEAU buildings).

Usage: python3 -I fetch_full.py WORKDIR
Only OSM tiles / PLATEAU 1 km meshes within CORRIDOR metres of the line are downloaded.
"""
import json, os, sys, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from shapely.geometry import LineString, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_osm, fetch_plateau
from dem import DEM

REL = 1972960
CORRIDOR = 550.0          # metres either side of the line
UA = {"User-Agent": "RailSimulator-map-generator/1.0"}
WARDS = ["千代田区", "中央区", "港区", "新宿区", "文京区", "台東区", "品川区", "目黒区", "渋谷区", "豊島区", "北区", "荒川区"]


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()


def line_lonlat(work):
    fn = os.path.join(work, "yamanote_rel.xml")
    if not os.path.exists(fn):
        open(fn, "wb").write(get("https://api.openstreetmap.org/api/0.6/relation/%d/full" % REL))
    r = ET.parse(fn).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in r.iter("node")}
    lines = []
    for w in r.iter("way"):
        pts = [nodes[n.get("ref")] for n in w.findall("nd") if n.get("ref") in nodes]
        if len(pts) > 1:
            lines.append(LineString(pts))
    return unary_union(lines)


def mesh3(lon, lat):
    p = int(lat * 1.5); u = int(lon - 100)
    q = int((lat * 1.5 - p) * 8); v = int((lon - 100 - u) * 8)
    r = int(((lat * 1.5 - p) * 8 - q) * 10); w = int(((lon - 100 - u) * 8 - v) * 10)
    return "%02d%02d%d%d%d%d" % (p, u, q, v, r, w)


def plateau_url(ward):
    q = urllib.parse.urlencode({"q": "plateau " + ward, "rows": 10})
    res = json.loads(get("https://www.geospatial.jp/ckan/api/3/action/package_search?" + q))["result"]["results"]
    best = None
    for p in res:
        if ward not in p["title"]:
            continue
        for r in p["resources"]:
            u = r["url"]
            if "citygml" in u and u.endswith(".zip"):
                year = int(p["name"].rsplit("-", 1)[-1]) if p["name"].rsplit("-", 1)[-1].isdigit() else 0
                if best is None or year > best[0]:
                    best = (year, u)
    return best[1] if best else None


def main(work):
    data = os.path.join(work, "data")
    os.makedirs(data, exist_ok=True)
    geo = line_lonlat(work)
    dlon, dlat = CORRIDOR / 90500.0, CORRIDOR / 110574.0
    corr = geo.buffer(max(dlon, dlat))
    w, s, e, n = corr.bounds
    print("corridor bbox", [round(v, 4) for v in (w, s, e, n)], flush=True)
    # OSM: only tiles that touch the corridor
    step = 0.004
    tiles = []
    y = s
    while y < n:
        x = w
        while x < e:
            if corr.intersects(box(x, y, x + step, y + step)):
                tiles.append((x, y))
            x += step
        y += step
    print("osm tiles", len(tiles), flush=True)
    os.makedirs(os.path.join(data, "osmcache"), exist_ok=True)
    nodes, ways, rels = {}, {}, {}
    for k, (x, y) in enumerate(tiles):
        root = fetch_osm.fetch_tile((x, y, min(x + step, e), min(y + step, n)), os.path.join(data, "osmcache"))
        for el in root:
            if el.tag not in ("node", "way", "relation"):
                continue
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            i = int(el.get("id"))
            if el.tag == "node":
                nodes[i] = (float(el.get("lon")), float(el.get("lat")), tags or None)
            elif el.tag == "way":
                ways[i] = {"nodes": [int(nd.get("ref")) for nd in el.findall("nd")], "tags": tags}
            else:
                rels[i] = {"members": [(m.get("type"), int(m.get("ref")), m.get("role")) for m in el.findall("member")], "tags": tags}
        if k % 25 == 0:
            print("osm", k, "/", len(tiles), flush=True)
    json.dump({"nodes": nodes, "ways": ways, "relations": rels}, open(os.path.join(data, "osm.json"), "w"))
    print("osm done", len(nodes), len(ways), flush=True)
    # DEM for the whole bbox
    DEM(w, s, e, n, os.path.join(data, "dem"))
    print("dem done", flush=True)
    # PLATEAU: 1 km meshes touching the corridor
    meshes = set()
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            if corr.intersects(box(lon, lat, lon + 0.0125, lat + 1 / 120)):
                meshes.add(mesh3(lon + 0.006, lat + 0.004))
            lon += 0.0125
        lat += 1 / 120
    pats = [p for m in sorted(meshes) for p in ("udx/bldg/%s_" % m,)]
    print("plateau meshes", len(meshes), flush=True)
    for ward in WARDS:
        url = plateau_url(ward)
        if not url:
            print("no plateau url for", ward); continue
        out = os.path.join(data, "plateau", ward)
        try:
            fetch_plateau.main(url, out, *pats)
        except Exception as ex:
            print("plateau failed", ward, ex)
        print("plateau", ward, "done", flush=True)
    json.dump({"bbox": [w, s, e, n], "meshes": sorted(meshes)}, open(os.path.join(data, "extent.json"), "w"))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
