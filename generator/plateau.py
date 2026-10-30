"""Parse PLATEAU CityGML buildings inside a corridor into a compact pickle.

Each building: id, name, usage, height, footprint (local xy), ground z, and surfaces
[(kind, (n,3) local xyz ring)] from LOD2 when present, else from LOD1.
"""
import glob, os, pickle, sys
from multiprocessing import Pool
import numpy as np
from lxml import etree
from shapely.geometry import Polygon, Point
from shapely import wkb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import proj

NS = {
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "uro": "https://www.geospatial.jp/iur/uro/3.1",
}
B = "{%s}" % NS["bldg"]; G = "{%s}" % NS["gml"]


def ring_xyz(pos):
    a = np.array(pos.split(), float).reshape(-1, 3)
    x, y = proj(a[:, 1], a[:, 0])
    return np.c_[x, y, a[:, 2]]


def surfaces(el, lod):
    out = []
    if lod == 2:
        for kind in ("RoofSurface", "WallSurface", "GroundSurface", "OuterCeilingSurface", "OuterFloorSurface", "ClosureSurface"):
            for s in el.iter(B + kind):
                for p in s.iter(G + "posList"):
                    out.append((kind, ring_xyz(p.text)))
        if not out:  # some LOD2 buildings only carry lod2Solid without thematic surfaces
            for s in el.iter(B + "lod2Solid"):
                for p in s.iter(G + "posList"):
                    out.append(("Mixed", ring_xyz(p.text)))
    else:
        for s in el.iter(B + "lod1Solid"):
            for p in s.iter(G + "posList"):
                out.append(("Mixed", ring_xyz(p.text)))
    return out


def parse_file(args):
    fn, zone_wkb = args
    zone = wkb.loads(zone_wkb)
    res = []
    for _, el in etree.iterparse(fn, tag=B + "Building", huge_tree=True):
        if el.getparent() is not None and el.getparent().tag == B + "consistsOfBuildingPart":
            continue
        gid = el.get(G + "id")
        first = next(el.iter(G + "posList"), None)
        if first is None:
            el.clear(); continue
        r = ring_xyz(first.text)
        c = r[:, :2].mean(0)
        if not zone.contains(Point(c)):
            el.clear(); continue
        h = el.find(B + "measuredHeight")
        name = el.find(G + "name")
        usage = el.find(B + "usage")
        lod = 2 if next(el.iter(B + "lod2Solid"), None) is not None or next(el.iter(B + "boundedBy"), None) is not None else 1
        surf = surfaces(el, lod)
        if not surf and lod == 2:
            surf = surfaces(el, 1); lod = 1
        fp = None
        for s in el.iter(B + "lod0RoofEdge", B + "lod0FootPrint"):
            for p in s.iter(G + "posList"):
                fp = ring_xyz(p.text); break
            if fp is not None: break
        allz = np.concatenate([s[1][:, 2] for s in surf]) if surf else np.array([0.0])
        if fp is None and surf:
            pts = np.concatenate([s[1] for s in surf])
            from shapely.geometry import MultiPoint
            fp = np.asarray(MultiPoint(pts[:, :2]).convex_hull.exterior.coords)
        if fp is None:
            el.clear(); continue
        res.append(dict(id=gid, name=name.text if name is not None else "", usage=usage.text if usage is not None else "",
                        height=float(h.text) if h is not None and h.text and float(h.text) > 0 else None,
                        footprint=np.asarray(fp)[:, :2], zmin=float(allz.min()), zmax=float(allz.max()), lod=lod,
                        surfaces=surf))
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]
    return res


def main(src_dir, zone_pickle, out):
    zone = pickle.load(open(zone_pickle, "rb"))
    files = sorted(glob.glob(os.path.join(src_dir, "*", "*_bldg_*.gml")), key=os.path.getsize, reverse=True)
    with Pool(4) as pool:
        parts = pool.map(parse_file, [(f, zone.wkb) for f in files])
    seen, allb = set(), []
    for p in parts:
        for b in p:
            if b["id"] in seen:
                continue
            seen.add(b["id"]); allb.append(b)
    pickle.dump(allb, open(out, "wb"))
    print(len(allb), "buildings;", sum(b["lod"] == 2 for b in allb), "LOD2")


if __name__ == "__main__":
    main(*sys.argv[1:])
