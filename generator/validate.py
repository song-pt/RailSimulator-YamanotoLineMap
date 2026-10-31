"""Clearance and placement checks. A build with violations is reported, not silently accepted.

Kinematic envelope per track (relative to rail top, half widths from track centre):
  0.00-1.15 m : 1.50 m  (car body below platform level; platform edge sits at 1.58 m)
  1.15-4.10 m : 1.75 m  (car body)
  4.10-5.60 m : 1.00 m  (pantograph zone; contact wire lives here by design)
Every solid obstacle (pier, mast, beam, platform, canopy, PSD, building, slab,
expressway, stairs) is a footprint prism and must not intersect any envelope.
"""
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

ZONES = [(0.05, 1.15, 1.50), (1.15, 4.10, 1.75), (4.10, 5.60, 1.00)]


def envelope_polys(p, half):
    """Rectangles along consecutive points of a track polyline."""
    out = []
    for a, b in zip(p[:-1], p[1:]):
        d = b[:2] - a[:2]
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        d /= L
        n = np.array([-d[1], d[0]]) * half
        out.append((Polygon([a[:2] - n, b[:2] - n, b[:2] + n, a[:2] + n]), min(a[2], b[2]), max(a[2], b[2])))
    return out


def check(m, region=None, tag="all"):
    obs = m.sc.obstacles
    tree = STRtree([o[0] for o in obs])
    rep = []
    viol = {}
    examples = {}
    for ti, tr in enumerate(m.tracks):
        p = tr.pts
        if region is not None and not tr.line.intersects(region):
            continue
        for z0, z1, half in ZONES:
            for poly, za, zb in envelope_polys(p, half):
                for c in tree.query(poly, predicate="intersects"):
                    op, oa, ob, lab = obs[c]
                    if oa < zb + z1 and ob > za + z0:
                        inter_g = op.intersection(poly)
                        if inter_g.area < 0.02:
                            continue
                        zr = getattr(m, "cell_zref", {}).get(c)
                        if zr is not None:   # platform cell: test the exact platform top over the overlap
                            pts = np.vstack([np.asarray(g.exterior.coords)[:, :2] for g in getattr(inter_g, "geoms", [inter_g]) if g.geom_type == "Polygon"])
                            top = float(m.plat_z(pts, zr).max()) + 1.10
                            if top <= za + z0 + 0.08:   # grade over one 2 m envelope segment
                                continue
                        viol[lab] = viol.get(lab, 0) + 1
                        examples.setdefault(lab, []).append((tr.name, round(float(poly.centroid.x), 1), round(float(poly.centroid.y), 1), round(za + z0, 1)))
    rep.append("clearance violations [%s]: %d %s" % (tag, sum(viol.values()), viol))
    for lab, ex in examples.items():
        rep.append("  %s e.g. %s" % (lab, ex[:4]))
    # below-street running must be an open cutting (retaining walls) or an explicitly covered section
    net = m.net
    below = (net.base_h < net.street - 1.5)
    cov = net.main_cls == "covered"
    rep.append("main line below street: %.0f m open cutting, %.0f m covered; covered sections roofed: %s" % (
        2 * np.sum(below & ~cov), 2 * np.sum(cov), any(pc["mode"] == "roof" and pc["covered"] for pc in m.pieces)))
    return rep
