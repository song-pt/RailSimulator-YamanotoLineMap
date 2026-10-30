"""Build the corridor model: structures, track, platforms, stations, catenary, roads, buildings.

All placement goes through explicit clearance rules (see validate.py), so masts, piers,
canopies and buildings never enter a track's kinematic envelope, and nothing that should
sit on a structure floats or sinks below ground.
"""
import math, os, pickle, sys, zlib
import numpy as np
from shapely import segmentize, distance as sdist, points as spoints, contains_xy
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union, substring, polygonize, linemerge
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshkit as mk
from geo import OSM
from dem import DEM
from network import Network

GAUGE = 1.067
RAIL_HALF = GAUGE / 2 + 0.0325     # rail head centre offset
RAIL_H = 0.153
SLEEPER_TOP = -RAIL_H - 0.010
SLEEPER_H = 0.17
BALLAST_TOP = -0.25
DECK = 0.62                        # rail top -> structure deck top
RIBBON = 2.9                       # deck half width around a track
PLAT_EDGE = 1.58                   # platform edge from track centre
PLAT_H = 1.10                      # platform top above rail
CANOPY_H = 3.45                    # canopy underside above platform
CONTACT_H = 5.10
MESSENGER_H = 6.20
BEAM_H = 7.30
SPAN = 45.0
SLAB = 1.1                         # slab thickness of open spans
DETAIL_DIST = 70.0                 # tracks closer than this to the driven line get sleepers & fastenings


def hsh(s):
    return zlib.crc32(str(s).encode()) & 0xffffffff


class TrackIdx:
    """Segment index over all tracks: nearest-track queries with z interpolation."""

    def __init__(self, tracks):
        segs, meta = [], []
        for ti, tr in enumerate(tracks):
            p = tr.pts
            for i in range(len(p) - 1):
                segs.append(LineString([p[i, :2], p[i + 1, :2]])); meta.append((ti, i))
        self.segs = np.array(segs, dtype=object)
        self.meta = np.array(meta)
        self.tree = STRtree(self.segs)
        self.tracks = tracks

    def z_on(self, ti, i, x, y):
        p = self.tracks[ti].pts
        a, b = p[i], p[i + 1]
        d = b[:2] - a[:2]
        t = np.clip(((x - a[0]) * d[0] + (y - a[1]) * d[1]) / max(1e-9, d @ d), 0, 1)
        return a[2] + t * (b[2] - a[2])

    def near(self, x, y, r):
        """All (track index, distance, z) within r of point."""
        pt = Point(x, y)
        cand = self.tree.query(pt, predicate="dwithin", distance=r)
        best = {}
        for c in cand:
            ti, i = self.meta[c]
            d = self.segs[c].distance(pt)
            if ti not in best or d < best[ti][0]:
                best[ti] = (d, i)
        return [(ti, d, self.z_on(ti, i, x, y)) for ti, (d, i) in best.items()]

    def nearest_z(self, xy, zref=None, r=12.0):
        """z of nearest track (optionally the nearest with z close to zref) for points (n,2)."""
        out = np.empty(len(xy))
        for k, (x, y) in enumerate(xy):
            hits = self.near(x, y, r)
            if zref is not None:
                hits = [h for h in hits if abs(h[2] - zref) < 2.5] or hits
            if not hits:
                out[k] = zref if zref is not None else 0.0
            else:
                out[k] = min(hits, key=lambda h: h[1])[2]
        return out


class Model:
    def __init__(self, data, s_range=(250, 3650), drive=(650, 3250)):
        self.data = data
        self.osm = OSM(os.path.join(data, "osm.json"))
        self.dem = DEM(139.756, 35.672, 139.784, 35.706, os.path.join(data, "dem"))
        self.net = Network(self.osm, self.dem, 1972960, s_range)
        self.tracks = self.net.tracks
        self.idx = TrackIdx(self.tracks)
        self.sc = mk.Scene()
        self.drive = drive
        self.main = self.net.main
        self.zone = self.net.zone
        self.drive_line = substring(self.main, drive[0] - 50, drive[1] + 50)
        for tr in self.tracks:
            tr.near_main = tr.line.distance(self.drive_line) < DETAIL_DIST
        self.log = []

    # ------------------------------------------------------------------ helpers
    def lower_tracks(self, x, y, z, r, dz=0.5):
        return [h for h in self.idx.near(x, y, r) if h[2] < z - dz]

    def clear_of_tracks(self, x, y, z0, z1, r=2.45):
        """True if a vertical member at (x,y) spanning z0..z1 stays out of every track envelope."""
        for ti, d, z in self.idx.near(x, y, r):
            if z - 0.5 < z1 and z + 6.0 > z0:
                return False
        return True

    # ------------------------------------------------------------------ track
    def build_track(self):
        sc = self.sc
        for tr in self.tracks:
            p = tr.pts
            if len(p) < 2:
                continue
            t, left, up = mk.frames(p)
            rail = [(-0.0325, 0), (0.0325, 0), (0.0325, -0.05), (0.008, -0.06), (0.008, -0.125), (0.065, -0.14),
                    (0.065, -RAIL_H), (-0.065, -RAIL_H), (-0.065, -0.14), (-0.008, -0.125), (-0.008, -0.06), (-0.0325, -0.05)]
            for side in (-1, 1):
                prof = [(side * RAIL_HALF + a, b) for a, b in rail]
                mk.sweep(sc, "rail_head", p, prof[:3])
                mk.sweep(sc, "rail", p, prof[2:] + prof[:1])
            # ballast prism (or slab plinth for Shinkansen)
            if tr.shinkansen:
                prof = [(-1.25, -DECK), (-1.25, -0.30), (1.25, -0.30), (1.25, -DECK)]
                mk.sweep(sc, "slab_track", p, prof)
            else:
                prof = [(-2.05, -DECK + 0.01), (-1.35, BALLAST_TOP), (1.35, BALLAST_TOP), (2.05, -DECK + 0.01)]
                mk.sweep(sc, "ballast", p, prof)
            # sleepers (+ fastenings) only near the driven line
            if tr.near_main or tr.shinkansen:
                s = mk.cumlen(p)
                spacing = 0.625 if not tr.shinkansen else 1.25
                ss = np.arange(0.3, s[-1] - 0.3, spacing)
                c = np.c_[np.interp(ss, s, p[:, 0]), np.interp(ss, s, p[:, 1]), np.interp(ss, s, p[:, 2])]
                tt = np.c_[np.interp(ss, s, t[:, 0]), np.interp(ss, s, t[:, 1]), np.zeros(len(ss))]
                tt /= np.linalg.norm(tt, axis=1, keepdims=True)
                ll = np.c_[-tt[:, 1], tt[:, 0], np.zeros(len(ss))]
                if tr.shinkansen:
                    c2 = c.copy(); c2[:, 2] += -0.30 + 0.06 - 0.0  # track pads on slab
                    for side in (-1, 1):
                        mk.boxes(sc, "sleeper", c2 + ll * side * RAIL_HALF + [0, 0, -RAIL_H - 0.05], tt, ll, [0.5, 0.6, 0.12])
                else:
                    cz = c + [0, 0, SLEEPER_TOP - SLEEPER_H / 2]
                    mk.boxes(sc, "sleeper", cz, tt, ll, [0.24, 2.0, SLEEPER_H])
                    for side in (-1, 1):
                        for off in (-0.11, 0.11):
                            mk.boxes(sc, "fastening", c + ll * (side * RAIL_HALF + off) + [0, 0, SLEEPER_TOP + 0.02], tt, ll,
                                     [0.09, 0.07, 0.05])

    # ------------------------------------------------------------------ structures
    def build_structures(self):
        sc = self.sc
        pieces = []
        brick_lines = ("山手線", "京浜東北線")
        for ti, tr in enumerate(self.tracks):
            p = tr.pts
            n = len(p)
            step = 5
            for k in range(0, n - 1, step):
                sub = p[k:min(n, k + step + 1)]
                if len(sub) < 2:
                    continue
                line = LineString(sub[:, :2])
                zc = float(sub[:, 2].mean())
                fp = line.buffer(RIBBON, cap_style="flat", join_style="mitre")
                mid = sub[len(sub) // 2]
                low = self.lower_tracks(mid[0], mid[1], zc, RIBBON + 3.0, dz=0.45)
                if low:
                    cut = unary_union([self.tracks[t2].line.buffer(2.25) for t2, _, _ in low])
                    fp = fp.difference(cut)
                    mode = "over"
                elif tr.bridge:
                    mode = "girder"
                else:
                    mode = "solid"
                if fp.is_empty or fp.area < 1.0:
                    continue
                pieces.append(dict(ti=ti, fp=fp, sub=sub, mode=mode, z=zc, low=[t2 for t2, _, _ in low],
                                   mat="brick" if (any(b in tr.name for b in brick_lines) and mode == "solid") else "concrete"))
        self.pieces = pieces
        self.deck_union = unary_union([pc["fp"] for pc in pieces])
        self.piece_tree = STRtree([pc["fp"] for pc in pieces])

        for pc in pieces:
            sub = pc["sub"]
            line = LineString(sub[:, :2])
            sd = mk.cumlen(sub)

            def ztop(xy, sub=sub, line=line, sd=sd):
                s = np.array([line.project(Point(a, b)) for a, b in xy])
                return np.interp(s, sd, sub[:, 2]) - DECK

            fp = segmentize(pc["fp"], 2.5)
            if pc["mode"] == "solid":
                zb = lambda xy: self.dem(xy[:, 0], xy[:, 1]) - 0.4
                mk.prism(sc, "deck_top", pc["mat"], fp, ztop, zb)
            else:
                zb = lambda xy, zt=ztop: zt(xy) - SLAB
                mk.prism(sc, "deck_top", "girder" if pc["mode"] == "girder" else "concrete", fp, ztop, zb, bottom=True)
                sc.obstacle(pc["fp"], pc["z"] - DECK - SLAB, pc["z"] - DECK, "slab")
                if pc["mode"] == "over":
                    self.piers_for(pc, ztop)

    def piers_for(self, pc, ztop):
        sub = pc["sub"]
        tr = self.tracks[pc["ti"]]
        t, left, _ = mk.frames(sub)
        s = mk.cumlen(sub)
        for s0 in np.arange(2.0, s[-1], 12.0):
            k = min(len(sub) - 1, int(np.searchsorted(s, s0)))
            for off in (0.0, 1.6, -1.6, 2.2, -2.2):
                x, y = sub[k, :2] + left[k, :2] * off
                if not contains_xy(pc["fp"], x, y):
                    continue
                top = float(ztop(np.array([[x, y]]))[0]) - SLAB
                g = float(self.dem(x, y))
                if top - g < 1.0:
                    break
                if not self.clear_of_tracks(x, y, g, top, r=2.5):
                    continue
                mk.boxes(self.sc, "concrete", [[x, y, (top + g) / 2]], t[k], left[k], [1.4, 1.4, top - g])
                self.sc.obstacle(Point(x, y).buffer(1.0, quad_segs=2), g, top, "pier")
                break

    def build_parapets(self):
        sc = self.sc
        U = unary_union([self.deck_union] + [p for p, _ in self.platforms])
        U = U.buffer(0.05).buffer(-0.05)
        self.U = U
        lines = []
        for poly in mk.polys(U):
            for ring in [poly.exterior] + list(poly.interiors):
                if ring.length < 6:
                    continue
                lines.append(ring)
        plat_union = unary_union([p for p, _ in self.platforms]) if self.platforms else Polygon()
        for ring in lines:
            r = np.asarray(segmentize(LineString(ring.coords), 2.0).coords)
            if len(r) < 3:
                continue
            # pull the wall 0.15 m inside the deck
            c = LineString(r).offset_curve(0.0)
            z = self.idx.nearest_z(r[:, :2]) - DECK
            onplat = np.array([plat_union.distance(Point(a, b)) < 0.4 for a, b in r[:, :2]])
            pz = np.array([self.plat_top_at(a, b) for a, b in r[:, :2]])
            z = np.where(onplat & ~np.isnan(pz), pz, z)
            path = np.c_[r[:, :2], z]
            mk.sweep(sc, "parapet", path, [(-0.1, 0), (-0.1, 1.05), (0.1, 1.05), (0.1, 0)], left=self._ring_left(path))
            for a, b in zip(path[:-1], path[1:]):
                pass
        self.parapet_lines = lines

    def _ring_left(self, path):
        _, left, _ = mk.frames(path)
        return left

    # ------------------------------------------------------------------ platforms
    def platform_spans(self):
        osm = self.osm
        spans = []

        def side_of(tr, pt):
            s = tr.line.project(pt)
            a = np.array(tr.line.interpolate(max(0, s - 1)).coords[0]); b = np.array(tr.line.interpolate(min(tr.line.length, s + 1)).coords[0])
            q = np.array(tr.line.interpolate(s).coords[0])
            d = b - a; v = np.array(pt.coords[0]) - q
            return 1 if d[0] * v[1] - d[1] * v[0] > 0 else -1

        for wid, w in osm.ways.items():
            t = w["tags"]
            lay = str(t.get("layer", "0"))
            if lay.startswith("-") or t.get("tunnel") or t.get("indoor"):
                continue
            xy = osm.way_xy(wid)
            if len(xy) < 2:
                continue
            if t.get("railway") == "platform_edge":
                line = LineString(xy)
                if not line.intersects(self.zone):
                    continue
                mid = line.interpolate(0.5, normalized=True)
                hits = sorted(self.idx.near(mid.x, mid.y, 4.0), key=lambda h: h[1])
                if not hits:
                    continue
                tr = self.tracks[hits[0][0]]
                s0, s1 = sorted([tr.line.project(Point(xy[0])), tr.line.project(Point(xy[-1]))])
                if s1 - s0 > 30:
                    spans.append((hits[0][0], s0, s1, side_of(tr, mid)))
            elif t.get("railway") == "platform" or t.get("public_transport") == "platform":
                if xy[0] != xy[-1] or len(xy) < 4:
                    continue
                P = Polygon(xy).buffer(0)
                if not P.intersects(self.zone) or P.area < 50:
                    continue
                for ti, tr in enumerate(self.tracks):
                    if tr.line.distance(P) > 3.0 or tr.line.intersection(P.buffer(-0.6)).length > 5:
                        continue
                    seg = tr.line.intersection(P.buffer(3.0))
                    if seg.length < 30:
                        continue
                    ends = [Point(c) for g in getattr(seg, "geoms", [seg]) for c in (g.coords[0], g.coords[-1])]
                    ss = [tr.line.project(e) for e in ends]
                    s0, s1 = min(ss), max(ss)
                    spans.append((ti, s0, s1, side_of(tr, P.representative_point())))
        self.spans = spans
        return spans

    def build_platforms(self):
        spans = self.platform_spans()
        polys = []
        span_geo = []
        for ti, s0, s1, side in spans:
            tr = self.tracks[ti]
            ss = np.append(np.arange(s0, s1, 1.0), s1)
            inner, outer, mids = [], [], []
            for s in ss:
                q = np.array(tr.line.interpolate(s).coords[0])
                a = np.array(tr.line.interpolate(max(0, s - 1)).coords[0]); b = np.array(tr.line.interpolate(min(tr.line.length, s + 1)).coords[0])
                d = (b - a) / max(1e-9, np.linalg.norm(b - a)); n = np.array([-d[1], d[0]]) * side
                z = float(self.idx.nearest_z(q[None], r=1.0)[0])
                ray = LineString([q + n * 0.6, q + n * 22.0])
                dmin = None
                for c in self.idx.tree.query(ray, predicate="intersects"):
                    t2, i2 = self.idx.meta[c]
                    if t2 == ti or abs(self.idx.tracks[t2].pts[i2, 2] - z) > 1.2:
                        continue
                    ip = self.idx.segs[c].intersection(ray)
                    if ip.is_empty:
                        continue
                    dd = Point(q).distance(ip)
                    dmin = dd if dmin is None else min(dmin, dd)
                w = (dmin - PLAT_EDGE) if dmin is not None else PLAT_EDGE + 5.0
                if w < PLAT_EDGE + 1.5:
                    w = PLAT_EDGE + 1.5
                inner.append(q + n * PLAT_EDGE); outer.append(q + n * w); mids.append((q + n * (PLAT_EDGE + w) / 2, z, (w - PLAT_EDGE)))
            P = Polygon(inner + outer[::-1]).buffer(0)
            if P.area > 30:
                polys.append((P, ti))
                span_geo.append(dict(ti=ti, s0=s0, s1=s1, side=side, inner=np.array(inner), mids=mids, poly=P))
        # cut each span polygon by the envelopes of tracks on the same level, then merge
        cutp = []
        for P, ti in polys:
            zt = float(self.tracks[ti].pts[:, 2].mean())
            cut = [tr.line.buffer(PLAT_EDGE - 0.001) for tr in self.tracks
                   if tr.line.distance(P) < 3.0 and abs(float(self.idx.nearest_z(np.array([P.representative_point().coords[0]]), zref=zt, r=40)[0]) - zt) < 3
                   and abs(tr.pts[:, 2].mean() - zt) < 2.5]
            cutp.append(P.difference(unary_union(cut)) if cut else P)
        allp = unary_union(cutp)
        self.platforms = []
        for P in mk.polys(allp):
            if P.area < 40:
                continue
            c = P.representative_point()
            ti = min(polys, key=lambda pp: pp[0].distance(c))[1]
            self.platforms.append((P, ti))
        self.span_geo = span_geo
        self.plat_tree = STRtree([p for p, _ in self.platforms]) if self.platforms else None
        self.log.append("platforms: %d polygons from %d spans" % (len(self.platforms), len(spans)))

    def plat_top_at(self, x, y):
        if not self.plat_tree:
            return np.nan
        for c in self.plat_tree.query(Point(x, y), predicate="dwithin", distance=0.5):
            return float(self.idx.nearest_z(np.array([[x, y]]), r=25.0)[0]) + PLAT_H
        return np.nan

    def platform_meshes(self):
        sc = self.sc
        lowclear_all = []
        for P, ti in self.platforms:
            zref = float(np.mean([self.tracks[ti].pts[:, 2].mean()]))
            Pd = segmentize(P, 2.0)
            ztop = lambda xy, zr=zref: self.idx.nearest_z(xy, zref=zr, r=25.0) + PLAT_H
            zslab = lambda xy, zr=zref: self.idx.nearest_z(xy, zref=zr, r=25.0) - DECK - SLAB
            mk.prism(sc, "platform", "platform_side", Pd, ztop, zslab, bottom=True)
            zt = float(self.tracks[ti].pts[:, 2].mean())
            sc.obstacle(P, zt - DECK - SLAB, zt + PLAT_H, "platform")
            # fill below the platform slab down to ground, except above lower tracks
            low = [tr.line.buffer(2.6) for tr in self.tracks if tr.line.distance(P) < 3 and tr.pts[:, 2].mean() < zt - 1.5]
            fill = P.difference(unary_union(low)) if low else P
            for q in mk.polys(fill.buffer(-0.02)):
                mk.prism(sc, "concrete", "concrete", segmentize(q, 3.0), zslab, lambda xy: self.dem(xy[:, 0], xy[:, 1]) - 0.4)

    def platform_furniture(self):
        sc = self.sc
        cols_all = []
        self.columns = []
        stations = self.station_names()
        for sg in self.span_geo:
            tr = self.tracks[sg["ti"]]
            mids = sg["mids"]
            inner = sg["inner"]
            # tactile strip + white edge line + platform screen doors on Yamanote / Keihin-Tohoku edges
            zt = np.array([m[1] for m in mids]) + PLAT_H
            ii = np.c_[inner, zt]
            _, left, _ = mk.frames(ii)
            nrm = np.array([(m[0] - i) / max(1e-6, np.linalg.norm(m[0] - i)) for m, i in zip(mids, inner)])
            nrm3 = np.c_[nrm, np.zeros(len(nrm))]
            sign = np.sign(np.sum(left[:, :2] * nrm, 1))[:, None]
            L = left * sign
            mk.sweep(sc, "edge_white", ii + [0, 0, 0.004], [(0.0, 0), (0.12, 0)], left=L)
            mk.sweep(sc, "tactile", ii + [0, 0, 0.006], [(0.80, 0), (1.10, 0)], left=L)
            psd = any(k in tr.name for k in ("山手線", "京浜東北線"))
            if psd and len(ii) > 10:
                s = mk.cumlen(ii)
                posts = np.arange(2.0, s[-1] - 2.0, 2.0)
                base = np.c_[[np.interp(posts, s, ii[:, j]) for j in range(3)]].T + 0 * posts[:, None]
                Lp = np.c_[[np.interp(posts, s, L[:, j]) for j in range(3)]].T
                Lp /= np.linalg.norm(Lp, axis=1, keepdims=True) + 1e-9
                Tp = np.c_[Lp[:, 1], -Lp[:, 0], np.zeros(len(Lp))]
                pos = base + Lp * 0.42
                # door panels between posts (lower opaque band + glazing)
                mk.boxes(sc, "psd_frame", pos + [0, 0, 0.6], Tp, Lp, [1.95, 0.14, 1.2])
                mk.boxes(sc, "psd_frame", pos + [0, 0, 1.27], Tp, Lp, [1.98, 0.18, 0.14])
                for q in pos:
                    sc.obstacle(Point(q[0], q[1]).buffer(0.5, quad_segs=2), q[2], q[2] + 1.35, "psd")
            # columns along the platform middle, canopy is built per platform polygon below
            s_m = np.arange(6, len(mids) - 6, 12)
            for k in s_m:
                c, z, w = mids[int(k)]
                if w < 3.2:
                    continue
                if any(np.hypot(*(c - q)) < 3.0 for q, _ in self.columns):
                    continue
                self.columns.append((c, z + PLAT_H))
        # canopies: platform polygons shrunk at the ends
        for P, ti in self.platforms:
            zt = float(np.median(self.idx.nearest_z(np.asarray(P.exterior.coords)[::5, :2], zref=self.tracks[ti].pts[:, 2].mean(), r=25.0))) + PLAT_H
            roof = P.buffer(-0.12)
            cols = [c for c, z in self.columns if contains_xy(P, c[0], c[1])]
            if not cols or roof.is_empty:
                continue
            # limit canopy to the column extent (+6 m)
            hull = unary_union([Point(c).buffer(9.0) for c in cols])
            roof = roof.intersection(hull.convex_hull if len(cols) < 3 else unary_union([LineString([a, b]).buffer(9.0) for a, b in zip(cols[:-1], cols[1:])] + [hull]))
            for q in mk.polys(roof):
                mk.prism(sc, "canopy_top", "canopy_fascia", segmentize(q, 3.0), zt + CANOPY_H + 0.35, zt + CANOPY_H, bottom=False)
                mk.prism(sc, "canopy_under", "canopy_fascia", segmentize(q, 3.0), zt + CANOPY_H, zt + CANOPY_H - 0.001, top_only=True)
                sc.obstacle(q, zt + CANOPY_H, zt + CANOPY_H + 0.35, "canopy")
            cb = np.array([[c[0], c[1], zt] for c in cols])
            mk.cylinders(sc, "column", cb, CANOPY_H, 0.17)
            for c in cols:
                sc.obstacle(Point(c).buffer(0.2, quad_segs=2), zt, zt + CANOPY_H, "column")
            # station name boards hanging under the canopy + benches
            name = self.station_for(P)
            for k, c in enumerate(cols):
                d = self.platform_dir(P, c)
                if d is None:
                    continue
                ax = np.array([d[0], d[1], 0.0]); ay = np.array([-d[1], d[0], 0.0])
                if k % 3 == 1:
                    mk.boxes(sc, "sign_%s" % name, [[c[0], c[1], zt + CANOPY_H - 0.45]], ay, ax, [1.6, 0.06, 0.42])
                    mk.boxes(sc, "metal", [[c[0], c[1], zt + CANOPY_H - 0.12]], ay, ax, [0.04, 0.04, 0.24])
                if k % 3 == 2:
                    mk.boxes(sc, "bench", [[c[0] + d[0] * 2.5, c[1] + d[1] * 2.5, zt + 0.42]], ax, ay, [1.8, 0.45, 0.06])
                    mk.boxes(sc, "metal", [[c[0] + d[0] * 2.5, c[1] + d[1] * 2.5, zt + 0.2]], ax, ay, [1.6, 0.3, 0.4])
            # stair housings on wide platforms
            for k in range(1, len(cols) - 1, 5):
                c = cols[k]
                d = self.platform_dir(P, c)
                if d is None or P.buffer(-1.6).distance(Point(c)) > 0:
                    continue
                w = self.platform_width_at(P, c, d)
                if w < 8.0:
                    continue
                ax = np.array([d[0], d[1], 0.0]); ay = np.array([-d[1], d[0], 0.0])
                cc = np.array([c[0] + d[0] * 6, c[1] + d[1] * 6, zt])
                mk.boxes(sc, "stair_pit", [cc + [0, 0, 0.005]], ax, ay, [11.0, 2.6, 0.01])
                for sd in (-1, 1):
                    mk.boxes(sc, "psd_frame", [cc + ay * sd * 1.35 + [0, 0, 0.55]], ax, ay, [11.0, 0.1, 1.1])
                mk.boxes(sc, "psd_frame", [cc - ax * 5.5 + [0, 0, 0.55]], ax, ay, [0.1, 2.7, 1.1])
                fp = Polygon([cc[:2] + ax[:2] * a * 5.6 + ay[:2] * b * 1.45 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                sc.obstacle(fp, zt, zt + 1.1, "stairs")

    def platform_dir(self, P, c):
        hits = sorted(self.idx.near(c[0], c[1], 25.0), key=lambda h: h[1])
        if not hits:
            return None
        tr = self.tracks[hits[0][0]]
        s = tr.line.project(Point(c))
        a = np.array(tr.line.interpolate(max(0, s - 2)).coords[0]); b = np.array(tr.line.interpolate(min(tr.line.length, s + 2)).coords[0])
        d = b - a
        return d / max(1e-9, np.linalg.norm(d))

    def platform_width_at(self, P, c, d):
        n = np.array([-d[1], d[0]])
        seg = LineString([c - n * 20, c + n * 20]).intersection(P)
        return seg.length

    def station_names(self):
        if hasattr(self, "_stations"):
            return self._stations
        st = []
        for nid, (lon, lat, tags) in self.osm.nodes.items():
            if tags and tags.get("railway") == "station" and tags.get("name"):
                from geo import proj
                x, y = proj(lon, lat)
                if self.zone.contains(Point(x, y)):
                    st.append((tags["name"], tags.get("name:en", ""), np.array([x, y])))
        self._stations = st
        return st

    def station_for(self, P):
        c = np.array(P.representative_point().coords[0])
        best = min(self.station_names(), key=lambda s: np.hypot(*(s[2] - c)))
        return best[0]

    # ------------------------------------------------------------------ catenary
    def build_catenary(self):
        sc = self.sc
        supports = {ti: [] for ti in range(len(self.tracks))}
        main = self.main
        s0, s1 = self.drive[0] - 400, self.drive[1] + 400
        col_pts = [c for c, _ in self.columns]
        nmast = 0
        for s in np.arange(max(0, s0), min(main.length, s1), SPAN):
            placed = False
            for ds in (0, 4, -4, 8, -8, 12, -12):
                if self.gantry(s + ds, supports, col_pts):
                    placed = True; nmast += 1
                    break
        # wires
        for ti, tr in enumerate(self.tracks):
            p = tr.pts
            if len(p) < 3:
                continue
            s = mk.cumlen(p)
            sup = sorted(supports[ti])
            _, left, _ = mk.frames(p)
            zig = np.zeros(len(p))
            if sup:
                ss = np.array([a for a, _ in sup]); zz = np.array([b for _, b in sup])
                zig = np.interp(s, ss, zz)
            ceil = self.ceiling(tr)
            ch = np.minimum(CONTACT_H, ceil - 0.6)
            mh = np.minimum(MESSENGER_H, ceil - 0.25)
            mh = np.maximum(mh, ch + 0.25)
            # messenger sag between supports
            sag = np.zeros(len(p))
            if len(sup) >= 2:
                ss = np.array([a for a, _ in sup])
                k = np.clip(np.searchsorted(ss, s) - 1, 0, len(ss) - 2)
                u = np.clip((s - ss[k]) / np.maximum(1e-6, ss[k + 1] - ss[k]), 0, 1)
                sag = 4 * u * (1 - u) * np.minimum(0.55, (mh - ch - 0.25))
            cw = p + left * zig[:, None] + np.c_[np.zeros((len(p), 2)), ch]
            mw = p + left * zig[:, None] + np.c_[np.zeros((len(p), 2)), mh - sag]
            mk.tubes(sc, "wire_copper", [cw], 0.007)
            mk.tubes(sc, "wire_steel", [mw], 0.006)
            dd = np.arange(2.5, s[-1], 5.0)
            a = np.c_[[np.interp(dd, s, cw[:, j]) for j in range(3)]].T
            b = np.c_[[np.interp(dd, s, mw[:, j]) for j in range(3)]].T
            mk.segment_boxes(sc, "wire_steel", a, b, 0.008, 0.008)
        self.log.append("gantries placed: %d" % nmast)

    def ceiling(self, tr):
        """Lowest structure soffit above each track point (relative to rail), default high."""
        p = tr.pts
        out = np.full(len(p), 99.0)
        for k, q in enumerate(p):
            for c in self.piece_tree.query(Point(q[0], q[1])):
                pc = self.pieces[c]
                if pc["z"] - q[2] > 3.0 and contains_xy(pc["fp"], q[0], q[1]):
                    out[k] = min(out[k], pc["z"] - DECK - SLAB - q[2])
        return out

    def gantry(self, s, supports, col_pts):
        sc = self.sc
        main = self.main
        if s < 1 or s > main.length - 1:
            return False
        P = np.array(main.interpolate(s).coords[0])
        a = np.array(main.interpolate(s - 1).coords[0]); b = np.array(main.interpolate(s + 1).coords[0])
        d = (b - a) / np.linalg.norm(b - a); n = np.array([-d[1], d[0]])
        cross = LineString([P - n * 70, P + n * 70])
        hits = []
        for c in self.idx.tree.query(cross, predicate="intersects"):
            ti, i = self.idx.meta[c]
            ip = self.idx.segs[c].intersection(cross)
            if ip.is_empty or ip.geom_type != "Point":
                continue
            t = float(np.dot(np.array(ip.coords[0]) - P, n))
            z = self.idx.z_on(ti, i, ip.x, ip.y)
            seg = self.idx.segs[c].coords
            dirv = np.array(seg[1]) - np.array(seg[0]); dirv /= np.linalg.norm(dirv) + 1e-9
            if abs(np.dot(dirv, d)) < 0.75:  # crossing at a steep angle: not part of this cross-section
                continue
            hits.append((t, ti, z, i))
        if not hits:
            return False
        hits.sort()
        # dedupe very close hits of the same track
        groups, cur = [], [hits[0]]
        for h in hits[1:]:
            prev = cur[-1]
            if abs(h[2] - prev[2]) < 1.0 and h[0] - prev[0] < 16.0:
                cur.append(h)
            else:
                groups.append(cur); cur = [h]
        groups.append(cur)
        ok_any = False
        for g in groups:
            ts = [h[0] for h in g]
            zg = float(np.median([h[2] for h in g]))
            cand = [ts[0] - 2.6, ts[-1] + 2.6]
            # intermediate masts in wide gaps (platform middles etc.) to bound the beam span
            for (t1, *_), (t2, *_) in zip(g[:-1], g[1:]):
                if t2 - t1 > 7.0:
                    cand.append((t1 + t2) / 2)
            masts = []
            for t in sorted(cand):
                okm = None
                for dt in (0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                    tt = t + dt
                    x, y = P + n * tt
                    if not self.mast_ok(x, y, zg, col_pts):
                        continue
                    okm = tt; break
                masts.append(okm)
            if masts[0] is None or masts[-1] is None:
                continue
            masts = [m for m in masts if m is not None]
            beam_z = zg + BEAM_H
            A = P + n * masts[0]; B = P + n * masts[-1]
            beam_fp = LineString([A, B]).buffer(0.3)
            if self.blocked(beam_fp, beam_z - 0.4, beam_z + 0.6):
                continue
            for m in masts:
                x, y = P + n * m
                base = self.base_z(x, y, zg)
                mk.cylinders(sc, "mast", [[x, y, base]], beam_z + 0.35 - base, 0.16)
                mk.boxes(sc, "concrete", [[x, y, base + 0.3]], [d[0], d[1], 0], [n[0], n[1], 0], [0.9, 0.9, 0.6])
                sc.obstacle(Point(x, y).buffer(0.2, quad_segs=2), base, beam_z, "mast")
            mk.segment_boxes(sc, "mast", [[A[0], A[1], beam_z + 0.2]], [[B[0], B[1], beam_z + 0.2]], 0.25, 0.35)
            mk.segment_boxes(sc, "mast", [[A[0], A[1], beam_z - 0.15]], [[B[0], B[1], beam_z - 0.15]], 0.12, 0.12)
            sc.obstacle(beam_fp, beam_z - 0.25, beam_z + 0.4, "beam")
            for t, ti, z, i in g:
                tr = self.tracks[ti]
                ss = tr.line.project(Point(P + n * t))
                zig = 0.2 if (int(s / SPAN) % 2) else -0.2
                supports[ti].append((ss, zig))
                x, y = P + n * t
                # drop pipe + registration arm + insulator
                mk.segment_boxes(sc, "mast", [[x, y, beam_z - 0.2]], [[x, y, z + MESSENGER_H + 0.25]], 0.06, 0.06)
                mk.segment_boxes(sc, "mast", [[x, y, z + MESSENGER_H + 0.2]], [[x + n[0] * zig, y + n[1] * zig, z + CONTACT_H + 0.05]], 0.035, 0.035)
                mk.cylinders(sc, "insulator", [[x, y, z + MESSENGER_H + 0.25]], 0.35, 0.07, sides=8)
            ok_any = True
        return ok_any

    def base_z(self, x, y, zg):
        """Top of whatever structure carries a mast at (x,y): platform, deck or ground."""
        pz = self.plat_top_at(x, y)
        if not np.isnan(pz) and abs(pz - zg - PLAT_H) < 1.5:
            return pz
        return zg - DECK

    def mast_ok(self, x, y, zg, col_pts):
        if not self.clear_of_tracks(x, y, zg - DECK, zg + BEAM_H, r=2.35):
            return False
        if not contains_xy(self.U.buffer(-0.35), x, y):
            return False  # must stand on the deck / platform, not hang in the air
        for c in col_pts:
            if abs(c[0] - x) < 1.2 and abs(c[1] - y) < 1.2:
                return False
        if self.blocked(Point(x, y).buffer(0.25), zg - DECK + 0.3, zg + BEAM_H, skip=("slab", "platform")):
            return False
        return True

    def blocked(self, fp, z0, z1, skip=()):
        if not hasattr(self, "_otree") or self._otree_n != len(self.sc.obstacles):
            self._otree = STRtree([o[0] for o in self.sc.obstacles]); self._otree_n = len(self.sc.obstacles)
        for c in self._otree.query(fp, predicate="intersects"):
            poly, a, b, lab = self.sc.obstacles[c]
            if lab in skip:
                continue
            if a < z1 and b > z0:
                return True
        return False

    # ------------------------------------------------------------------ expressways
    def build_expressways(self):
        sc = self.sc
        for wid, w in self.osm.ways.items():
            t = w["tags"]
            if t.get("highway") not in ("motorway", "motorway_link") or t.get("tunnel"):
                continue
            xy = self.osm.way_xy(wid)
            if len(xy) < 2:
                continue
            line = LineString(xy)
            if not line.intersects(self.zone):
                continue
            p = mk.np.array(segmentize(line, 4.0).coords)
            g = self.dem(p[:, 0], p[:, 1])
            z = g + 10.0
            for k, (x, y) in enumerate(p):
                for ti, d, zr in self.idx.near(x, y, 12.0):
                    z[k] = max(z[k], zr + 8.6 + 1.6)
            # lift only (keeps clearance) and limit the resulting grade to ~5 %
            seg = np.r_[0, np.hypot(*np.diff(p, axis=0).T)]
            for _ in range(2):
                for k in range(1, len(z)):
                    z[k] = max(z[k], z[k - 1] - 0.05 * seg[k])
                for k in range(len(z) - 2, -1, -1):
                    z[k] = max(z[k], z[k + 1] - 0.05 * seg[k + 1])
            path = np.c_[p, z]
            width = 3.6 * int(t.get("lanes", "2")) / 2 + 1.5
            mk.sweep(sc, "asphalt", path, [(-width, 0), (width, 0)])
            mk.sweep(sc, "concrete", path, [(width, 0), (width, -2.0), (-width, -2.0), (-width, 0)])
            mk.sweep(sc, "parapet", path, [(width, 0), (width, 1.1), (width - 0.25, 1.1), (width - 0.25, 0)])
            mk.sweep(sc, "parapet", path, [(-width + 0.25, 0), (-width + 0.25, 1.1), (-width, 1.1), (-width, 0)])
            fp = LineString(p).buffer(width)
            sc.obstacle(fp, float(z.min()) - 2.0, float(z.max()) + 1.1, "expressway")
            s = mk.cumlen(path)
            _, left, _ = mk.frames(path)
            for s0 in np.arange(10, s[-1], 30):
                k = int(np.searchsorted(s, s0))
                k = min(k, len(p) - 1)
                x, y = p[k]
                if not self.clear_of_tracks(x, y, g[k], z[k], r=3.0) or contains_xy(self.U.buffer(1.0), x, y):
                    continue
                mk.boxes(sc, "concrete", [[x, y, (z[k] - 2 + g[k]) / 2]], [left[k][1], -left[k][0], 0], left[k], [2.0, 3.0, z[k] - 2 - g[k]])

    # ------------------------------------------------------------------ ground, roads, water
    def build_ground(self):
        sc = self.sc
        zone = self.zone.buffer(60)
        minx, miny, maxx, maxy = zone.bounds
        step = 6.0
        xs = np.arange(minx, maxx + step, step); ys = np.arange(miny, maxy + step, step)
        X, Y = np.meshgrid(xs, ys)
        Z = self.dem(X.ravel(), Y.ravel()).reshape(X.shape)
        water = self.water_polys()
        if water:
            W = unary_union(water)
            inside = contains_xy(W.buffer(4.0), X.ravel(), Y.ravel()).reshape(X.shape)
            Z = np.where(inside, -1.5, Z)
        keep = contains_xy(zone, X.ravel(), Y.ravel()).reshape(X.shape)
        ny, nx = X.shape
        idx = np.arange(nx * ny).reshape(ny, nx)
        q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1)
        qk = keep[:-1, :-1] | keep[1:, 1:]
        sc.b("ground").add(np.c_[X.ravel(), Y.ravel(), Z.ravel()], q[qk].reshape(-1, 4), np.c_[X.ravel(), Y.ravel()])
        # roads
        widths = {"trunk": 10.0, "primary": 9.0, "secondary": 7.5, "tertiary": 6.0, "residential": 3.5, "unclassified": 3.5,
                  "living_street": 3.0, "service": 2.5, "pedestrian": 3.0}
        polys = []
        marks = []
        for wid, w in self.osm.ways.items():
            t = w["tags"]
            hw = t.get("highway")
            if hw not in widths or t.get("bridge") or t.get("tunnel") or t.get("area") == "yes":
                continue
            if str(t.get("layer", "0")) not in ("0", ""):
                continue
            xy = self.osm.way_xy(wid)
            if len(xy) < 2:
                continue
            ln = LineString(xy)
            if not ln.intersects(zone):
                continue
            polys.append(ln.buffer(widths[hw], cap_style="round"))
            if widths[hw] >= 6.0:
                marks.append(ln)
        roads = unary_union(polys).intersection(zone)
        dz = lambda xy: self.dem(xy[:, 0], xy[:, 1]) + 0.08
        for r in mk.polys(roads):
            mk.prism(sc, "asphalt", "asphalt", segmentize(r, 4.0), dz, None, top_only=True)
        kerb = roads.buffer(3.0).difference(roads).intersection(zone)
        for r in mk.polys(kerb):
            mk.prism(sc, "sidewalk", "sidewalk", segmentize(r, 4.0), lambda xy: self.dem(xy[:, 0], xy[:, 1]) + 0.12, None, top_only=True)
        for ln in marks:
            ln = ln.intersection(zone)
            for g in getattr(ln, "geoms", [ln]):
                if g.geom_type != "LineString" or g.length < 8:
                    continue
                pts = np.asarray(segmentize(g, 2.0).coords)
                path = np.c_[pts, self.dem(pts[:, 0], pts[:, 1]) + 0.1]
                mk.sweep(sc, "road_mark", path, [(-0.08, 0), (0.08, 0)])
        for wp in water:
            wp = wp.intersection(zone)
            for q in mk.polys(wp):
                mk.prism(sc, "water", "concrete", segmentize(q, 4.0), 1.0, None, top_only=True)
                ring = np.asarray(segmentize(q.exterior, 3.0).coords)
                g = self.dem(ring[:, 0], ring[:, 1])
                path = np.c_[ring, np.full(len(ring), -1.5)]
                V = np.concatenate([path, np.c_[ring, np.maximum(g, 1.5)]])
                n = len(ring); i = np.arange(n - 1)
                sc.b("concrete").add(V, np.stack([i, i + 1, i + 1 + n, i + n], 1), np.c_[np.r_[mk.cumlen(path), mk.cumlen(path)], V[:, 2]])

    def water_polys(self):
        if hasattr(self, "_water"):
            return self._water
        out = []
        for wid, w in self.osm.ways.items():
            t = w["tags"]
            if t.get("natural") == "water" or t.get("waterway") == "riverbank":
                xy = self.osm.way_xy(wid)
                if len(xy) > 3 and xy[0] == xy[-1]:
                    out.append(Polygon(xy).buffer(0))
        for rid, r in self.osm.rels.items():
            t = r["tags"]
            if t.get("natural") == "water" or t.get("waterway") == "riverbank":
                lines = [LineString(self.osm.way_xy(ref)) for typ, ref, role in r["members"]
                         if typ == "way" and role == "outer" and ref in self.osm.ways and len(self.osm.way_xy(ref)) > 1]
                for p in polygonize(linemerge(lines) if lines else []):
                    out.append(p)
        self._water = [p for p in out if p.intersects(self.zone.buffer(60)) and p.area > 30]
        return self._water

    # ------------------------------------------------------------------ buildings
    def build_buildings(self):
        sc = self.sc
        B = pickle.load(open(os.path.join(self.data, "buildings.pkl"), "rb"))
        corr = unary_union([tr.line.buffer(3.6) for tr in self.tracks])
        keepout = unary_union([corr, self.U.buffer(0.6)])
        # expressway decks
        exp = [o[0] for o in sc.obstacles if o[3] == "expressway"]
        if exp:
            keepout = unary_union([keepout] + [e.buffer(0.5) for e in exp])
        ko_tree = keepout
        nk = nc = nd = 0
        for b in B:
            fp = Polygon(b["footprint"]).buffer(0)
            if fp.is_empty or not fp.intersects(self.zone):
                continue
            facade = self.facade_for(b)
            if not fp.intersects(ko_tree):
                self.emit_lod2(b, facade); nk += 1
                continue
            rest = fp.difference(ko_tree).buffer(-1.0).buffer(1.0)
            if rest.area < 40:
                nd += 1; continue
            top = b["zmax"]
            for q in mk.polys(rest):
                if q.area < 25:
                    continue
                mk.prism(sc, "roof", facade, segmentize(q, 6.0), top, b["zmin"] - 0.5, side_v0=b["zmin"])
                sc.obstacle(q, b["zmin"], top, "building")
            nc += 1
        self.log.append("buildings kept %d, clipped %d, dropped %d" % (nk, nc, nd))

    def facade_for(self, b):
        h = b["zmax"] - b["zmin"]
        name = b.get("name") or ""
        c = b["footprint"].mean(0)
        if "東京駅" in name and c[0] < -40:
            return "facade_station_brick"
        r = hsh(b["id"]) % 100
        if h > 60:
            return ["facade_glass", "facade_glass", "facade_office_tile"][r % 3]
        if h > 25:
            return ["facade_office_tile", "facade_rect", "facade_glass", "facade_plaster_win"][r % 4]
        return ["facade_plaster_win", "facade_beige_win", "facade_rect", "facade_shop"][r % 4]

    def emit_lod2(self, b, facade):
        sc = self.sc
        z0 = b["zmin"]
        for kind, ring in b["surfaces"]:
            if kind == "GroundSurface" or len(ring) < 4:
                continue
            r = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
            if len(r) < 3:
                continue
            # Newell normal
            nrm = np.zeros(3)
            for i in range(len(r)):
                a, c = r[i], r[(i + 1) % len(r)]
                nrm += [(a[1] - c[1]) * (a[2] + c[2]), (a[2] - c[2]) * (a[0] + c[0]), (a[0] - c[0]) * (a[1] + c[1])]
            ln = np.linalg.norm(nrm)
            if ln < 1e-6:
                continue
            nrm /= ln
            ax = int(np.argmax(np.abs(nrm)))
            uvax = [i for i in range(3) if i != ax]
            pts2 = r[:, uvax]
            try:
                tri = mk.earcut.triangulate_float64(pts2, np.array([len(pts2)], np.uint32)).reshape(-1, 3)
            except Exception:
                continue
            if len(tri) == 0:
                continue
            is_wall = kind == "WallSurface" or (kind == "Mixed" and abs(nrm[2]) < 0.3)
            if is_wall:
                hd = np.array([-nrm[1], nrm[0], 0.0]); hd /= np.linalg.norm(hd) + 1e-9
                uv = np.c_[r @ hd, r[:, 2] - z0]
                mat = facade
            else:
                uv = r[:, :2]
                mat = "roof"
            sc.b(mat).add(r, tri, uv)
        fp = Polygon(b["footprint"]).buffer(0)
        sc.obstacle(fp, b["zmin"], b["zmax"], "building")

    # ------------------------------------------------------------------ run
    def build(self):
        self.build_track()
        self.build_platforms()
        self.build_structures()
        self.platform_meshes()
        self.platform_furniture()
        self.build_parapets()
        self.build_expressways()
        self.build_catenary()
        self.build_ground()
        self.build_buildings()
        return self

    def camera_path(self):
        s = np.arange(self.drive[0], self.drive[1], 0.5)
        xy = np.array([self.main.interpolate(v).coords[0] for v in s])
        z = np.interp(s, self.net.ms, self.net.base_h)
        return np.c_[xy, z], s

    def export(self, out):
        meshes = {}
        for mat, bk in self.sc.buckets.items():
            a = bk.arrays()
            if a is not None:
                meshes[mat] = a
        cam, s = self.camera_path()
        stations = [(n, en, p.tolist()) for n, en, p in self.station_names()]
        pickle.dump(dict(meshes=meshes, camera=cam, cam_s=s, stations=stations, log=self.log), open(out, "wb"))


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    m = Model(data).build()
    import validate
    rep = validate.check(m)
    m.log += rep
    m.export(out)
    print("\n".join(m.log))
