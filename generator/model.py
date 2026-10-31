"""Build the corridor model: structures, track, platforms, stations, catenary, roads, buildings.

All placement goes through explicit clearance rules (see validate.py), so masts, piers,
canopies and buildings never enter a track's kinematic envelope, and nothing that should
sit on a structure floats or sinks below ground.
"""
import json, math, os, pickle, sys, zlib
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
RIBBON = 3.2                       # deck half width around a track
PLAT_EDGE = 1.58                   # platform edge from track centre
PLAT_H = 1.10                      # platform top above rail
CANOPY_H = 3.45                    # canopy underside above platform
CONTACT_H = 5.10
MESSENGER_H = 6.20
BEAM_H = 7.30
SPAN = 45.0
SLAB = 1.1                         # slab thickness of open spans
DETAIL_DIST = 25.0                 # tracks closer than this to the driven line get sleepers
BRICK_ZONE = (-1000.0, -2000.0, 800.0, 2000.0)   # Shimbashi - Akihabara brick arch viaducts (local x0, y0, x1, y1)


PALETTE = [  # (weight, sRGB) - mostly greys/whites, some cream/yellow, few red-orange tile and browns
    (14, (0.90, 0.90, 0.88)), (12, (0.82, 0.82, 0.80)), (12, (0.72, 0.72, 0.71)), (10, (0.60, 0.60, 0.59)),
    (8, (0.50, 0.50, 0.50)), (8, (0.88, 0.86, 0.81)), (6, (0.78, 0.76, 0.72)),
    (6, (0.90, 0.82, 0.62)), (5, (0.85, 0.76, 0.56)), (4, (0.93, 0.87, 0.70)),
    (4, (0.72, 0.40, 0.27)), (3, (0.64, 0.31, 0.22)), (3, (0.80, 0.50, 0.32)),
    (3, (0.47, 0.35, 0.27)), (2, (0.36, 0.29, 0.23))]


def pick_colour(rng):
    w = np.array([p[0] for p in PALETTE], float)
    c = np.array(PALETTE[int(rng.choice(len(PALETTE), p=w / w.sum()))][1])
    return np.clip(c * rng.uniform(0.94, 1.04) + rng.normal(0, 0.012, 3), 0, 1)


def srgb_lin(c):
    return tuple(float(x) ** 2.2 for x in c)


def runs(mask):
    """Index arrays of consecutive True runs (each run extended by one point to close gaps)."""
    out, cur = [], []
    for i, v in enumerate(mask):
        if v:
            cur.append(i)
        elif cur:
            cur.append(i); out.append(np.array(cur)); cur = []
    if cur:
        out.append(np.array(cur))
    return out


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
    def __init__(self, data, s_range=None, drive=None):
        self.data = data
        self.osm = OSM(os.path.join(data, "osm.json"))
        ext = os.path.join(data, "extent.json")
        bbox = json.load(open(ext))["bbox"] if os.path.exists(ext) else (139.756, 35.672, 139.784, 35.706)
        self.dem = DEM(*bbox, os.path.join(data, "dem"))
        self.net = Network(self.osm, self.dem, 1972960, s_range or (0, 1e9))
        drive = drive or (0.0, self.net.main.length)
        self.tracks = self.net.tracks
        self.idx = TrackIdx(self.tracks)
        self.sc = mk.Scene()
        self.drive = drive
        self.main = self.net.main
        self.zone = self.net.zone
        L = self.main.length
        self.drive_line = substring(self.main, max(0.0, drive[0] - 50), min(L, drive[1] + 50))   # (negative offsets would wrap)
        for tr in self.tracks:
            tr.near_main = tr.line.distance(self.drive_line) < DETAIL_DIST
        self.log = []
        self.level_stations()

    # ------------------------------------------------------------------ helpers
    def lower_tracks(self, x, y, z, r, dz=0.5):
        return [h for h in self.idx.near(x, y, r) if h[2] < z - dz]

    def lower_under(self, fp, ti, zc, dz=0.45):
        """Tracks lower than zc - dz anywhere within 3 m of a deck footprint: [(track, distance, z)]."""
        found = {}
        for c in self.idx.tree.query(fp.buffer(3.0), predicate="intersects"):
            t2, i = self.idx.meta[c]
            if t2 == ti:
                continue
            z = float(self.tracks[t2].pts[i:i + 2, 2].min())
            if z < zc - dz:
                found[t2] = (t2, 0.0, z)
        return list(found.values())

    def clear_of_tracks(self, x, y, z0, z1, r=2.45):
        """True if a vertical member at (x,y) spanning z0..z1 stays out of every track envelope."""
        for ti, d, z in self.idx.near(x, y, r):
            if z - 0.5 < z1 and z + 6.0 > z0:
                return False
        return True

    def level_stations(self):
        """Second height pass: every platform span and the track across its platform get one common level."""
        spans = self.platform_spans()
        level = {}
        for ti, s0, s1, side in spans:
            tr = self.tracks[ti]
            sm = (s0 + s1) / 2
            q = np.array(tr.line.interpolate(sm).coords[0])
            a = np.array(tr.line.interpolate(max(0, sm - 1)).coords[0]); b = np.array(tr.line.interpolate(min(tr.line.length, sm + 1)).coords[0])
            d = (b - a) / max(1e-9, np.linalg.norm(b - a)); n = np.array([-d[1], d[0]]) * side
            zt = float(np.interp(sm, mk.cumlen(tr.pts), tr.pts[:, 2]))
            group = [ti]
            ray = LineString([q + n * 0.6, q + n * 22.0])
            best = None
            for c in self.idx.tree.query(ray, predicate="intersects"):
                t2, i2 = self.idx.meta[c]
                if t2 != ti and abs(self.tracks[t2].pts[i2, 2] - zt) < 3.0:
                    dd = Point(q).distance(self.idx.segs[c].intersection(ray))
                    if best is None or dd < best[0]:
                        best = (dd, t2)
            if best:
                group.append(best[1])
            # the whole connected run of ways alongside the platform
            area = LineString([tr.line.interpolate(v).coords[0] for v in np.linspace(s0, s1, 20)]).buffer(25.0)
            members = [t for t in range(len(self.tracks)) if t in group or
                       (self.tracks[t].line.intersects(area) and any(self.tracks[t].line.distance(self.tracks[g].line) < 0.5 and
                        self.tracks[t].tags.get("name") == self.tracks[g].tags.get("name") for g in group))]
            zs = [float(self.tracks[t].pts[contains_xy(area, self.tracks[t].pts[:, 0], self.tracks[t].pts[:, 1]), 2].mean())
                  for t in members if contains_xy(area, self.tracks[t].pts[:, 0], self.tracks[t].pts[:, 1]).any()]
            zl = float(np.mean(zs)) if zs else zt
            for t in members:
                for nd in self.tracks[t].nodes:
                    if contains_xy(area, *self.osm.xy(nd)):
                        level[nd] = zl
        self.net.solve_heights(level)
        self.idx = TrackIdx(self.tracks)
        self.log.append("station levelling: %d nodes pinned" % len(level))

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
                    if not (tr.main or any(k in tr.name for k in ("山手線", "京浜東北線"))):
                        continue
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
                low = self.lower_under(fp, ti, zc)
                if low:
                    cut = unary_union([self.tracks[t2].line.buffer(2.25) for t2, _, _ in low])
                    fp = fp.difference(cut)
                    mode = "over"
                elif tr.bridge and zc - float(self.dem(*mid[:2])) > 4.0:
                    mode = "girder"
                else:
                    mode = "solid"
                if fp.is_empty or fp.area < 1.0:
                    continue
                rel = zc - float(self.dem(*mid[:2]))
                x0, y0, x1, y1 = BRICK_ZONE
                brick = (mode == "solid" and rel > 4.5 and any(b in tr.name for b in brick_lines)
                         and x0 < mid[0] < x1 and y0 < mid[1] < y1)
                mat = "brick" if brick else ("concrete" if rel > 4.5 else "retaining")
                pieces.append(dict(ti=ti, fp=fp, sub=sub, mode=mode, z=zc, low=[t2 for t2, _, _ in low], mat=mat,
                                   covered=bool(tr.tags.get("tunnel"))))
        self.pieces = pieces
        self.deck_union = unary_union([pc["fp"] for pc in pieces])
        # covered sections: roof slab over the track bed (soffit 7.2 m above rail), walls come from the boundary pass
        roofs = []
        for pc in pieces:
            if not pc["covered"]:
                continue
            fp = pc["fp"].buffer(0.4)
            z0 = pc["z"] + 7.2
            hit = set()
            for c in self.idx.tree.query(fp.buffer(2.0), predicate="intersects"):
                t2, i = self.idx.meta[c]
                z2 = float(self.tracks[t2].pts[i:i + 2, 2].max())
                same_tube = self.tracks[t2].tags.get("tunnel") and abs(z2 - pc["z"]) < 1.0
                if t2 != pc["ti"] and not same_tube and z2 + 5.8 > z0:
                    hit.add(t2)
            if hit:
                fp = fp.difference(unary_union([self.tracks[t2].line.buffer(2.0) for t2 in hit]))
            if fp.is_empty or fp.area < 2.0:
                continue
            g = float(self.dem(*pc["sub"][len(pc["sub"]) // 2][:2]))
            mk.prism(sc, "concrete", "concrete", segmentize(fp, 3.0), max(z0 + 1.0, g), z0, bottom=True)
            sc.obstacle(fp, z0, max(z0 + 1.0, g), "slab")
            roofs.append(dict(ti=-1, fp=fp, sub=pc["sub"], mode="roof", z=z0 + DECK + SLAB, low=[], mat="concrete", covered=True))
        self.pieces = pieces + roofs
        self.piece_tree = STRtree([pc["fp"] for pc in self.pieces])

        for pc in pieces:
            if pc["mode"] == "roof":
                continue
            sub = pc["sub"]
            line = LineString(sub[:, :2])
            sd = mk.cumlen(sub)

            def ztop(xy, sub=sub, line=line, sd=sd):
                s = np.array([line.project(Point(a, b)) for a, b in xy])
                return np.interp(s, sd, sub[:, 2]) - DECK

            fp = segmentize(pc["fp"], 2.5)
            if pc["mode"] == "solid":
                zb = lambda xy, zt=ztop: np.minimum(self.dem(xy[:, 0], xy[:, 1]) - 0.4, zt(xy) - 0.6)
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
                if not self.clear_of_tracks(x, y, g, top, r=2.95):
                    continue
                mk.boxes(self.sc, "concrete", [[x, y, (top + g) / 2]], t[k], left[k], [1.4, 1.4, top - g])
                self.sc.obstacle(Point(x, y).buffer(1.0, quad_segs=2), g, top, "pier")
                break

    def build_parapets(self):
        from shapely.geometry.polygon import orient
        sc = self.sc
        U = unary_union([self.deck_union] + [p for p, _ in self.platforms])
        U = U.buffer(0.05).buffer(-0.05)
        self.U = U
        lines = []
        plat_union = unary_union([p for p, _ in self.platforms]) if self.platforms else Polygon()
        n_ret = 0
        for poly in mk.polys(U):
            poly = orient(poly, 1.0)       # interior on the left of every ring
            for ring in [poly.exterior] + list(poly.interiors):
                if ring.length < 6:
                    continue
                lines.append(ring)
                r = np.asarray(segmentize(LineString(ring.coords), 2.0).coords)
                if len(r) < 3:
                    continue
                t = np.gradient(r, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
                out = np.c_[t[:, 1], -t[:, 0]]                     # right of travel = outside
                z = self.idx.nearest_z(r) - DECK
                onplat = np.array([plat_union.distance(Point(a, b)) < 0.4 for a, b in r])
                pz = np.array([self.plat_top_at(a, b) for a, b in r])
                base = np.where(onplat & ~np.isnan(pz), pz, z)
                g_out = self.dem(*(r + out * 1.5).T)
                cut = g_out > base + 0.8
                top = np.where(cut, g_out + 1.1, base + 1.05)
                for m, mat in ((cut, "retaining"), (~cut, "parapet")):
                    for run in runs(m):
                        if len(run) < 2:
                            continue
                        self.wall_strip(r[run], base[run] - (0.2 if mat == "retaining" else 0.0), top[run], out[run], mat)
                        n_ret += mat == "retaining"
        self.parapet_lines = lines
        self.log.append("boundary wall runs: %d retaining" % n_ret)

    def wall_strip(self, r, zb, zt, out, mat, thick=0.25):
        """Wall of varying height along points r (N,2), standing on the boundary and extending outward."""
        n = len(r)
        inner = r; outer = r + out * thick
        V = np.concatenate([np.c_[inner, zb], np.c_[inner, zt], np.c_[outer, zt], np.c_[outer, zb]])
        i = np.arange(n - 1)
        F = np.concatenate([np.stack([i, i + 1, i + 1 + n, i + n], 1),                  # inner face
                            np.stack([i + n, i + 1 + n, i + 1 + 2 * n, i + 2 * n], 1),  # top
                            np.stack([i + 2 * n, i + 1 + 2 * n, i + 1 + 3 * n, i + 3 * n], 1)])  # outer face
        sd = mk.cumlen(np.c_[r, np.zeros(n)])
        uv = np.concatenate([np.c_[sd, zb], np.c_[sd, zt], np.c_[sd, zt + thick], np.c_[sd, zb]])
        self.sc.b(mat).add(V, F, uv)

    def build_arches(self):
        """Arcade relief on brick viaducts: pilasters, arch rings and infill (shop fronts / bricked bays)."""
        sc = self.sc
        BAY = 7.6
        n_arch = 0
        for ring in self.parapet_lines:
            ln = LineString(ring.coords)
            for d0 in np.arange(BAY / 2, ln.length - BAY / 2, BAY):
                c = np.array(ln.interpolate(d0).coords[0])
                a = np.array(ln.interpolate(d0 - 1.0).coords[0]); b = np.array(ln.interpolate(d0 + 1.0).coords[0])
                tdir = (b - a) / max(1e-9, np.linalg.norm(b - a))
                # straight enough over the whole bay?
                e0 = np.array(ln.interpolate(d0 - BAY / 2).coords[0]); e1 = np.array(ln.interpolate(d0 + BAY / 2).coords[0])
                if np.linalg.norm(e1 - e0) < BAY * 0.97:
                    continue
                nrm = np.array([tdir[1], -tdir[0]])
                if contains_xy(self.U, *(c + nrm * 0.5)):
                    nrm = -nrm
                hit = [self.pieces[i] for i in self.piece_tree.query(Point(c).buffer(0.6))]
                if not any(pc["mode"] == "solid" and pc["mat"] == "brick" and pc["fp"].exterior.distance(Point(c)) < 0.6 for pc in hit):
                    continue
                zt = float(self.idx.nearest_z(c[None], r=8.0)[0]) - DECK
                g = float(self.dem(*c))
                H = zt - g
                if H < 4.5:
                    continue
                r = 3.0
                hs = max(1.8, min(H - 0.9 - r, 3.8))
                ry = min(r, H - 0.9 - hs)
                if ry < 0.8:
                    continue
                ang = np.linspace(0, np.pi, 17)
                arc = np.c_[np.cos(ang) * r, hs + np.sin(ang) * ry]
                opening = np.vstack([[[-r, 0.0], [r, 0.0]], arc])           # (u, v) polygon
                band = np.vstack([np.c_[np.cos(ang) * (r + 0.45), hs + np.sin(ang) * (ry + 0.45)], arc[::-1]])
                T3 = np.array([tdir[0], tdir[1], 0.0]); N3 = np.array([nrm[0], nrm[1], 0.0])
                base = np.array([c[0], c[1], g])

                def put(mat, poly, off, col=None):
                    tri = mk.earcut.triangulate_float64(poly, np.array([len(poly)], np.uint32)).reshape(-1, 3)
                    V = base + T3 * poly[:, :1] + np.array([0, 0, 1.0]) * poly[:, 1:2] + N3 * off
                    sc.b(mat).add(V, tri, poly, col)

                rng = np.random.default_rng(hsh((round(c[0]), round(c[1]))))
                kind = rng.random()
                col = srgb_lin((0.10, 0.11, 0.12)) if kind < 0.45 else (srgb_lin((0.48, 0.47, 0.45)) if kind < 0.7 else srgb_lin((0.75, 0.62, 0.45)))
                put("arch_infill", opening, 0.04, col)
                put("brick_dark", band, 0.08)
                for side in (-1, 1):
                    mk.boxes(sc, "brick", [base + T3 * side * BAY / 2 + N3 * 0.15 + [0, 0, (H - 0.4) / 2]], T3, N3, [1.0, 0.3, H - 0.4])
                mk.boxes(sc, "brick_dark", [base + N3 * 0.12 + [0, 0, H - 0.25]], T3, N3, [BAY, 0.25, 0.3])
                n_arch += 1
        self.log.append("brick arches: %d" % n_arch)

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
                zone = line.buffer(2.6)
                for ti, tr in enumerate(self.tracks):
                    if tr.line.distance(line) > 2.6:
                        continue
                    part = tr.line.intersection(zone)
                    if part.length < 6:
                        continue
                    ends = [Point(c) for g in getattr(part, "geoms", [part]) for c in (g.coords[0], g.coords[-1])]
                    ss = [tr.line.project(e) for e in ends]
                    mid = tr.line.interpolate((min(ss) + max(ss)) / 2)
                    # the platform is on the side of the edge line away from the track
                    q = line.interpolate(line.project(mid))
                    spans.append((ti, min(ss), max(ss), side_of(tr, q)))
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
                    if seg.length < 6:
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
            c = P.representative_point()
            zt = float(self.idx.nearest_z(np.array([c.coords[0]]), r=25)[0])
            cut = []
            for tr in self.tracks:
                if tr.line.distance(P) > 3.0:
                    continue
                near = contains_xy(P.buffer(3.0), tr.pts[:, 0], tr.pts[:, 1])
                zq = tr.pts[near, 2] if near.any() else tr.pts[:, 2]
                if np.any((zq - zt > -7.0) & (zq - zt < 6.0)):   # envelope overlaps the platform / canopy band
                    cut.append(tr.line.buffer(PLAT_EDGE - 0.001))
            cutp.append(P.difference(unary_union(cut)) if cut else P)
        allp = unary_union(cutp)
        self.platforms = []
        queue = [P for P in mk.polys(allp) if P.area >= 40]
        while queue:
            P = queue.pop()
            # final safety cut against every track whose envelope overlaps this platform's height band
            for _ in range(2):
                zl = self.idx.nearest_z(np.asarray(P.exterior.coords)[:, :2], r=25.0)
                z_lo, z_hi = float(zl.min()), float(zl.max())
                cut = []
                for tr in self.tracks:
                    if tr.line.distance(P) > 3.0:
                        continue
                    near = contains_xy(P.buffer(3.0), tr.pts[:, 0], tr.pts[:, 1])
                    zq = tr.pts[near, 2] if near.any() else tr.pts[:, 2]
                    if np.any((zq > z_lo - 7.0) & (zq < z_hi + 6.0)):
                        cut.append(tr.line.buffer(PLAT_EDGE - 0.001))
                if cut:
                    P = P.difference(unary_union(cut))
                parts = sorted(mk.polys(P), key=lambda q: q.area)
                if not parts:
                    P = Polygon(); break
                P = parts.pop()
                queue += [q for q in parts if q.area >= 40]   # other pieces are re-checked on their own
            if P.is_empty or P.area < 40:
                continue
            c = P.representative_point()
            ti = min(polys, key=lambda pp: pp[0].distance(c))[1]
            self.platforms.append((P, ti))
        self.span_geo = span_geo
        self.plat_tree = STRtree([p for p, _ in self.platforms]) if self.platforms else None
        self.log.append("platforms: %d polygons from %d spans" % (len(self.platforms), len(spans)))

    def cells(self, P, size=3.0):
        from shapely.geometry import box
        x0, y0, x1, y1 = P.bounds
        out = []
        for x in np.arange(x0, x1, size):
            for y in np.arange(y0, y1, size):
                c = P.intersection(box(x, y, x + size, y + size))
                if not c.is_empty and c.area > 0.01:
                    out.append(c)
        return out

    def local_obstacles(self, P, zref, dz0, dz1, label):
        """Register a polygon as obstacle cells whose heights follow the local rail level."""
        for c in self.cells(P):
            pts = np.vstack([np.asarray(c.exterior.coords)[:, :2], np.array(c.representative_point().coords)]) if c.geom_type == "Polygon" \
                else np.array(c.representative_point().coords)
            z = self.plat_z(pts, zref)
            self.sc.obstacle(c, float(z.min()) + dz0, float(z.max()) + dz1, label)
            self.cell_zref = getattr(self, "cell_zref", {})
            self.cell_zref[len(self.sc.obstacles) - 1] = zref   # validator re-evaluates the exact surface

    def plat_top_at(self, x, y):
        if not self.plat_tree:
            return np.nan
        for c in self.plat_tree.query(Point(x, y), predicate="dwithin", distance=0.5):
            P, ti = self.platforms[c]
            return float(self.plat_z(np.array([[x, y]]), self.plat_ref(P, ti))[0]) + PLAT_H
        return np.nan

    def plat_ref(self, P, ti):
        """Rail level of the platform's own track next to this platform (local, not the whole way's mean)."""
        tr = self.tracks[ti]
        sd = mk.cumlen(tr.pts)
        return float(np.interp(tr.line.project(P.representative_point()), sd, tr.pts[:, 2]))

    def plat_z(self, xy, zref):
        """Platform reference rail level: the lowest same-level track within 9 m (never above any adjacent rail)."""
        out = np.empty(len(xy))
        for k, (x, y) in enumerate(np.asarray(xy)[:, :2]):
            hits = [h[2] for h in self.idx.near(x, y, 9.0) if abs(h[2] - zref) < 2.5]
            out[k] = min(hits) if hits else zref
        return out

    def platform_meshes(self):
        sc = self.sc
        lowclear_all = []
        for P, ti in self.platforms:
            zref = self.plat_ref(P, ti)
            Pd = segmentize(P, 2.0)
            ztop = lambda xy, zr=zref: self.plat_z(xy, zr) + PLAT_H
            zslab = lambda xy, zr=zref: self.plat_z(xy, zr) - DECK - SLAB
            mk.prism(sc, "platform", "platform_side", Pd, ztop, zslab, bottom=True)
            self.local_obstacles(P, zref, -DECK - SLAB, PLAT_H, "platform")
            zt = zref
            # fill below the platform slab down to ground, except above lower tracks
            low = [tr.line.buffer(2.6) for tr in self.tracks if tr.line.distance(P) < 3 and tr.pts[:, 2].mean() < zt - 1.5]
            fill = P.difference(unary_union(low)) if low else P
            for q in mk.polys(fill.buffer(-0.02)):
                mk.prism(sc, "concrete", "concrete", segmentize(q, 3.0), zslab, lambda xy: self.dem(xy[:, 0], xy[:, 1]) - 0.4)

    def platform_furniture(self):
        sc = self.sc
        self.plat_union = unary_union([p for p, _ in self.platforms]) if self.platforms else Polygon()
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
            on = contains_xy(self.plat_union, *(ii[:, :2] + L[:, :2] * 0.6).T)
            for run in runs(on):
                if len(run) < 3:
                    continue
                mk.sweep(sc, "edge_white", ii[run] + [0, 0, 0.004], [(0.0, 0), (0.12, 0)], left=L[run])
                mk.sweep(sc, "tactile", ii[run] + [0, 0, 0.006], [(0.80, 0), (1.10, 0)], left=L[run])
            psd = any(k in tr.name for k in ("山手線", "京浜東北線"))
            if psd and len(ii) > 10:
                for run in runs(on):
                    if len(run) > 10:
                        self.psd(ii[run], L[run])
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
            zt = float(np.max(self.idx.nearest_z(np.asarray(P.exterior.coords)[::3, :2], zref=self.tracks[ti].pts[:, 2].mean(), r=25.0))) + PLAT_H
            roof = P.buffer(-0.12)
            # trim the roof wherever it would hang inside any track's car-body envelope
            cut = [tr.line.buffer(1.85) for tr in self.tracks if tr.line.distance(P) < 4.0 and
                   np.any(contains_xy(P.buffer(4.0), tr.pts[:, 0], tr.pts[:, 1]) &
                          (tr.pts[:, 2] + 4.15 > zt + CANOPY_H) & (tr.pts[:, 2] - 1.0 < zt + CANOPY_H + 0.4))]
            if cut:
                roof = roof.difference(unary_union(cut))
            cols = [c for c, z in self.columns if contains_xy(P, c[0], c[1])]
            if not cols or roof.is_empty:
                continue
            # limit canopy to the column extent (+6 m)
            hull = unary_union([Point(c).buffer(9.0) for c in cols])
            roof = roof.intersection(hull.convex_hull if len(cols) < 3 else unary_union([LineString([a, b]).buffer(9.0) for a, b in zip(cols[:-1], cols[1:])] + [hull]))
            for q in mk.polys(roof):
                mk.prism(sc, "canopy_top", "canopy_fascia", segmentize(q, 3.0), zt + CANOPY_H + 0.35, zt + CANOPY_H, bottom=False)
                mk.prism(sc, "canopy_under", "canopy_fascia", segmentize(q, 3.0), zt + CANOPY_H, zt + CANOPY_H - 0.001, top_only=True)
                for cell in self.cells(q):
                    sc.obstacle(cell, zt + CANOPY_H, zt + CANOPY_H + 0.35, "canopy")
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
                fp = Polygon([cc[:2] + ax[:2] * a * 5.6 + ay[:2] * b * 1.45 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
                if not P.buffer(-1.0).contains(fp):
                    continue
                mk.boxes(sc, "stair_pit", [cc + [0, 0, 0.005]], ax, ay, [11.0, 2.6, 0.01])
                for sd in (-1, 1):
                    mk.boxes(sc, "psd_frame", [cc + ay * sd * 1.35 + [0, 0, 0.55]], ax, ay, [11.0, 0.1, 1.1])
                mk.boxes(sc, "psd_frame", [cc - ax * 5.5 + [0, 0, 0.55]], ax, ay, [0.1, 2.7, 1.1])
                sc.obstacle(fp, zt, zt + 1.1, "stairs")

    def psd(self, ii, L):
        """Platform screen doors: 5 m modules = fixed panel (2.9 m) + bi-parting door (2 x 1.0 m)."""
        sc = self.sc
        s = mk.cumlen(ii)

        def at(v, off):
            p = np.c_[[np.interp(v, s, ii[:, j]) for j in range(3)]].T
            l = np.c_[[np.interp(v, s, L[:, j]) for j in range(3)]].T
            l /= np.linalg.norm(l, axis=1, keepdims=True) + 1e-9
            t = np.c_[l[:, 1], -l[:, 0], np.zeros(len(l))]
            return p + l * off, t, l

        m0 = np.arange(1.5, s[-1] - 6.0, 5.0)
        if len(m0) == 0:
            return
        p, t, l = at(m0 + 1.45, 0.42)                      # fixed panels
        mk.boxes(sc, "psd_frame", p + [0, 0, 0.45], t, l, [2.9, 0.12, 0.9])
        mk.boxes(sc, "psd_glass", p + [0, 0, 1.1], t, l, [2.8, 0.04, 0.4])
        for dx in (0.5, 1.5):                              # door leaves (closed)
            q, tq, lq = at(m0 + 2.9 + dx, 0.36)
            mk.boxes(sc, "psd_door", q + [0, 0, 0.62], tq, lq, [0.97, 0.06, 1.24])
        for dx in (0.0, 2.9, 4.95):                        # posts
            q, tq, lq = at(m0 + dx, 0.42)
            mk.boxes(sc, "psd_frame", q + [0, 0, 0.67], tq, lq, [0.12, 0.2, 1.34])
        q, tq, lq = at(m0 + 1.45, 0.42)
        mk.boxes(sc, "psd_frame", q + [0, 0, 1.33], tq, lq, [2.95, 0.16, 0.06])
        keep = contains_xy(self.plat_union.buffer(-0.05), *at(m0 + 2.5, 0.40)[0][:, :2].T)
        if not keep.all():   # drop modules that would stand off the final platform surface
            m0 = m0[keep]
            if len(m0) == 0:
                return
            return self.psd_modules(at, m0)
        pc, tc, lc = at(m0 + 2.5, 0.40)
        for c, tq, lq in zip(pc, tc, lc):
            c4 = [c[:2] + tq[:2] * a * 2.5 + lq[:2] * b * 0.11 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            sc.obstacle(Polygon(c4), c[2], c[2] + 1.4, "psd")

    def psd_modules(self, at, m0):
        sc = self.sc
        p, t, l = at(m0 + 1.45, 0.42)
        mk.boxes(sc, "psd_frame", p + [0, 0, 0.45], t, l, [2.9, 0.12, 0.9])
        mk.boxes(sc, "psd_glass", p + [0, 0, 1.1], t, l, [2.8, 0.04, 0.4])
        for dx in (0.5, 1.5):
            q, tq, lq = at(m0 + 2.9 + dx, 0.36)
            mk.boxes(sc, "psd_door", q + [0, 0, 0.62], tq, lq, [0.97, 0.06, 1.24])
        for dx in (0.0, 2.9, 4.95):
            q, tq, lq = at(m0 + dx, 0.42)
            mk.boxes(sc, "psd_frame", q + [0, 0, 0.67], tq, lq, [0.12, 0.2, 1.34])
        q, tq, lq = at(m0 + 1.45, 0.42)
        mk.boxes(sc, "psd_frame", q + [0, 0, 1.33], tq, lq, [2.95, 0.16, 0.06])
        pc, tc, lc = at(m0 + 2.5, 0.40)
        for c, tq, lq in zip(pc, tc, lc):
            c4 = [c[:2] + tq[:2] * a * 2.5 + lq[:2] * b * 0.11 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            sc.obstacle(Polygon(c4), c[2], c[2] + 1.4, "psd")

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
        self.sup_pts = []
        main = self.main
        s0, s1 = self.drive[0] - 400, self.drive[1] + 400
        col_pts = [c for c, _ in self.columns]
        nmast = 0
        for s in np.arange(max(0, s0), min(main.length, s1), SPAN):
            need = None
            for ds in (0, 4, -4, 8, -8, 12, -12):
                cov, hit = self.gantry(s + ds, supports, col_pts, need)
                nmast += bool(cov)
                rem = (hit if need is None else need) - cov
                if not rem:
                    break
                need = rem
        # second pass: close support gaps longer than 60 m on tracks seen from the driven line
        region = self.drive_line.buffer(45)
        near = {}
        for ti, tr in enumerate(self.tracks):
            inside = contains_xy(region, tr.pts[:, 0], tr.pts[:, 1])
            if inside.sum() >= 20:   # at least ~40 m of this track is seen from the driven line
                sd = mk.cumlen(tr.pts)[inside]
                near[ti] = (float(sd.min()), float(sd.max()))

        def marks_of(ti):
            lo, hi = near[ti]
            ss = sorted(a for a, _ in supports[ti] if lo <= a <= hi)
            return [lo] + ss + [hi]

        for _ in range(3):
            for ti in near:
                tr = self.tracks[ti]
                marks = marks_of(ti)
                for a0, a1 in zip(marks[:-1], marks[1:]):
                    if a1 - a0 <= 60.0:
                        continue
                    for frac in ((0.5,) if a1 - a0 < 110 else (1 / 3, 2 / 3)):
                        sm = main.project(tr.line.interpolate(a0 + (a1 - a0) * frac))
                        for ds in (0, 4, -4, 8, -8, 12, -12, 16, -16, 20, -20):
                            if ti in self.gantry(sm + ds, supports, col_pts, need={ti})[0]:
                                nmast += 1
                                break
        gaps = []
        for ti in near:
            marks = marks_of(ti)
            gaps.append(max(b - a for a, b in zip(marks[:-1], marks[1:])))
        self.supports = supports
        self.gap_report = sorted(((g, self.tracks[ti].name, self.tracks[ti].wid) for g, ti in zip(gaps, near)), reverse=True)[:8]
        self.log.append("tracks near drive with a support gap > 65 m: %d of %d" % (sum(g > 65 for g in gaps), len(gaps)))
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
        self.log.append("gantries placed: %d, rejects %s" % (nmast, getattr(self, "why", {})))

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

    def soffit(self, x, y, z):
        """Height (relative to rail z) of the lowest structure soffit above (x,y), or None."""
        best = None
        for c in self.piece_tree.query(Point(x, y)):
            pc = self.pieces[c]
            if pc["z"] - z > 3.0 and contains_xy(pc["fp"], x, y):
                h = pc["z"] - DECK - SLAB - z
                best = h if best is None else min(best, h)
        return best

    def gantry(self, s, supports, col_pts, need=None):
        sc = self.sc
        main = self.main
        if s < 1 or s > main.length - 1:
            return set(), set()
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
            return set(), set()
        hits.sort()
        hit_set = {h[1] for h in hits}
        cov = set()
        zig = 0.2 if (int(s / SPAN) % 2) else -0.2
        ok_any = False

        def register(t, ti, z, top):
            x, y = P + n * t
            supports[ti].append((self.tracks[ti].line.project(Point(x, y)), zig))
            self.sup_pts.append((x, y, z))
            cov.add(ti)
            mk.segment_boxes(sc, "mast", [[x, y, top]], [[x, y, z + MESSENGER_H + 0.25]], 0.06, 0.06)
            mk.segment_boxes(sc, "mast", [[x, y, z + MESSENGER_H + 0.2]], [[x + n[0] * zig, y + n[1] * zig, z + CONTACT_H + 0.05]], 0.035, 0.035)
            mk.cylinders(sc, "insulator", [[x, y, z + MESSENGER_H + 0.25]], 0.35, 0.07, sides=8)

        # tracks under a low soffit hang their catenary from the structure above
        rest = []
        for h in hits:
            t, ti, z, i = h
            if need is not None and ti not in need:
                rest.append(h); continue
            x, y = P + n * t
            so = self.soffit(x, y, z)
            if so is not None and so < BEAM_H + 0.8:
                if so > MESSENGER_H + 0.5:
                    register(t, ti, z, z + so)
                    ok_any = True
                continue
            rest.append(h)
        hits = rest
        if not hits:
            return cov, hit_set
        groups, cur = [], [hits[0]]
        for h in hits[1:]:
            prev = cur[-1]
            if abs(h[2] - prev[2]) < 1.0 and h[0] - prev[0] < 16.0:
                cur.append(h)
            else:
                groups.append(cur); cur = [h]
        groups.append(cur)
        for g in groups:
            if need is not None and not any(h[1] in need for h in g):
                continue
            # already supported close by? (by position, since OSM splits one physical track into many ways)
            sp = np.array(self.sup_pts) if self.sup_pts else np.zeros((0, 3))
            near_sup = [len(sp) > 0 and bool(np.any((np.hypot(sp[:, 0] - (P + n * t)[0], sp[:, 1] - (P + n * t)[1]) < 22.0) & (np.abs(sp[:, 2] - z) < 1.5)))
                        for t, ti, z, i in g]
            if all(near_sup):
                cov |= {h[1] for h in g}
                continue
            ts = [h[0] for h in g]
            zg = float(np.median([h[2] for h in g]))
            # lower the gantry under an overhanging upper deck (but keep >= 6.55 m: lower chord stays above the 6.2 m messenger)
            span_fp = LineString([P + n * (ts[0] - 4.5), P + n * (ts[-1] + 4.5)]).buffer(1.0)
            low = [o[1] for o in self.obstacles_in(span_fp) if o[3] == "slab" and o[1] > zg + 5.5 and o[1] < zg + BEAM_H + 1.0]
            beam_h = min(BEAM_H, min(low) - zg - 0.5) if low else BEAM_H
            if beam_h < 6.55:
                self.why["low_soffit"] = self.why.get("low_soffit", 0) + 1
                continue
            cand = [ts[0] - 2.5, ts[-1] + 2.5]
            for (t1, *_), (t2, *_) in zip(g[:-1], g[1:]):
                if t2 - t1 > 4.8:   # a mast fits between two tracks (>= 2.35 m to each)
                    cand.append((t1 + t2) / 2)
            masts = []
            for t in sorted(cand):
                outer = t < ts[0] or t > ts[-1]
                steps = (0, 0.25, 0.5, 0.8, 1.1, 1.5) if outer else (0, 0.3, -0.3, 0.6, -0.6, 1.2, -1.2)
                for dt in steps:
                    tt = t + (dt if not outer else (-dt if t < ts[0] else dt))
                    x, y = P + n * tt
                    if self.mast_ok(x, y, zg, col_pts, beam_h):
                        masts.append(tt); break
            if not masts:
                self.why["group_fail"] = self.why.get("group_fail", 0) + 1
                continue
            mn, mx = min(masts), max(masts)
            covered = [h for h in g if mn - 9.0 <= h[0] <= mx + 9.0]
            if not covered:
                continue
            lo = min(mn, min(h[0] for h in covered) - 0.6); hi = max(mx, max(h[0] for h in covered) + 0.6)
            beam_z = zg + beam_h
            A = P + n * lo; B = P + n * hi
            beam_fp = LineString([A, B]).buffer(0.3)
            beam_clear = all(self.clear_of_tracks(*(A + (B - A) * u), beam_z - 0.3, beam_z + 0.45, r=2.3)
                             for u in np.linspace(0, 1, max(2, int(np.hypot(*(B - A)) / 1.0))))
            if not beam_clear or self.blocked(beam_fp, beam_z - 0.4, beam_z + 0.6):
                self.why["beam_blocked"] = self.why.get("beam_blocked", 0) + 1
                continue
            for m in masts:
                x, y = P + n * m
                base = self.base_z(x, y, zg)
                mk.cylinders(sc, "mast", [[x, y, base]], beam_z + 0.35 - base, 0.16)
                mk.boxes(sc, "concrete", [[x, y, base + 0.2]], [d[0], d[1], 0], [n[0], n[1], 0], [0.6, 0.6, 0.4])
                sc.obstacle(Point(x, y).buffer(0.2, quad_segs=2), base, beam_z, "mast")
            mk.segment_boxes(sc, "mast", [[A[0], A[1], beam_z + 0.2]], [[B[0], B[1], beam_z + 0.2]], 0.25, 0.35)
            mk.segment_boxes(sc, "mast", [[A[0], A[1], beam_z - 0.15]], [[B[0], B[1], beam_z - 0.15]], 0.12, 0.12)
            sc.obstacle(beam_fp, beam_z - 0.25, beam_z + 0.4, "beam")
            for t, ti, z, i in covered:
                register(t, ti, z, beam_z - 0.2)
            ok_any = True
        return cov, hit_set

    def base_z(self, x, y, zg):
        """Top of whatever structure carries a mast at (x,y): platform, deck or ground."""
        pz = self.plat_top_at(x, y)
        if not np.isnan(pz) and abs(pz - zg - PLAT_H) < 1.5:
            return pz
        return zg - DECK

    def mast_ok(self, x, y, zg, col_pts, beam_h=BEAM_H):
        why = self.why = getattr(self, "why", {})
        if not hasattr(self, "_U_in"):
            self._U_in = self.U.buffer(-0.35)
        if not self.clear_of_tracks(x, y, zg - DECK, zg + beam_h, r=2.35):
            why["track"] = why.get("track", 0) + 1; return False
        if not contains_xy(self._U_in, x, y):
            why["no_deck"] = why.get("no_deck", 0) + 1; return False  # must stand on deck / platform
        for c in col_pts:
            if abs(c[0] - x) < 1.2 and abs(c[1] - y) < 1.2:
                why["column"] = why.get("column", 0) + 1; return False
        if self.blocked(Point(x, y).buffer(0.25), zg - DECK + 0.3, zg + beam_h + 0.4, skip=("platform",)):
            why["obstacle"] = why.get("obstacle", 0) + 1; return False
        return True

    def obstacles_in(self, fp):
        self.blocked(fp, 0, 0)   # refreshes the index
        obs = self.sc.obstacles
        out = [obs[c] for c in self._otree.query(fp, predicate="intersects")]
        return out + [obs[i] for i in range(self._otree_n, len(obs)) if obs[i][0].intersects(fp)]

    def blocked(self, fp, z0, z1, skip=()):
        obs = self.sc.obstacles
        if not hasattr(self, "_otree") or len(obs) - self._otree_n > 300:
            self._otree = STRtree([o[0] for o in obs]); self._otree_n = len(obs)
        cand = list(self._otree.query(fp, predicate="intersects"))
        cand += [i for i in range(self._otree_n, len(obs)) if obs[i][0].intersects(fp)]
        for c in cand:
            poly, a, b, lab = obs[c]
            if lab in skip:
                continue
            if a < z1 and b > z0:
                return True
        return False

    def build_overpasses(self):
        sc = self.sc
        widths = {"trunk": 10.0, "primary": 9.0, "secondary": 7.5, "tertiary": 6.0, "residential": 3.5, "unclassified": 3.5,
                  "living_street": 3.0, "service": 2.5, "pedestrian": 2.5, "footway": 1.5, "cycleway": 1.5}
        n = 0
        for wid, w in self.osm.ways.items():
            t = w["tags"]
            hw = t.get("highway")
            if hw not in widths or not t.get("bridge") or t.get("tunnel"):
                continue
            xy = self.osm.way_xy(wid)
            if len(xy) < 2:
                continue
            line = LineString(xy)
            if not line.intersects(self.U):
                continue
            p = np.asarray(segmentize(line, 2.0).coords)
            g = self.dem(p[:, 0], p[:, 1])
            zend = np.interp(mk.cumlen(np.c_[p, g]), [0, mk.cumlen(np.c_[p, g])[-1]], [g[0], g[-1]])
            need = np.full(len(p), -1e9)
            for k, (x, y) in enumerate(p):
                for ti, d, zr in self.idx.near(x, y, widths[hw] + 2.5):
                    need[k] = max(need[k], zr + 7.6 + 1.0)
            z = np.maximum(zend, need)
            seg = np.r_[0, np.hypot(*np.diff(p, axis=0).T)]
            for _ in range(2):
                for k in range(1, len(z)):
                    z[k] = max(z[k], z[k - 1] - 0.06 * seg[k])
                for k in range(len(z) - 2, -1, -1):
                    z[k] = max(z[k], z[k + 1] - 0.06 * seg[k + 1])
            path = np.c_[p, z]
            hwid = widths[hw]
            mk.sweep(sc, "asphalt" if hw not in ("footway", "pedestrian", "cycleway") else "sidewalk", path, [(-hwid, 0), (hwid, 0)])
            mk.sweep(sc, "concrete", path, [(hwid, 0), (hwid, -1.0), (-hwid, -1.0), (-hwid, 0)])
            for sgn in (-1, 1):
                mk.sweep(sc, "parapet", path, [(sgn * hwid, 0), (sgn * hwid, 1.1), (sgn * (hwid - 0.2), 1.1), (sgn * (hwid - 0.2), 0)])
            for k in range(len(p) - 1):
                fp = LineString(p[k:k + 2]).buffer(hwid, cap_style="flat")
                sc.obstacle(fp, float(min(z[k], z[k + 1])) - 1.0, float(max(z[k], z[k + 1])) + 1.1, "overpass")
            fp_all = LineString(p).buffer(hwid)
            self.pieces.append(dict(ti=-1, fp=fp_all, sub=path, mode="roof", z=float(z.min()) - 1.0 + DECK + SLAB, low=[],
                                    mat="concrete", covered=False))
            n += 1
        self.piece_tree = STRtree([pc["fp"] for pc in self.pieces])
        self.log.append("overpasses: %d" % n)

    # ------------------------------------------------------------------ expressways
    def build_expressways(self):
        sc = self.sc
        self._U_out = self.U.buffer(1.0)
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
            for k in range(len(p) - 1):
                fp = LineString(p[k:k + 2]).buffer(width, cap_style="flat")
                sc.obstacle(fp, float(min(z[k], z[k + 1])) - 2.0, float(max(z[k], z[k + 1])) + 1.1, "expressway")
            s = mk.cumlen(path)
            _, left, _ = mk.frames(path)
            for s0 in np.arange(10, s[-1], 30):
                k = int(np.searchsorted(s, s0))
                k = min(k, len(p) - 1)
                x, y = p[k]
                if not self.clear_of_tracks(x, y, g[k], z[k] - 2.0, r=3.6) or contains_xy(self._U_out, x, y):
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
        inU = contains_xy(self.U.buffer(1.0), X.ravel(), Y.ravel())
        if inU.any():
            pts = np.c_[X.ravel()[inU], Y.ravel()[inU]]
            bed = self.idx.nearest_z(pts, r=40.0) - DECK - 0.4
            Zf = Z.ravel().copy()
            Zf[inU] = np.minimum(Zf[inU], bed)
            Z = Zf.reshape(Z.shape)
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
        B = [b for b in B if not Polygon(b["footprint"]).buffer(0).is_empty and Polygon(b["footprint"]).buffer(0).intersects(self.zone)]
        fps = [Polygon(b["footprint"]).buffer(0) for b in B]
        tree = STRtree(fps)
        parent = list(range(len(B)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        drop = set()
        for i, f in enumerate(fps):
            hi = B[i]["zmax"] - B[i]["zmin"]
            for j in tree.query(f.buffer(0.35)):
                if j <= i or j in drop:
                    continue
                hj = B[j]["zmax"] - B[j]["zmin"]
                ov = f.intersection(fps[j]).area
                if ov > 0.6 * min(f.area, fps[j].area) and abs(B[i]["zmax"] - B[j]["zmax"]) < 1.5:
                    drop.add(j if fps[j].area <= f.area else i)    # duplicate entry of the same building
                    continue
                if abs(hi - hj) < 2.5 and max(hi, hj) <= 25:       # one visible volume -> one colour and style
                    parent[find(j)] = find(i)
        style_of = {}
        for i, b in enumerate(B):
            if i in drop:
                nd += 1; continue
            root = find(i)
            if root not in style_of:
                style_of[root] = self.facade_for(B[root])
            b["_style"] = style_of[root]
        for i, b in enumerate(B):
            if i in drop:
                continue
            fp = fps[i]
            facade = b["_style"]
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
                self.block(q.simplify(0.3), b["zmin"], top, facade)
                sc.obstacle(q, b["zmin"], top, "building")
            nc += 1
        self.log.append("buildings kept %d, clipped %d, dropped %d" % (nk, nc, nd))

    def block(self, q, z0, top, style):
        """Flat-roofed extrusion with whole-wall quads (window layout needs per-wall width / height)."""
        sc = self.sc
        mat, wall_col, roof_col = style
        pts, tri = mk.triangulate(q)
        if len(tri):
            a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
            cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            tri = np.where(cr[:, None] < 0, tri[:, ::-1], tri)
            sc.b("roof").add(np.c_[pts, np.full(len(pts), top)], tri, pts, roof_col)
        for ring in [q.exterior] + list(q.interiors):
            r = np.asarray(ring.coords)
            if ring.is_ccw != (ring is q.exterior):
                r = r[::-1]
            for p0, p1 in zip(r[:-1], r[1:]):
                L = float(np.hypot(*(p1 - p0)))
                if L < 0.05:
                    continue
                V = np.array([[*p0, z0 - 0.5], [*p1, z0 - 0.5], [*p1, top], [*p0, top]])
                uv = np.array([[0, -0.5], [L, -0.5], [L, top - z0], [0, top - z0]])
                sc.b(mat).add(V, np.array([[0, 1, 2, 3]]), uv, wall_col, (L, top - z0, 1.0))

    def facade_for(self, b):
        """(wall material, wall colour, roof colour). Below 25 m: matte Japanese low-rise styles with per-building colour."""
        h = b["zmax"] - b["zmin"]
        name = b.get("name") or ""
        c = b["footprint"].mean(0)
        rng = np.random.default_rng(hsh(b["id"]))
        grey = float(rng.uniform(0.42, 0.75))
        roof = srgb_lin((grey, grey, grey * 0.98))
        if "東京駅" in name and c[0] < -40:
            return "facade_station_brick", (1.0, 1.0, 1.0), roof
        r = hsh(b["id"]) % 100
        if h > 60:
            return ["facade_glass", "facade_glass", "facade_office_tile"][r % 3], (1.0, 1.0, 1.0), roof
        if h > 25:
            return ["facade_office_tile", "facade_rect", "facade_glass", "facade_office_tile"][r % 4], (1.0, 1.0, 1.0), roof
        col = srgb_lin(pick_colour(rng))
        if h <= 12:
            kinds = ["res_siding", "res_siding", "res_mortar", "res_tile", "res_shop"]
        else:
            kinds = ["res_tile", "res_tile", "res_mortar", "res_shop"]
        return kinds[int(rng.integers(len(kinds)))], col, roof

    def emit_lod2(self, b, style):
        facade, wall_col, roof_col = style
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
                u = r @ hd
                u0, u1 = float(u.min()), float(u.max())
                # wall top = eave height at the wall's two ends (gable triangles get no windows)
                ends = (u - u0 < 0.3) | (u1 - u < 0.3)
                top = float(r[ends, 2].max()) - z0
                uv = np.c_[u - u0, r[:, 2] - z0]
                sc.b(facade).add(r, tri, uv, wall_col, (u1 - u0, top, 1.0))
            else:
                sc.b("roof").add(r, tri, r[:, :2], roof_col)
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
        self.build_arches()
        self.build_overpasses()
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
        """Write one .npy triple per material plus scene.json and station-sign PNGs (Blender-version neutral)."""
        os.makedirs(out, exist_ok=True)
        mats, tiles = [], []
        T = 1000.0
        for mat, bk in self.sc.buckets.items():
            a = bk.arrays()
            if a is None:
                continue
            v, f, uv, col, geo = a
            cen = v[f].mean(1)
            key = np.floor(cen[:, :2] / T).astype(int)
            for kx, ky in {tuple(k) for k in key}:
                sel = np.where((key[:, 0] == kx) & (key[:, 1] == ky))[0]
                ff = f[sel]
                used, inv = np.unique(ff.ravel(), return_inverse=True)
                name = "%s@%d_%d" % (mat, kx, ky)
                np.save(os.path.join(out, name + ".v.npy"), v[used].astype(np.float32))
                np.save(os.path.join(out, name + ".f.npy"), inv.reshape(-1, 3).astype(np.int32))
                np.save(os.path.join(out, name + ".uv.npy"), uv[used].astype(np.float32))
                np.save(os.path.join(out, name + ".c.npy"), col[used].astype(np.float32))
                np.save(os.path.join(out, name + ".g.npy"), geo[used].astype(np.float32))
                tiles.append([name, mat, (kx + 0.5) * T, (ky + 0.5) * T])
            mats.append(mat)
        cam, s = self.camera_path()
        stations = [(n, en, p.tolist()) for n, en, p in self.station_names()]
        for n, en, _ in stations:
            if "sign_" + n in mats:
                sign_png(os.path.join(out, "sign_%s.png" % n), n, en)
        json.dump(dict(materials=mats, tiles=tiles, camera=cam.tolist(), cam_s=s.tolist(), stations=stations, log=self.log),
                  open(os.path.join(out, "scene.json"), "w"), ensure_ascii=False)


def sign_png(path, name, en):
    from PIL import Image, ImageDraw, ImageFont
    font = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "RailSimulator", "Source", "UnityProject",
                        "Assets", "Resources", "UIFont.ttf")
    im = Image.new("RGB", (1024, 270), (245, 246, 244))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 200, 1024, 232], fill=(154, 205, 50))   # Yamanote line colour band
    try:
        f1 = ImageFont.truetype(font, 130); f2 = ImageFont.truetype(font, 40)
        f1.set_variation_by_axes([700]) if hasattr(f1, "set_variation_by_axes") else None
    except Exception:
        f1 = f2 = ImageFont.load_default()
    d.text((512, 95), name, fill=(20, 20, 20), font=f1, anchor="mm")
    d.text((512, 175), en, fill=(40, 40, 40), font=f2, anchor="mm")
    im.save(path)

if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    m = Model(data).build()
    import validate
    rep = validate.check(m)
    m.log += rep
    m.export(out)
    print("\n".join(m.log))
