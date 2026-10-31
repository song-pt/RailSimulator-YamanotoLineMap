"""Rail network in the corridor with a solved vertical profile.

Heights are absolute (Tokyo Peil, same datum as GSI DEM and PLATEAU). The reference
track (Yamanote) sits RAIL_ABOVE_GROUND above a heavily smoothed ground line; every
other track gets base + offset(layer). OSM layer tags are only trusted on long ways,
short layer flips (bridges over streets) are ignored, and the result is smoothed on
the track graph so level changes become ramps instead of steps.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

RAIL_ABOVE_GROUND = 7.6
LAYER_OFFSET = {1: 0.0, 2: 8.4, 3: 16.8, 4: 11.0}
GOOD_LAYER_LEN = 80.0
SMOOTH = 6.0
# lines that run on the same structure as the Yamanote line around the loop (shared viaducts / island platforms)
PAIRED_LINES = ("山手線", "京浜東北線")


def parse_layer(tags):
    v = str(tags.get("layer", "0") or "0").split(";")[0].strip()
    try:
        return int(float(v))
    except ValueError:
        return 0


def resample(line, step):
    n = max(2, int(np.ceil(line.length / step)) + 1)
    return np.array([line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)])


class Track:
    def __init__(self, wid, tags, xy):
        self.wid, self.tags = wid, tags
        self.line = LineString(xy)
        self.name = tags.get("name", "")
        self.bridge = tags.get("bridge") == "yes"
        self.shinkansen = "新幹線" in self.name
        self.pts = None  # (N,3) resampled with heights


class Network:
    def __init__(self, osm, dem, main_rel, s_range, corridor=160.0):
        self.osm, self.dem = osm, dem
        chain = osm.route_chain(main_rel)
        self.main = LineString([osm.xy(n) for n, _ in chain])
        self.main_ways = {w for _, w in chain}
        s0, s1 = s_range
        from shapely.ops import substring
        self.zone = substring(self.main, max(0, s0), min(self.main.length, s1)).buffer(corridor)
        # structure-aware reference profile along the main line (see profile.py)
        import profile as pf
        cxy = np.array([osm.xy(n) for n, _ in chain])
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(cxy, axis=0).T))]
        wids = [w for _, w in chain]
        self._chain_cum, self._chain_wids = cum, wids
        self.ms, self.base_h, self.main_cls, self.street, _ = pf.solve(self.main, self.main_tags_at, dem)
        self.main_layer = np.array([parse_layer(self.main_tags_at(v)) for v in self.ms])
        near_main = self.main.buffer(30.0)
        self.tracks = []
        for wid, w in osm.ways.items():
            t = w["tags"]
            if t.get("railway") != "rail":
                continue
            layer = parse_layer(t)
            below = t.get("tunnel") or layer < 0
            xy = osm.way_xy(wid)
            if len(xy) < 2:
                continue
            if below and wid not in self.main_ways:
                # cuttings / covered sections: only shallow tracks running with the main line
                if layer < -1 or not LineString(xy).within(near_main.buffer(5.0)):
                    continue
            if len(xy) < 2:
                continue
            ln = LineString(xy)
            if not ln.intersects(self.zone):
                continue
            tr = Track(wid, t, xy)
            tr.main = wid in self.main_ways
            tr.nodes = [n for n in w["nodes"] if n in osm.nodes]
            self.tracks.append(tr)
        self.solve_heights()

    def main_tags_at(self, v):
        return self.osm.ways[self._chain_wids[min(len(self._chain_wids) - 1, int(np.searchsorted(self._chain_cum, v)))]]["tags"]

    def main_layer_at(self, x, y):
        s = np.array([self.main.project(Point(a, b)) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))])
        return self.main_layer[np.clip(np.searchsorted(self.ms, s), 0, len(self.ms) - 1)]

    def base(self, x, y):
        s = np.array([self.main.project(Point(a, b)) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))])
        return np.interp(s, self.ms, self.base_h)

    def solve_heights(self, level=None):
        idx = {}
        for tr in self.tracks:
            for n in tr.nodes:
                idx.setdefault(n, len(idx))
        N = len(idx)
        xy = np.array([self.osm.xy(n) for n in idx])
        base = self.base(xy[:, 0], xy[:, 1])
        mlay = self.main_layer_at(xy[:, 0], xy[:, 1])
        target = np.zeros(N); wsum = np.zeros(N)
        rows, cols, vals = [], [], []
        for tr in self.tracks:
            L = tr.line.length
            layer = parse_layer(tr.tags)
            w = 1.0 if L >= GOOD_LAYER_LEN else 0.0
            if tr.main:
                off, w = 0.0, 1e4
            for n in tr.nodes:
                i = idx[n]
                wn = w
                # offset relative to the main line's own layer here (main is layer 1 on viaducts, -1 in cuttings ...)
                off = 0.0 if tr.main else LAYER_OFFSET.get(int(np.clip(layer - mlay[i] + 1, 1, 4)), 0.0)
                paired = any(k in tr.name for k in PAIRED_LINES) and self.main.distance(Point(xy[i])) < 20.0
                if not tr.main and layer <= mlay[i] and (paired or self.main.distance(Point(xy[i])) < 6.5):
                    wn = 1e3   # tracks running alongside the main line share its structure level
                target[i] += wn * (base[i] + off); wsum[i] += wn
            for a, b in zip(tr.nodes[:-1], tr.nodes[1:]):
                i, j = idx[a], idx[b]
                d = max(1.0, float(np.hypot(*(xy[i] - xy[j]))))
                k = 900.0 / d  # stiffness: smoothing length scale ~ sqrt(900) = 30 m per unit anchor
                rows += [i, j, i, j]; cols += [i, j, j, i]; vals += [k, k, -k, -k]
        Lmat = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
        # nodes without any trusted anchor follow the base profile weakly
        weak = wsum == 0
        target[weak] = base[weak]; wsum[weak] = 1e-3
        for n, z in (level or {}).items():  # station levelling: tracks along one platform share a level
            if n in idx:
                i = idx[n]
                target[i] = z * 5e4; wsum[i] = 5e4
        A = Lmat * SMOOTH + sp.diags(wsum)
        h = spla.spsolve(A.tocsc(), target)
        self.node_h = {n: h[i] for n, i in idx.items()}
        for tr in self.tracks:
            hn = np.array([self.node_h[n] for n in tr.nodes])
            dn = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(np.array(tr.line.coords), axis=0).T))])
            p = resample(tr.line, 2.0)
            dp = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
            tr.pts = np.c_[p, np.interp(dp, dn, hn)]

    def report(self):
        out = []
        for tr in self.tracks:
            p = tr.pts
            d = np.hypot(*np.diff(p[:, :2], axis=0).T)
            g = np.abs(np.diff(p[:, 2])) / np.maximum(d, 1e-3)
            gnd = self.dem(p[:, 0], p[:, 1])
            out.append((tr.wid, tr.name, tr.tags.get("layer"), round(tr.line.length), round(float(g.max()) * 1000, 1),
                        round(float((p[:, 2] - gnd).min()), 1)))
        return out
