"""Vectorised mesh building blocks. Units: metres, Z-up, x east, y north.

Every mesh is accumulated into a Bucket keyed by material name. UVs are in metres
(materials scale them), so textures keep a constant texel density everywhere.
"""
import numpy as np
import mapbox_earcut as earcut
from shapely.geometry import Polygon, MultiPolygon


class Bucket:
    def __init__(self):
        self.v, self.f, self.uv, self.c = [], [], [], []   # verts, faces, uv per vertex, linear rgb per vertex
        self.n = 0

    def add(self, verts, faces, uvs, col=None):
        verts = np.asarray(verts, np.float64).reshape(-1, 3)
        uvs = np.asarray(uvs, np.float64).reshape(-1, 2)
        faces = np.asarray(faces, np.int64)
        if len(verts) == 0 or len(faces) == 0:
            return
        self.v.append(verts); self.uv.append(uvs); self.f.append(faces + self.n)
        self.c.append(np.broadcast_to(np.asarray(col if col is not None else (1.0, 1.0, 1.0), np.float32), (len(verts), 3)))
        self.n += len(verts)

    def arrays(self):
        if not self.v:
            return None
        tris = []
        for f in self.f:
            if f.shape[1] == 3:
                tris.append(f)
            else:  # quads -> two triangles
                tris.append(f[:, [0, 1, 2]]); tris.append(f[:, [0, 2, 3]])
        return np.concatenate(self.v), np.concatenate(tris), np.concatenate(self.uv), np.concatenate(self.c)


class Scene:
    def __init__(self):
        self.buckets = {}
        self.obstacles = []   # (shapely footprint, zmin, zmax, label) for clearance validation

    def b(self, mat):
        if mat not in self.buckets:
            self.buckets[mat] = Bucket()
        return self.buckets[mat]

    def obstacle(self, poly, z0, z1, label):
        self.obstacles.append((poly, float(z0), float(z1), label))


def frames(path):
    """Tangent, horizontal left-normal and up for a polyline (N,3)."""
    p = np.asarray(path, float)
    t = np.gradient(p, axis=0)
    t[:, 2] = 0
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    left = np.stack([-t[:, 1], t[:, 0], np.zeros(len(t))], 1)
    up = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    return t, left, up


def cumlen(p):
    d = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(d)])


def sweep(sc, mat, path, profile, closed=False, cap=False, left=None, vscale=1.0):
    """Sweep a 2D profile [(lateral, up), ...] along path (N,3). Returns nothing; adds quads."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return
    _, lf, up = frames(path)
    if left is not None:
        lf = left
    prof = np.asarray(profile, float)
    if closed:
        prof = np.vstack([prof, prof[:1]])
    P = len(prof)
    N = len(path)
    verts = path[:, None, :] + lf[:, None, :] * prof[None, :, 0, None] + up[:, None, :] * prof[None, :, 1, None]
    s = cumlen(path)
    pv = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]) * vscale
    uv = np.stack(np.broadcast_arrays(s[:, None], pv[None, :]), -1)
    idx = np.arange(N * P).reshape(N, P)
    a = idx[:-1, :-1]; b = idx[1:, :-1]; c = idx[1:, 1:]; d = idx[:-1, 1:]
    faces = np.stack([a, b, c, d], -1).reshape(-1, 4)
    sc.b(mat).add(verts.reshape(-1, 3), faces, uv.reshape(-1, 2))
    if cap and P >= 3:
        ring = prof[:-1] if closed else prof
        tri = earcut.triangulate_float64(ring, np.array([len(ring)], np.uint32)).reshape(-1, 3)
        for end, flip in ((0, True), (N - 1, False)):
            v = path[end] + lf[end] * ring[:, 0, None] + up[end] * ring[:, 1, None]
            sc.b(mat).add(v, tri[:, ::-1] if flip else tri, ring)


def boxes(sc, mat, centers, ax, ay, size, uvscale=1.0):
    """Oriented boxes. centers (M,3); ax, ay horizontal unit axes (M,3); size (M,3) or (3,) = full extents x,y,z."""
    c = np.asarray(centers, float).reshape(-1, 3)
    M = len(c)
    if M == 0:
        return
    ax = np.broadcast_to(np.asarray(ax, float), (M, 3)); ay = np.broadcast_to(np.asarray(ay, float), (M, 3))
    az = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (M, 3))
    h = np.broadcast_to(np.asarray(size, float), (M, 3)) / 2
    sg = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float)  # 8 corners
    corners = (c[:, None] + ax[:, None] * (sg[None, :, 0, None] * h[:, None, 0, None])
               + ay[:, None] * (sg[None, :, 1, None] * h[:, None, 1, None])
               + az[:, None] * (sg[None, :, 2, None] * h[:, None, 2, None]))
    # faces with outward CCW winding (corner index = x + 2y + 4z) and the two local axes used for UV
    F = [([1, 3, 7, 5], 1, 2), ([0, 4, 6, 2], 1, 2), ([2, 6, 7, 3], 0, 2), ([0, 1, 5, 4], 0, 2),
         ([4, 5, 7, 6], 0, 1), ([0, 2, 3, 1], 0, 1)]
    V, UV = [], []
    for idx, ua, va in F:
        V.append(corners[:, idx])
        UV.append(np.stack([sg[idx, ua][None] * h[:, ua, None], sg[idx, va][None] * h[:, va, None]], -1) * uvscale)
    V = np.concatenate(V, 1); UV = np.concatenate(UV, 1)
    f = np.arange(24).reshape(6, 4)
    Fi = (f[None] + 24 * np.arange(M)[:, None, None]).reshape(-1, 4)
    sc.b(mat).add(V.reshape(-1, 3), Fi, UV.reshape(-1, 2))


def segment_boxes(sc, mat, a, b, w, h):
    """Box beams between point arrays a and b (M,3) with cross-section w (horizontal) x h."""
    a = np.asarray(a, float).reshape(-1, 3); b = np.asarray(b, float).reshape(-1, 3)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    ok = L > 1e-6
    a, b, d, L = a[ok], b[ok], d[ok], L[ok]
    x = d / L[:, None]
    ref = np.where(np.abs(x[:, 2:3]) > 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
    y = np.cross(ref, x); y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    M = len(a)
    if M == 0:
        return
    hw, hh = w / 2, h / 2
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    ring = y[:, None, :] * (corners[None, :, 0, None] * hw) + z[:, None, :] * (corners[None, :, 1, None] * hh)
    V = np.concatenate([a[:, None] + ring, b[:, None] + ring], 1)  # (M,8,3)
    F = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [3, 2, 1, 0], [4, 5, 6, 7]])
    F = (F[None] + 8 * np.arange(M)[:, None, None]).reshape(-1, 4)
    uv = np.zeros((M, 8, 2)); uv[:, 4:, 0] = L[:, None]; uv[:, [1, 2, 5, 6], 1] = max(w, h)
    sc.b(mat).add(V.reshape(-1, 3), F, uv.reshape(-1, 2))


def cylinders(sc, mat, base, height, radius, sides=10):
    base = np.asarray(base, float).reshape(-1, 3)
    M = len(base)
    if M == 0:
        return
    h = np.broadcast_to(np.asarray(height, float), (M,))
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    ring = np.stack([np.cos(ang) * radius, np.sin(ang) * radius, np.zeros_like(ang)], 1)
    bot = base[:, None] + ring[None]
    top = bot + np.stack([np.zeros(M), np.zeros(M), h], 1)[:, None]
    V = np.concatenate([bot, top], 1)
    i = np.arange(sides)
    F = np.stack([i, i + 1, i + 1 + sides + 1, i + sides + 1], 1)
    capt = np.stack([np.full(sides - 2, sides + 1), np.arange(sides + 2, 2 * sides), np.arange(sides + 3, 2 * sides + 1)], 1)
    F = np.concatenate([F, np.c_[capt, capt[:, 2]]])
    Ff = (F[None] + 2 * (sides + 1) * np.arange(M)[:, None, None]).reshape(-1, 4)
    uv = np.zeros((M, 2 * (sides + 1), 2))
    uv[:, :, 0] = np.tile(ang * radius, 2)
    uv[:, sides + 1:, 1] = h[:, None]
    sc.b(mat).add(V.reshape(-1, 3), Ff, uv.reshape(-1, 2))


def tubes(sc, mat, polylines, r):
    """Thin square tubes along each polyline (list of (N,3)) - wires."""
    for p in polylines:
        p = np.asarray(p, float)
        if len(p) < 2:
            continue
        d = np.gradient(p, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        ref = np.where(np.abs(d[:, 2:3]) > 0.9, [[1.0, 0, 0]], [[0, 0, 1.0]])
        y = np.cross(ref, d); y /= np.linalg.norm(y, axis=1, keepdims=True)
        z = np.cross(d, y)
        prof = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float) * r
        V = p[:, None] + y[:, None] * prof[None, :, 0, None] + z[:, None] * prof[None, :, 1, None]
        N = len(p)
        idx = np.arange(N * 5).reshape(N, 5)
        F = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        uv = np.zeros((N, 5, 2)); uv[:, :, 0] = cumlen(p)[:, None]; uv[:, :, 1] = np.arange(5) * r * 2
        sc.b(mat).add(V.reshape(-1, 3), F, uv.reshape(-1, 2))


def polys(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def triangulate(poly):
    """Earcut a shapely polygon (with holes). Returns (pts (n,2), tris (m,3))."""
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    pts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = earcut.triangulate_float64(pts, ends).reshape(-1, 3)
    return pts, tri


def densify(poly, step):
    from shapely import segmentize
    return segmentize(poly, step)


def prism(sc, top_mat, side_mat, poly, ztop, zbot, bottom=False, side_v0=None, top_only=False, col_top=None, col_side=None):
    """Extrude polygon. ztop/zbot: float or callable(xy (n,2))->z (n,)."""
    for p in polys(poly):
        if p.area < 0.01:
            continue
        p = p.buffer(0)
        for q in polys(p):
            pts, tri = triangulate(q)
            zt = ztop(pts) if callable(ztop) else np.full(len(pts), ztop)
            if len(tri):
                # earcut output winding: make faces point up
                a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
                cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                tri = np.where(cross[:, None] < 0, tri[:, ::-1], tri)
                sc.b(top_mat).add(np.c_[pts, zt], tri, pts, col_top)
                if bottom:
                    zb = zbot(pts) if callable(zbot) else np.full(len(pts), zbot)
                    sc.b(side_mat).add(np.c_[pts, zb], tri[:, ::-1], pts)
            if top_only:
                continue
            for ring in [q.exterior] + list(q.interiors):
                r = np.asarray(ring.coords)
                if ring.is_ccw == (ring is q.exterior):
                    pass
                else:
                    r = r[::-1]
                zt_r = ztop(r[:, :2]) if callable(ztop) else np.full(len(r), ztop)
                zb_r = zbot(r[:, :2]) if callable(zbot) else np.full(len(r), zbot)
                n = len(r)
                V = np.concatenate([np.c_[r[:, :2], zb_r], np.c_[r[:, :2], zt_r]])
                i = np.arange(n - 1)
                F = np.stack([i, i + 1, i + 1 + n, i + n], 1)
                s = cumlen(np.c_[r[:, :2], np.zeros(n)])
                v0 = zb_r if side_v0 is None else np.full(n, side_v0)
                uv = np.concatenate([np.c_[s, zb_r - v0], np.c_[s, zt_r - v0]])
                sc.b(side_mat).add(V, F, uv, col_side)
