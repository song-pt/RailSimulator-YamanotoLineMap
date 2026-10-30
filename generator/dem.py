"""GSI DEM5A (fallback DEM10B) tiles -> bilinear ground-height sampler in the local frame."""
import math, os, urllib.request
import numpy as np
from geo import proj, unproj

Z = 15

def tile_xy(lon, lat):
    n = 2 ** Z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y

def fetch(tx, ty, cache):
    os.makedirs(cache, exist_ok=True)
    out = np.full((256, 256), np.nan)
    for kind in ("dem5a", "dem5b", "dem"):
        fn = os.path.join(cache, "%s_%d_%d.txt" % (kind, tx, ty))
        if not os.path.exists(fn):
            url = "https://cyberjapandata.gsi.go.jp/xyz/%s/%d/%d/%d.txt" % (kind, Z, tx, ty)
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
            except Exception:
                data = b""
            open(fn, "wb").write(data)
        txt = open(fn).read().strip()
        if not txt:
            continue
        a = np.array([[float(v) if v != "e" else np.nan for v in row.split(",")] for row in txt.split("\n")])
        out = np.where(np.isnan(out), a, out)
        if not np.isnan(out).any():
            break
    return out

class DEM:
    def __init__(self, w, s, e, n, cache):
        x0, y1 = tile_xy(w, n); x1, y0 = tile_xy(e, s)
        self.tx0, self.ty0 = int(x0), int(y1)
        nx, ny = int(x1) - self.tx0 + 1, int(y0) - self.ty0 + 1
        g = np.zeros((ny * 256, nx * 256))
        for j in range(ny):
            for i in range(nx):
                g[j*256:(j+1)*256, i*256:(i+1)*256] = fetch(self.tx0 + i, self.ty0 + j, cache)
        # fill holes (water, buildings voids) by iterative neighbour averaging
        for _ in range(200):
            m = np.isnan(g)
            if not m.any(): break
            p = np.pad(g, 1, constant_values=np.nan)
            nb = np.stack([p[:-2,1:-1], p[2:,1:-1], p[1:-1,:-2], p[1:-1,2:]])
            avg = np.nanmean(nb, axis=0)
            g[m] = avg[m]
        self.g = np.nan_to_num(g, nan=float(np.nanmean(g)))
    def __call__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        lon, lat = unproj(x, y)
        n = 2 ** Z
        px = ((lon + 180) / 360 * n - self.tx0) * 256 - 0.5
        lr = np.radians(lat)
        py = ((1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * n - self.ty0) * 256 - 0.5
        i0 = np.clip(np.floor(px).astype(int), 0, self.g.shape[1] - 2); j0 = np.clip(np.floor(py).astype(int), 0, self.g.shape[0] - 2)
        fx = np.clip(px - i0, 0, 1); fy = np.clip(py - j0, 0, 1)
        g = self.g
        return (g[j0, i0] * (1-fx) * (1-fy) + g[j0, i0+1] * fx * (1-fy) + g[j0+1, i0] * (1-fx) * fy + g[j0+1, i0+1] * fx * fy)
