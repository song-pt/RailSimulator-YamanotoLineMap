"""Structure-aware vertical profile for the main line around the whole loop.

Structure class per metre of main line comes from the OSM tags of the way it lies on:
  elevated  : bridge=viaduct / bridge=* / layer >= 1        -> street ground + RAIL_ABOVE_STREET
  covered   : tunnel=*                                        -> interpolated between portals
  earthwork : everything else (embankment, cutting, at grade) -> laser DEM along the centreline
GSI DEM5A is bare-earth laser data, so embankments and cuttings are already in it while
viaducts are not. Street ground is a low percentile of the DEM in a band beside the line.
The result is smoothed and grade-limited; stations are made level.
"""
import numpy as np
from shapely.geometry import Point

RAIL_ABOVE_STREET = 7.6
RAIL_ABOVE_FORMATION = 0.55   # formation (DEM) -> rail top on earthworks
MAX_GRADE = 0.025


def classify(tags):
    if tags.get("tunnel"):
        return "covered"
    layer = str(tags.get("layer", "0") or "0")
    if tags.get("bridge") or (layer.lstrip("-").isdigit() and int(layer) >= 1 and not tags.get("embankment") and not tags.get("cutting")):
        return "elevated"
    return "earthwork"


def street_ground(dem, xy, normals, offsets=(18, 25, 32, 40)):
    """Low percentile of DEM samples beside the line (both sides), i.e. the surrounding street level."""
    samples = []
    for o in offsets:
        for sgn in (-1, 1):
            p = xy + normals * (o * sgn)
            samples.append(dem(p[:, 0], p[:, 1]))
    return np.percentile(np.stack(samples), 25, axis=0)


def solve(main, way_tags_at, dem, ds=2.0, stations=()):
    """main: shapely LineString of the driven line; way_tags_at(s) -> tags of the way at s.
    stations: [(s0, s1)] platform ranges to keep level. Returns s, z, cls, ground arrays."""
    s = np.arange(0, main.length, ds)
    xy = np.array([main.interpolate(v).coords[0] for v in s])
    t = np.gradient(xy, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[-t[:, 1], t[:, 0]]
    cls = np.array([classify(way_tags_at(v)) for v in s])
    centre = np.median(np.stack([dem(*(xy + nrm * o).T) for o in (-1.5, 0, 1.5)]), axis=0)
    street = street_ground(dem, xy, nrm)
    # smooth street level heavily (it is a reference surface, not a track)
    k = np.exp(-0.5 * (np.arange(-60, 61) / 25.0) ** 2); k /= k.sum()
    street_s = np.convolve(np.pad(street, 60, mode="wrap"), k, mode="valid")
    target = np.where(cls == "elevated", street_s + RAIL_ABOVE_STREET, centre + RAIL_ABOVE_FORMATION)
    w = np.where(cls == "covered", 0.0, 1.0)
    # covered: interpolate between portals (weight 0 -> filled by smoothing below)
    z = target.copy()
    idx = np.arange(len(s))
    good = w > 0
    z[~good] = np.interp(idx[~good], idx[good], target[good], period=len(s))
    # robust smoothing: median then gaussian (loop is periodic)
    from scipy.ndimage import median_filter
    z = median_filter(z, size=15, mode="wrap")
    k = np.exp(-0.5 * (np.arange(-40, 41) / 12.0) ** 2); k /= k.sum()
    z = np.convolve(np.pad(z, 40, mode="wrap"), k, mode="valid")
    # level stations
    for a, b in stations:
        m = (s >= a - 10) & (s <= b + 10)
        if m.any():
            z[m] = z[m].mean()
    # grade limit (forward/backward passes, periodic)
    for _ in range(4):
        for i in range(1, len(z)):
            z[i] = np.clip(z[i], z[i - 1] - MAX_GRADE * ds, z[i - 1] + MAX_GRADE * ds)
        for i in range(len(z) - 2, -1, -1):
            z[i] = np.clip(z[i], z[i + 1] - MAX_GRADE * ds, z[i + 1] + MAX_GRADE * ds)
    return s, z, cls, street_s, centre
