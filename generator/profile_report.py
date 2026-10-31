# Full-loop profile report: python3 -I generator/profile_report.py (run from generator/), data in ~/railwork/full/data
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from geo import OSM
from dem import DEM
import profile as pf
from shapely.geometry import LineString, Point
D = os.path.expanduser('~/railwork/full/data')
o = OSM(D + '/osm.json')
ext = json.load(open(D + '/extent.json'))['bbox']
dem = DEM(*ext, D + '/dem')
ch = o.route_chain(1972960)
xy = [o.xy(n) for n, _ in ch]
main = LineString(xy)
print('main length', round(main.length), 'closed gap', round(Point(xy[0]).distance(Point(xy[-1])), 1))
# way tags along s
wids = [w for _, w in ch]
cum = np.r_[0, np.cumsum(np.hypot(*np.diff(np.array(xy), axis=0).T))]
def tags_at(s):
    i = min(len(wids) - 1, int(np.searchsorted(cum, s)))
    return o.ways[wids[i]]['tags']
s, z, cls, street, centre = pf.solve(main, tags_at, dem)
st = [(t.get('name'), np.array(o.xy(k))) for k, (lon, lat, t) in o.nodes.items() if t and t.get('railway') == 'station' and 'JR' in (t.get('operator', '') + t.get('network', '') + 'JR') and t.get('name')]
# segments
seg = []
start = 0
for i in range(1, len(s) + 1):
    if i == len(s) or cls[i] != cls[start]:
        a, b = start, i
        if (b - a) * 2 >= 60:
            rs = z[a:b] - street[a:b]; rc = z[a:b] - centre[a:b]
            p = main.interpolate(s[(a + b) // 2])
            near = min(st, key=lambda q: np.hypot(*(q[1] - np.array(p.coords[0]))))[0] if st else ''
            seg.append((round(s[a]), round(s[b - 1]), cls[a], round(float(rs.mean()), 1), round(float(rs.min()), 1), round(float(rc.mean()), 1), near))
        start = i
for x in seg: print(x)
g = np.abs(np.diff(z)) / 2
print('max grade permil', round(float(g.max()) * 1000, 1))
print('rail below street (non-covered) >1.5m: %.0f m' % (2 * np.sum((z - street < -1.5) & (cls != 'covered'))))
print('rail below local DEM centre on earthwork >1m (would be buried): %.0f m' % (2 * np.sum((z - centre < -1.0) & (cls == 'earthwork'))))
np.save(os.path.expanduser('~/railwork/full/profile.npy'), np.c_[s, z, street, centre])
