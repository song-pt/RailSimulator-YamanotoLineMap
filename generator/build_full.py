# Full-loop build: python3 -I generator/build_full.py  (data in ~/railwork/full/data, output ~/railwork/fullscene; resumes from ~/railwork/ckpt_structures.pkl)
import sys, os, time, json
sys.path.insert(0, '/home/user/RailSimulator-YamanotoLineMap/generator')
import model, validate
D = os.path.expanduser('~/railwork/full/data')
t0 = time.time()
import pickle
CK = os.path.expanduser('~/railwork/ckpt_structures.pkl')
if os.path.exists(CK):
    m = pickle.load(open(CK, 'rb')); stages = []
    print('resumed from checkpoint', flush=True)
else:
    m = model.Model(D)
    print('init', round(time.time() - t0), 'tracks', len(m.tracks), 'main', round(m.main.length), flush=True)
    stages = ['build_track', 'build_platforms', 'build_structures', 'platform_meshes', 'platform_furniture', 'build_parapets',
              'build_arches', 'build_overpasses']
for f in stages:
    t = time.time(); getattr(m, f)(); print(f, round(time.time() - t, 1), flush=True)
if stages:
    try:
        pickle.dump(m, open(CK, 'wb'), protocol=5); print('checkpoint saved', flush=True)
    except Exception as ex:
        print('checkpoint not saved:', ex, flush=True)
        if os.path.exists(CK): os.remove(CK)
for f in ['build_expressways', 'build_catenary', 'build_ground', 'build_buildings']:
    t = time.time(); getattr(m, f)(); print(f, round(time.time() - t, 1), flush=True)
t = time.time(); rep = validate.check(m); print('validate', round(time.time() - t, 1), flush=True); m.log += rep
m.export(os.path.expanduser('~/railwork/fullscene'))
print('\n'.join(m.log))
print('gap report', m.gap_report)
print('triangles', sum(len(b.arrays()[1]) for b in m.sc.buckets.values() if b.arrays() is not None))
# station positions along the main line for picking views
st = [(n, round(m.main.project(__import__('shapely').geometry.Point(p)))) for n, en, p in m.station_names() if m.main.distance(__import__('shapely').geometry.Point(p)) < 150]
print('stations', sorted(st, key=lambda a: a[1]))
