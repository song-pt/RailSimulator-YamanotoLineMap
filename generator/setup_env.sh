#!/bin/sh
# One-shot environment + data setup for a fresh Linux (Ubuntu 24.04) cloud container.
# Usage: sh generator/setup_env.sh [WORKDIR]   (default WORKDIR=$HOME/railwork)
# Network needs: api.openstreetmap.org, cyberjapandata.gsi.go.jp, www.geospatial.jp,
# assets.cms.plateau.reearth.io, api.polyhaven.com, dl.polyhaven.org, download.blender.org
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
W="${1:-$HOME/railwork}"
D="$W/data"
mkdir -p "$D"

# system libs: Blender headless needs EGL/Mesa (CPU llvmpipe), ffmpeg for mp4
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1-mesa-dri libegl-mesa0 libxkbcommon0 \
  libsm6 libxi6 libxrender1 libxxf86vm1 libxfixes3 ffmpeg >/dev/null
pip install -q numpy shapely mapbox_earcut lxml scipy pillow

if [ ! -x "$W/blender/blender" ]; then
  curl -sSL -o "$W/blender.tar.xz" https://download.blender.org/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz
  tar xf "$W/blender.tar.xz" -C "$W" && mv "$W/blender-4.5.3-linux-x64" "$W/blender" && rm "$W/blender.tar.xz"
fi

cd "$HERE"
# OSM for Tokyo-Akihabara test bbox (~1 min)
[ -f "$D/osm.json" ] || python3 -I fetch_osm.py 139.758 35.674 139.782 35.704 "$D/osmcache" "$D/osm.json"
# GSI DEM tiles are fetched lazily by dem.py into $D/dem
python3 -I -c "import sys;sys.path.insert(0,'.');from dem import DEM;DEM(139.756,35.672,139.784,35.706,'$D/dem')"
# PLATEAU CityGML (only the needed 1 km mesh tiles via HTTP range requests, ~2 min)
P11="https://assets.cms.plateau.reearth.io/assets/d1/e11ec4-3ad6-4923-a441-32cbd7455dee/13101_chiyoda-ku_pref_2025_citygml_1_op.zip"
P12="https://assets.cms.plateau.reearth.io/assets/42/16c58f-1b9f-459d-976b-1f47fcdbbea7/13102_chuo-ku_pref_2025_citygml_1_op.zip"
P16="https://assets.cms.plateau.reearth.io/assets/80/45b2b1-5a88-40ab-9877-70b540b8707e/13106_taito-ku_city_2025_citygml_1_op.zip"
PAT=""; for m in 10 11 12 20 21 22 30 31 32 40 41 42; do PAT="$PAT udx/bldg/533946${m}_ udx/brid/533946${m}_"; done
[ -d "$D/plateau/chiyoda" ] || python3 -I fetch_plateau.py "$P11" "$D/plateau/chiyoda" $PAT
[ -d "$D/plateau/chuo" ] || python3 -I fetch_plateau.py "$P12" "$D/plateau/chuo" $PAT
[ -d "$D/plateau/taito" ] || python3 -I fetch_plateau.py "$P16" "$D/plateau/taito" $PAT
# corridor zone + building extraction (~30 s)
python3 -I - "$D" <<'EOF'
import sys, pickle; sys.path.insert(0, '.')
from geo import OSM
from shapely.geometry import LineString
from shapely.ops import substring
o = OSM(sys.argv[1] + '/osm.json')
main = LineString([o.xy(n) for n, _ in o.route_chain(1972960)])
pickle.dump(substring(main, 100, main.length).buffer(420), open(sys.argv[1] + '/zone.pkl', 'wb'))
EOF
[ -f "$D/buildings.pkl" ] || python3 -I plateau.py "$D/plateau" "$D/zone.pkl" "$D/buildings.pkl"
# Poly Haven CC0 textures
python3 -I fetch_textures.py "$D/tex"
echo "setup done: data in $D, blender in $W/blender"
