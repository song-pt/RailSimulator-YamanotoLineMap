"""Download Poly Haven (CC0) PBR maps: diffuse, normal (GL), roughness at 1k/2k."""
import json, os, sys, urllib.request
TEX = {  # local name: (polyhaven id, resolution)
    "ballast": ("gravel_stones", "2k"), "brick": ("brick_wall_02", "2k"), "concrete": ("concrete_wall_004", "2k"),
    "concrete_smooth": ("painted_concrete", "1k"), "asphalt": ("asphalt_02", "2k"), "platform": ("concrete_tiles_02", "2k"),
    "sleeper": ("concrete_floor_02", "1k"), "rust": ("rusty_metal_02", "1k"), "metal": ("metal_plate", "1k"),
    "roof_sheet": ("box_profile_metal_sheet", "1k"), "facade_tile": ("concrete_tile_facade", "2k"),
    "facade_rect": ("rectangular_facade_tiles", "2k"), "plaster": ("grey_plaster_02", "1k"), "beige": ("beige_wall_001", "1k"),
    "roof": ("concrete_floor_worn_001", "1k"), "pavement": ("concrete_pavement", "2k"), "granite": ("granite_tile_02", "1k"),
}
UA = {"User-Agent": "RailSimulator-map-generator/1.0"}

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()

MAPS = {"Diffuse": "diff", "nor_gl": "nor", "Rough": "rough"}

def main(out):
    os.makedirs(out, exist_ok=True)
    for name, (pid, res) in TEX.items():
        files = json.loads(get("https://api.polyhaven.com/files/" + pid))
        for key, short in MAPS.items():
            dst = os.path.join(out, "%s_%s.jpg" % (name, short))
            if os.path.exists(dst):
                continue
            try:
                url = files[key][res]["jpg"]["url"]
            except KeyError:
                print("missing", pid, key); continue
            open(dst, "wb").write(get(url))
        print(name, pid)

if __name__ == "__main__":
    main(sys.argv[1])
