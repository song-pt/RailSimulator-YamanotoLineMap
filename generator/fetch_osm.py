"""Download OSM data for a bbox via the OSM API in small tiles and merge into one JSON."""
import json, os, sys, time, urllib.request, xml.etree.ElementTree as ET

def fetch_tile(b, cache):
    fn = os.path.join(cache, "osm_%.4f_%.4f_%.4f_%.4f.xml" % b)
    if not os.path.exists(fn):
        url = "https://api.openstreetmap.org/api/0.6/map?bbox=%.4f,%.4f,%.4f,%.4f" % b
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RailSimulator-map-generator"})
                data = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception as e:
                print("retry", url, e); time.sleep(2 ** attempt)
        else:
            raise RuntimeError(url)
        open(fn, "wb").write(data)
    return ET.parse(fn).getroot()

def main(w, s, e, n, cache, out, step=0.004):
    os.makedirs(cache, exist_ok=True)
    nodes, ways, rels = {}, {}, {}
    y = s
    while y < n:
        x = w
        while x < e:
            root = fetch_tile((x, y, min(x + step, e), min(y + step, n)), cache)
            for el in root:
                if el.tag not in ("node", "way", "relation"):
                    continue
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                i = int(el.get("id"))
                if el.tag == "node":
                    nodes[i] = (float(el.get("lon")), float(el.get("lat")), tags or None)
                elif el.tag == "way":
                    ways[i] = {"nodes": [int(nd.get("ref")) for nd in el.findall("nd")], "tags": tags}
                elif el.tag == "relation":
                    rels[i] = {"members": [(m.get("type"), int(m.get("ref")), m.get("role")) for m in el.findall("member")], "tags": tags}
            x += step
        y += step
    json.dump({"nodes": nodes, "ways": ways, "relations": rels}, open(out, "w"))
    print(len(nodes), len(ways), len(rels))

if __name__ == "__main__":
    a = sys.argv[1:]
    main(*map(float, a[:4]), a[4], a[5])
