"""Extract selected CityGML members from a remote PLATEAU zip using HTTP range requests."""
import io, os, sys, urllib.request, zipfile

class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (self.pos, self.pos + n - 1)})
        data = urllib.request.urlopen(req, timeout=300).read()
        b[:len(data)] = data; self.pos += len(data)
        return len(data)

def main(url, out, *patterns):
    os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(url), 1 << 20))
    for info in z.infolist():
        name = info.filename
        if any(p in name for p in patterns) and name.endswith(".gml"):
            dst = os.path.join(out, os.path.basename(name))
            if not os.path.exists(dst):
                with z.open(info) as src, open(dst, "wb") as f:
                    f.write(src.read())
            print(name, info.file_size)

if __name__ == "__main__":
    main(*sys.argv[1:])
