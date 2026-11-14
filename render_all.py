import sys, time
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import wallgen
picks = [("sweep", s) for s in (1, 2, 4, 5, 9, 10, 13, 22, 24)] + [("curtains", s) for s in (2, 5, 6, 10, 14, 22, 24)] + [("dunes", s) for s in (4, 9, 12, 22)]
start, count = int(sys.argv[1]), int(sys.argv[2])
for number, (kind, seed) in list(enumerate(picks, 1))[start:start + count]:
    began = time.time()
    wallgen.render(wallgen.compose(kind, seed), 5120, 1440, seed).save(f"soft-{number:02d}-{kind}.jpg", quality=95, subsampling=0)
    print(f"soft-{number:02d}-{kind}.jpg {time.time() - began:.1f}s", flush=True)
