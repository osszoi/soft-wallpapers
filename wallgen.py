import sys, math
import numpy as np
from PIL import Image

# colour ramps: every stop is a neighbour of the next, so walking along one never jumps
RAMPS = {
    "twilight": [(0.05, 0.05, 0.14), (0.10, 0.11, 0.30), (0.20, 0.22, 0.52), (0.26, 0.40, 0.68), (0.36, 0.60, 0.76), (0.56, 0.78, 0.84)],
    "orchid":   [(0.06, 0.04, 0.14), (0.14, 0.09, 0.32), (0.30, 0.17, 0.56), (0.50, 0.24, 0.74), (0.70, 0.34, 0.80), (0.86, 0.56, 0.84)],
    "ember":    [(0.07, 0.04, 0.12), (0.20, 0.07, 0.22), (0.42, 0.11, 0.30), (0.64, 0.18, 0.34), (0.80, 0.32, 0.44), (0.90, 0.54, 0.58)],
    "lagoon":   [(0.04, 0.06, 0.14), (0.07, 0.14, 0.30), (0.10, 0.28, 0.46), (0.16, 0.46, 0.60), (0.30, 0.64, 0.70), (0.56, 0.82, 0.80)],
    "aurora":   [(0.05, 0.05, 0.15), (0.12, 0.12, 0.36), (0.28, 0.20, 0.60), (0.46, 0.28, 0.74), (0.62, 0.36, 0.72), (0.80, 0.46, 0.62)],
}

def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def ramp(stops, t):
    stops = np.array(stops, np.float32)
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    f = f * f * (3 - 2 * f)
    return stops[i] * (1 - f) + stops[i + 1] * f

def field(rng, x, y, scale):
    # a slow, smooth wobble built from a few long waves at odd angles
    total = np.zeros_like(x)
    for _ in range(4):
        angle = rng.uniform(0, math.pi)
        freq = rng.uniform(0.5, 1.2) / scale
        total += np.sin(2 * math.pi * freq * (x * math.cos(angle) + y * math.sin(angle)) + rng.uniform(0, 6.28))
    return total / 4

def render(spec, width, height, seed, dim=0.86):
    rng = np.random.default_rng(seed + 1000)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xs / height
    y = ys / height
    aspect = width / height
    stops, second = RAMPS[spec["ramp"]], RAMPS[spec.get("second", spec["ramp"])]
    # the second ramp takes over gradually across the picture, the way a mac wallpaper drifts
    # from one family of colours into its neighbour
    mixing = (0.5 + 0.5 * field(rng, x, y, 2.6))[..., None] * spec.get("blend", 0.0)
    wobble = field(rng, x, y, 2.2) * 0.12

    def colour_at(t):
        return ramp(stops, t) * (1 - mixing) + ramp(second, t) * mixing

    # one long gradient runs under everything, across the way the folds are stacked. A fold
    # mostly follows it and only leans a little towards its own tone, so the colour on either
    # side of an edge is nearly the same and the fold reads through light instead
    lead = math.radians(spec["layers"][0].get("angle", 0.0))
    across = -x * math.sin(lead) + y * math.cos(lead)
    across = (across - across.min()) / (across.max() - across.min())
    tones = [layer["tone"] for layer in spec["layers"]]
    if tones[0] > tones[-1]: across = 1 - across
    low, high = min(tones), max(tones)
    under = low + (high - low) * across
    own = spec.get("own", 0.3)

    img = colour_at(under - 0.04 + wobble).astype(np.float32)

    for layer in spec["layers"]:
        if spec["kind"] == "arcs":
            cx, cy = spec["centre"][0] * aspect, spec["centre"][1]
            u = np.arctan2(y - cy, x - cx) * spec.get("spin", 1.0)
            v = -np.hypot(x - cx, y - cy)
        else:
            angle = math.radians(layer["angle"])
            u = x * math.cos(angle) + y * math.sin(angle)
            v = -x * math.sin(angle) + y * math.cos(angle)
        crest = np.full_like(u, layer["at"])
        for wave in layer["waves"]:
            crest += wave[0] * np.sin(2 * math.pi * wave[1] * u + wave[2])
        d = v - crest

        depth = smooth(0.0, 0.7, d)
        along = np.sin(2 * math.pi * layer["drift"] * u + layer["phase"])
        colour = colour_at(under * (1 - own) + layer["tone"] * own + depth * 0.05 - along * 0.05 + wobble)
        # light catches the top of each fold and falls off behind it
        sheen = np.exp(-np.clip(d, 0, None) / layer["glow"])[..., None]
        colour = colour * (0.97 + 0.13 * sheen) + sheen * 0.012

        shade = np.exp(np.clip(d, None, 0) / 0.11) * (d < 0)
        img *= (1 - 0.11 * shade)[..., None]
        feather = layer["feather"]
        cover = smooth(-feather, feather, d)[..., None]
        img = img * (1 - cover) + colour * cover

    lx, ly = spec.get("light", (0.3, 0.2))
    light = np.exp(-(((x / aspect - lx) * aspect * 0.55) ** 2 + (y - ly) ** 2) / 0.9)
    img *= (0.82 + 0.3 * light)[..., None]
    vignette = 1 - 0.2 * (((x / aspect) * 2 - 1) ** 2 + (y * 2 - 1) ** 2) / 2
    img *= vignette[..., None] * dim
    img += rng.normal(0, 0.009, img.shape[:2])[..., None].astype(np.float32)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

def waves(rng, big, small):
    return [(big * rng.uniform(0.6, 1.0), rng.uniform(0.08, 0.18), rng.uniform(0, 6.28)),
            (small * rng.uniform(0.5, 1.0), rng.uniform(0.25, 0.45), rng.uniform(0, 6.28))]

def compose(kind, seed):
    rng = np.random.default_rng(seed)
    names = list(RAMPS)
    first = names[rng.integers(len(names))]
    second = names[rng.integers(len(names))]
    count = int(rng.integers(5, 8))
    spec = {"kind": "arcs" if kind == "arcs" else "lines", "ramp": first, "second": second, "blend": rng.uniform(0.35, 0.8),
            "light": (rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.5))}
    low, high = rng.uniform(0.12, 0.22), rng.uniform(0.74, 0.9)
    rising = rng.random() < 0.5
    layers = []
    for i in range(count):
        step = i / (count - 1)
        # tones walk steadily through the ramp, so two folds that touch are always close in colour
        tone = low + (high - low) * (step if rising else 1 - step)
        layer = {"tone": tone, "drift": rng.uniform(0.08, 0.2), "phase": rng.uniform(0, 6.28),
                 "glow": rng.uniform(0.07, 0.16), "feather": float(rng.choice([0.002, 0.006, 0.014, 0.03, 0.055]))}
        if kind == "dunes":
            base = rng.uniform(-5, 5) if i == 0 else layers[0]["angle"]
            layer.update(angle=base + rng.uniform(-4, 4), at=0.05 + step * 0.86 + rng.uniform(-0.03, 0.03), waves=waves(rng, 0.1, 0.035))
        elif kind == "curtains":
            base = 90 + rng.uniform(-12, 12) if i == 0 else layers[0]["angle"]
            layer.update(angle=base + rng.uniform(-6, 6), at=-3.3 + step * 3.0 + rng.uniform(-0.1, 0.1), waves=waves(rng, 0.22, 0.07))
        elif kind == "sweep":
            base = rng.uniform(16, 30) * rng.choice([-1, 1]) if i == 0 else layers[0]["angle"]
            start = -1.25 if base > 0 else -0.05
            layer.update(angle=base + rng.uniform(-5, 5), at=start + step * 1.75 + rng.uniform(-0.04, 0.04), waves=waves(rng, 0.14, 0.05))
        else:
            layer.update(at=-2.3 + step * 1.75 + rng.uniform(-0.03, 0.03), waves=waves(rng, 0.08, 0.03))
        layers.append(layer)
    if kind == "arcs":
        spec.update(centre=(rng.uniform(0.25, 0.75), rng.uniform(1.3, 1.8)), spin=1.3)
    spec["layers"] = layers
    return spec

if __name__ == "__main__":
    kind, seed, width, height, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    render(compose(kind, seed), width, height, seed).save(out, quality=95)
