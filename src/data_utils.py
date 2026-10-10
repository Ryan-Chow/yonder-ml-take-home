"""Shared dataset helpers: label parsing, source-name grouping and perceptual-hash near-duplicate grouping.

Used by scripts/audit_data.py (to measure leakage) and scripts/make_split.py (to prevent it), so both agree on
what counts as "the same source photo".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Sampled-YD-Object-Detection-2"
CLASS_NAMES = ["bottle", "mallet"]  # ids 0, 1 as in data.yaml
SPLITS = ("train", "valid")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float


@dataclass
class Sample:
    split: str
    image: Path
    label: Path
    boxes: list[Box] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.image.name

    @property
    def classes(self) -> set[int]:
        return {b.cls for b in self.boxes}


def parse_label(path: Path) -> list[Box]:
    """Parse a YOLO label file. Raises ValueError on anything malformed instead of silently skipping it."""
    boxes = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{i}: expected 5 fields, got {len(parts)}: {line!r}")
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        if cls not in range(len(CLASS_NAMES)):
            raise ValueError(f"{path.name}:{i}: unknown class id {cls}")
        if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path.name}:{i}: coordinates out of range: {line!r}")
        boxes.append(Box(cls, xc, yc, w, h))
    return boxes


def load_split(root: Path, split: str) -> list[Sample]:
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    out = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"missing label for {img.name}")
        out.append(Sample(split, img, lbl, parse_label(lbl)))
    return out


def load_all(root: Path = RAW) -> list[Sample]:
    return [s for split in SPLITS for s in load_split(root, split)]


# --- grouping ------------------------------------------------------------------------------------------------

def source_stem(name: str) -> str:
    """'<orig>_jpg.rf.<hash>.jpg' -> '<orig>_jpg' (the Roboflow source photo id)."""
    return name.split(".rf.")[0]


_EXT_TOKENS = re.compile(r"(_(jpe?g|png|JPE?G|PNG))+$")
_COPY = re.compile(r"([-_ ]copy(-\d+)?)+", re.IGNORECASE)
_VIDEO = re.compile(r"^(?P<video>.+_mp4)-\d+$")


def family_key(name: str) -> str:
    """Coarser 'same origin' key from the filename.

    - frames of one video ('<video>_mp4-<frame>') -> the whole video is one family
      (neighbouring frames are near-identical, so splitting a video across train/val leaks);
    - '-copy' / '-copy-2' duplicates collapse onto the original name.
    Anything else stays its own family; visual near-duplicates are caught by the perceptual hash instead.
    """
    stem = _EXT_TOKENS.sub("", source_stem(name))
    m = _VIDEO.match(stem)
    if m:
        return "video:" + m.group("video")
    return _COPY.sub("", stem)


_SEQ = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")
_TIMESTAMP = re.compile(r"^(?P<date>\d{8})_(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})$")


def sequence_key(name: str) -> tuple[str, int] | None:
    """('<prefix>', n) for numbered captures like 'IMG_0619', 't_bot_mal262', 'video_mp4-77', or
    ('<date>', seconds) for phone timestamps '20231021_173204'. None for un-numbered names."""
    stem = _COPY.sub("", _EXT_TOKENS.sub("", source_stem(name)))
    m = _TIMESTAMP.match(stem)
    if m:
        return "ts:" + m["date"], int(m["h"]) * 3600 + int(m["m"]) * 60 + int(m["s"])
    m = _SEQ.match(stem)
    if m and m["prefix"]:
        return m["prefix"], int(m["num"])
    return None


# CENTER crop: Roboflow rotation leaves black corners that dominate a whole-image hash and make unrelated
# rotated photos look alike (seen on the pair contact sheet). The central 40% is mostly scene content.
PHASH_CROP = 0.3
MIN_CROP_STD = 8.0  # near-uniform crops (white synthetic collages) all hash identically -> don't match on them


def compute_phashes(samples: list[Sample], cache: Path | None = None) -> dict[str, "object"]:
    """64-bit perceptual hash of each image's centre crop; None for near-uniform crops. Cached as hex."""
    import imagehash
    import numpy as np
    from PIL import Image

    cached = json.loads(cache.read_text()) if cache and cache.exists() else {}
    out = {}
    for s in samples:
        key = f"{s.split}/{s.name}"
        if key not in cached:
            with Image.open(s.image) as im:
                w, h = im.size
                c = PHASH_CROP
                crop = im.convert("RGB").crop((int(w * c), int(h * c), int(w * (1 - c)), int(h * (1 - c))))
                uniform = float(np.asarray(crop.convert("L"), dtype=np.float32).std()) < MIN_CROP_STD
                cached[key] = None if uniform else str(imagehash.phash(crop))
        out[key] = None if cached[key] is None else imagehash.hex_to_hash(cached[key])
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached))
    return out


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


SEQ_GAP = 2          # consecutive capture numbers within this gap are treated as the same scene
TIMESTAMP_GAP_S = 30  # phone shots taken within 30 s of each other
PHASH_THRESHOLD = 10  # centre-crop Hamming distance; <=10 were all true near-duplicates on the contact sheet


def hamming_matrix(samples: list[Sample], hashes: dict):
    """Pairwise Hamming distances; 64 (max) wherever either hash is missing."""
    import numpy as np

    keys = [f"{s.split}/{s.name}" for s in samples]
    valid = np.array([hashes[k] is not None for k in keys])
    bits = np.array([hashes[k].hash.flatten() if hashes[k] is not None else np.zeros(64, bool) for k in keys],
                    dtype=np.uint8)
    d = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    d[~valid, :] = 64
    d[:, ~valid] = 64
    return d


def link_pairs(samples: list[Sample], hashes: dict | None, rules=("family", "sequence", "phash")):
    """Yield (i, j, rule) for every pair of samples considered the same origin by each rule."""
    import numpy as np

    if "family" in rules:
        first = {}
        for i, s in enumerate(samples):
            k = family_key(s.name)
            if k in first:
                yield first[k], i, "family"
            else:
                first[k] = i
    if "sequence" in rules:
        by_prefix = {}
        for i, s in enumerate(samples):
            sk = sequence_key(s.name)
            if sk:
                by_prefix.setdefault(sk[0], []).append((sk[1], i))
        for prefix, items in by_prefix.items():
            items.sort()
            gap = TIMESTAMP_GAP_S if prefix.startswith("ts:") else SEQ_GAP
            for (a, i), (b, j) in zip(items, items[1:]):
                if b - a <= gap:
                    yield i, j, "sequence"
    if "phash" in rules and hashes is not None:
        d = hamming_matrix(samples, hashes)
        for i, j in zip(*np.nonzero(np.triu(d <= PHASH_THRESHOLD, k=1))):
            yield int(i), int(j), "phash"


def build_groups(samples: list[Sample], hashes: dict | None, rules=("family", "sequence", "phash")) -> list[int]:
    """Connected components over all links: a group id per sample."""
    uf = UnionFind(len(samples))
    for i, j, _ in link_pairs(samples, hashes, rules):
        uf.union(i, j)
    return [uf.find(i) for i in range(len(samples))]
