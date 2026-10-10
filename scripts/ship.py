"""Pick the arm that ships and copy its weights to weights/best.pt.

Rule (written down before the clahe_only / aug_only runs finished, see AI_LOG): the rover first has to FIND the
object, so mAP50 is primary and mAP50-95 (box tightness) breaks ties.
  1. best = arm with the highest mAP50 on grouped val
  2. candidates = arms whose mAP50 is not significantly below best (paired cluster-bootstrap 95% CI upper >= 0)
  3. ship the candidate with the highest mAP50-95

Reads results/ablation.json (scripts/bootstrap_ablation.py); writes weights/best.pt and results/shipped.json.

    python scripts/ship.py [--arm NAME]      # --arm overrides the rule (recorded as an override)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choose(ab: dict) -> tuple[str, dict]:
    pt = ab["point"]
    best = max(pt, key=lambda k: pt[k]["mAP50"])
    cands = [k for k in pt if k == best or ab["pairwise"][f"{k}-{best}:mAP50"]["ci"][1] >= 0]
    ship = max(cands, key=lambda k: pt[k]["mAP50-95"])
    return ship, {"best_mAP50_arm": best, "candidates": cands,
                  "rule": "max mAP50-95 among arms whose mAP50 is not significantly below the best mAP50 arm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    a = ap.parse_args()
    ab = json.loads((ROOT / "results" / "ablation.json").read_text())
    arm, why = choose(ab)
    if a.arm:
        why["override"] = f"rule chose {arm}, overridden to {a.arm}"
        arm = a.arm
    src = ROOT / "runs" / arm / "weights" / "best.pt"
    if not src.exists():
        sys.exit(f"{src} missing")
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy2(src, ROOT / "weights" / "best.pt")
    out = dict(why, arm=arm, metrics=ab["point"][arm], sha256=sha256(ROOT / "weights" / "best.pt"))
    (ROOT / "results" / "shipped.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
