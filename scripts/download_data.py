"""Download the Roboflow dataset (YOLOv8 format) into Sampled-YD-Object-Detection-2/.

The API key is read ONLY from the ROBOFLOW_API_KEY environment variable (a git-ignored .env is loaded as a
fallback via python-dotenv). The key is never printed; any exception text is scrubbed before it is shown,
because the Roboflow client can embed the key in request URLs.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Sampled-YD-Object-Detection-2"
WORKSPACE, PROJECT, VERSION, FORMAT = "malletbottle2", "sampled-yd-object-detection", 2, "yolov8"


def sanitize(msg: str, key: str) -> str:
    if key:
        msg = msg.replace(key, "<REDACTED>")
    # belt and braces: strip any api_key query parameter
    return re.sub(r"(api_key=)[^&\s'\"]+", r"\1<REDACTED>", msg)


def already_downloaded() -> bool:
    return all((DATASET_DIR / s / "images").is_dir() and any((DATASET_DIR / s / "images").iterdir())
               for s in ("train", "valid"))


def main() -> int:
    if already_downloaded():
        print(f"Dataset already present at {DATASET_DIR.name}/, skipping download.")
        return 0
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)  # real env var wins over .env
    except ImportError:
        pass
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print("ROBOFLOW_API_KEY is not set. Export it or put it in a git-ignored .env (see .env.example).",
              file=sys.stderr)
        return 1
    try:
        import truststore  # use OS cert store (fixes SSLCertVerificationError behind re-signing proxies)
        truststore.inject_into_ssl()
    except ImportError:
        pass

    os.chdir(ROOT)  # roboflow downloads into the cwd
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=key)
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(FORMAT, location=str(DATASET_DIR))
    except Exception as e:  # noqa: BLE001 - we only want to sanitize and report
        print(f"Download failed: {type(e).__name__}: {sanitize(str(e), key)}", file=sys.stderr)
        return 1
    if not already_downloaded():
        print("Download finished but train/valid images are missing.", file=sys.stderr)
        return 1
    print(f"Downloaded to {DATASET_DIR.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
