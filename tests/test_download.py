"""Checks for scripts/download_data.py that do not need network access or a real API key."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def test_sanitize_removes_key_everywhere():
    key = "abcSECRET123"
    msg = f"HTTPError for https://api.roboflow.com/x?api_key={key}&foo=1 (key={key})"
    out = dd.sanitize(msg, key)
    assert key not in out
    assert "api_key=<REDACTED>" in out


def test_sanitize_strips_api_key_param_even_without_known_key():
    assert "OTHERKEY" not in dd.sanitize("GET /v?api_key=OTHERKEY other", "")


def test_skip_when_dataset_present(tmp_path, monkeypatch, capsys):
    for split in ("train", "valid"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "images" / "a.jpg").touch()
    monkeypatch.setattr(dd, "DATASET_DIR", tmp_path)
    assert dd.main() == 0
    assert "skipping download" in capsys.readouterr().out


def test_missing_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "DATASET_DIR", tmp_path / "absent")
    monkeypatch.setattr(dd, "ROOT", tmp_path)  # no .env here, so dotenv cannot supply a key
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    assert dd.main() == 1
    assert "ROBOFLOW_API_KEY is not set" in capsys.readouterr().err
