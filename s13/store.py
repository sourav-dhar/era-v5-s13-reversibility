"""Results live as one JSON file per run, so a re-executed notebook never retrains a finished run."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def results_dir() -> Path:
    p = Path(os.environ.get('S13_RESULTS', ROOT / 'results'))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpt_dir() -> Path:
    p = Path(os.environ.get('S13_CKPT', ROOT / 'checkpoints')) / profile()
    p.mkdir(parents=True, exist_ok=True)
    return p


def profile() -> str:
    return os.environ.get('S13_PROFILE', 'full')


def path(key: str) -> Path:
    return results_dir() / profile() / f'{key}.json'


def load(key: str):
    p = path(key)
    return json.loads(p.read_text()) if p.exists() else None


def save(key: str, obj):
    p = path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float))
    return obj


def cached(key: str, fn):
    r = load(key)
    if r is None:
        r = save(key, fn())
    else:
        print(f'[cache] {key}: loaded from {path(key).relative_to(results_dir())}')
    return r
