import json
import os
from pathlib import Path


def load_paths():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "paths.json"
    data = json.loads(config_path.read_text())

    resolved = {}
    for key, raw in data.items():
        expanded = os.path.expanduser(raw)
        p = Path(expanded)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        resolved[key] = p

    return resolved
