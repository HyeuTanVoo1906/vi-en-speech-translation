"""Experiment config loading and run logging."""
import datetime
import json
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def _merge(base: dict, override: dict) -> dict:
    """Deep merge; a mapping with `_replace: true` replaces the base mapping instead (e.g. a new source list)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace"):
            out[key] = {k: v for k, v in value.items() if k != "_replace"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment(exp_id: str, data_dir: str | None = None) -> tuple[dict, Path]:
    """experiments/_base.yaml + experiments/<exp_id>/config.yaml (deep merge)."""
    exp_dir = EXPERIMENTS / exp_id
    if not (exp_dir / "config.yaml").exists():
        raise FileNotFoundError(f"Missing {exp_dir / 'config.yaml'}")
    base = yaml.safe_load((EXPERIMENTS / "_base.yaml").read_text(encoding="utf-8"))
    override = yaml.safe_load((exp_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    cfg = _merge(base, override)
    cfg["exp_id"] = exp_id
    data_path = Path(data_dir or cfg["data"]["dir"])
    cfg["data"]["dir"] = str(data_path if data_path.is_absolute() else ROOT / data_path)
    return cfg, exp_dir


def log_run(cfg: dict, exp_dir: Path, kind: str, **extra) -> None:
    """Append one line per run/resume to experiments/<ID>/runs.jsonl for reproducibility."""
    import datasets, peft, torch, transformers

    registry = Path(cfg["data"]["dir"]) / "registry.json"
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "config": cfg,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "datasets": datasets.__version__,
        },
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "data_registry": json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else None,
        **extra,
    }
    with open(exp_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
