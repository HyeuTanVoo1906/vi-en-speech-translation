"""Record the frozen final system BEFORE any TEST decoding: config, adapter hashes, decode policy, DEV evidence.

    python scripts/freeze_final.py
Writes reports/final_freeze_<date>/freeze.json. Refuses to overwrite, and refuses if any candidate already has TEST output.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.config import load_experiment

FINAL = "V4B_phowhisper_small_lora_vsplit_specaug_beam5"
CANDIDATES = {  # name: (experiment, DEV run with full metrics, TEST run-id the protocol will use)
    "B0V": ("B0V_phowhisper_small_zeroshot_vsplit", "dev_guard", "test_guard"),
    "V3": ("V3_phowhisper_small_lora_vsplit", "dev_local_v3", "test_local_v3"),
    "V3B": ("V3B_phowhisper_small_lora_vsplit_beam5", "dev_beam5", "test_beam5"),
    "V4": ("V4_phowhisper_small_lora_vsplit_specaug", "dev_local_v4", "test_local_v4"),
    "V4B": (FINAL, "dev_beam5", "test_beam5"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    runs = ROOT / "local/runs"
    seen = [str(p) for exp, _, _ in CANDIDATES.values() for p in (runs / exp).glob("*/eval/test*/metrics.json")]
    if seen:
        sys.exit(f"TEST output already exists, a freeze now would not precede TEST: {seen}")
    out = ROOT / f"reports/final_freeze_{datetime.date.today():%Y%m%d}"
    if (out / "freeze.json").exists():
        sys.exit(f"{out / 'freeze.json'} exists; not overwriting")

    cfg, _ = load_experiment(FINAL)
    adapter = ROOT / cfg["eval"]["checkpoint"]
    dev = {}
    for name, (exp, dev_run, _) in CANDIDATES.items():
        m = json.loads((runs / exp / dev_run / "eval/dev/metrics.json").read_text(encoding="utf-8"))
        dev[name] = {"wer": m["overall"]["wer"], "by_source": {s: v["wer"] for s, v in m["by_source"].items()},
                     "cs_term_recall": m["overall"].get("cs_term_recall"), "rtf_decode_only": m["rtf_decode_only"],
                     "checkpoint": m["checkpoint"]}
    registry = json.loads((ROOT / "data/registry.json").read_text(encoding="utf-8"))
    record = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "final_system": FINAL,
        "selection": "lowest full-DEV WER among candidates on the V3 DEV (ViMedCSS re-split by video); TEST not decoded yet",
        "resolved_config": cfg,
        "adapter": {"path": str(adapter.relative_to(ROOT)),
                    "adapter_model.safetensors": sha256(adapter / "adapter_model.safetensors"),
                    "adapter_config.json": sha256(adapter / "adapter_config.json")},
        "decode": {k: cfg["eval"].get(k) for k in
                   ("num_beams", "max_new_tokens", "silence_peak", "max_tokens_per_second", "max_tokens_offset")},
        "dev_evidence": dev,
        "test_protocol": {"splits": ["test", "hard"], "run_ids": {n: c[2] for n, c in CANDIDATES.items()},
                          "rule": "decode each candidate once; no model or decode change after reading TEST"},
        "test_data": {k: v for k, v in registry.items() if k.endswith(("/test", "/hard"))
                      and k.split("/")[0] in ("VIVOS", "FLEURS", "VietMed", "ViMedCSS_vsplit")},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "freeze.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / "freeze.json")
    print(json.dumps({"final": FINAL, "adapter_sha256": record["adapter"]["adapter_model.safetensors"],
                      "decode": record["decode"], "dev_wer": {n: round(100 * d["wer"], 2) for n, d in dev.items()}}, indent=1))


if __name__ == "__main__":
    main()
