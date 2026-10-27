"""Decode one split and score it. Output: experiments/<ID>/eval/<split>/{predictions.jsonl, metrics.json}

    python scripts/evaluate.py B0_phowhisper_small_zeroshot --split dev
    python scripts/evaluate.py V1_phowhisper_small_lora --split test --allow-test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.config import ROOT, load_experiment, log_run
from asr.data import SR, decode_audio, load_split
from asr.metrics import compute_metrics
from asr.model import load_model

LOCKED_SPLITS = {"test", "hard"}  # final evaluation only, never for model selection


def token_cap(ev: dict, duration: float) -> int:
    """Decode guard: max_new_tokens grows with clip length (eval.max_tokens_per_second, fit on TRAIN refs)."""
    if ev.get("max_tokens_per_second") is None:
        return ev["max_new_tokens"]
    return min(ev["max_new_tokens"], math.ceil(ev["max_tokens_per_second"] * duration) + ev["max_tokens_offset"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--checkpoint", help="default: experiments/<ID>/final if the experiment trains")
    ap.add_argument("--max-per-source", type=int, help="quick look on a subset")
    ap.add_argument("--data-dir")
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if args.split in LOCKED_SPLITS and not args.allow_test:
        sys.exit(f"'{args.split}' is locked. Freeze model/config first, then pass --allow-test.")

    cfg, exp_dir = load_experiment(args.exp_id, args.data_dir)
    ev = cfg["eval"]
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif ev.get("checkpoint"):  # evaluation-only experiment of an existing adapter (path relative to ASR/)
        ckpt = ROOT / ev["checkpoint"]
    else:
        ckpt = exp_dir / "final" if cfg["train"] else None
    if ckpt is not None and not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, processor = load_model(cfg, checkpoint=ckpt)
    model.to(device, dtype).eval()

    ds = load_split(cfg, args.split, args.max_per_source)
    silence_peak = ev.get("silence_peak")  # decode guard: max |wav| below this -> empty hyp, model not run
    eos = model.generation_config.eos_token_id
    rows, compute_s = [], 0.0
    for i, batch in enumerate(ds.iter(batch_size=ev["batch_size"])):
        waves = [decode_audio(a) for a in batch["audio"]]
        silent = [silence_peak is not None and (len(w) == 0 or float(np.abs(w).max()) < silence_peak) for w in waves]
        keep = [k for k, s in enumerate(silent) if not s]
        hyps, truncated = [""] * len(waves), [False] * len(waves)
        if keep:
            feats = processor.feature_extractor([waves[k] for k in keep], sampling_rate=SR,
                                                return_tensors="pt").input_features.to(device, dtype)
            cap = max(token_cap(ev, batch["duration"][k]) for k in keep)
            start = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(feats, language=cfg["model"]["language"], task=cfg["model"]["task"],
                                     num_beams=ev["num_beams"], max_new_tokens=cap)
            compute_s += time.perf_counter() - start
            for k, seq, hyp in zip(keep, ids, processor.batch_decode(ids, skip_special_tokens=True)):
                # Text token ids are < eos (specials start there); `cap` text tokens means EOS never came.
                hyps[k], truncated[k] = hyp, int((seq < eos).sum()) >= cap
        for k in range(len(waves)):
            rows.append({key: batch[key][k] for key in ("id", "source", "text", "cs_terms", "duration")}
                        | {"hyp": hyps[k].strip(), "silent": silent[k], "truncated": truncated[k]})
        print(f"\r{len(rows)}/{len(ds)}", end="", flush=True)
    print()

    def score(subset):
        m = compute_metrics([r["text"] for r in subset], [r["hyp"] for r in subset], [r["cs_terms"] for r in subset])
        m["audio_hours"] = sum(r["duration"] for r in subset) / 3600
        m["n_silent"] = sum(r["silent"] for r in subset)
        m["n_truncated"] = sum(r["truncated"] for r in subset)
        return m

    metrics = {"overall": score(rows),
               "by_source": {s: score([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})},
               "rtf_decode_only": compute_s / sum(r["duration"] for r in rows),
               "checkpoint": str(ckpt) if ckpt else cfg["model"]["name"],
               "decode": {"num_beams": ev["num_beams"], "batch_size": ev["batch_size"], "dtype": str(dtype), "device": device,
                          **{k: ev.get(k) for k in ("silence_peak", "max_tokens_per_second", "max_tokens_offset")}}}

    out = exp_dir / "eval" / (args.split if not args.max_per_source else f"{args.split}_sub{args.max_per_source}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log_run(cfg, exp_dir, f"eval:{out.name}", checkpoint=metrics["checkpoint"])
    print(json.dumps({"overall_wer": metrics["overall"]["wer"],
                      "by_source_wer": {s: m["wer"] for s, m in metrics["by_source"].items()}}, indent=2))


if __name__ == "__main__":
    main()
