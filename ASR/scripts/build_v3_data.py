"""V3 data: ViMedCSS re-split by source video + a TRAIN exclusion list. Existing data/ files are not modified.

    python scripts/build_v3_data.py [--data-dir data]

Official ViMedCSS splits share videos (81.7% of TRAIN clips come from videos also in DEV/TEST/HARD), so scores
on them do not measure generalization to new videos. This script writes
- data/ViMedCSS_vsplit/{train,dev,test,hard}.parquet: official train/dev/test/hard pooled and de-duplicated by id;
  each source video goes to one split via sha256(seed:video) -> 80/10/10. hard = official-hard clips that land in
  the new test split (a slice of test, not extra data). Silent clips (peak < SILENCE_PEAK) are dropped everywhere.
- data/v3_train_exclude.json: TRAIN ids of every source whose label cannot match the audio
  (silent, or more than MAX_TOKENS_PER_SECOND reference tokens per second of audio).
- data/ViMedCSS_vsplit/split_report.json, then data/registry.json is refreshed.
"""
import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asr.metrics import normalize_text
from prepare_data import SCHEMA, write_registry

SEED = 13
SILENCE_PEAK = 1e-3          # same threshold as the decode guard; derived from TRAIN peaks (see V1G config)
MAX_TOKENS_PER_SECOND = 16   # TRAIN p99.9 is 14.3 tokens/s; denser labels cover audio the clip does not contain
SPLITS = {"train": 0.8, "dev": 0.9, "test": 1.0}  # cumulative fractions of videos
OFFICIAL = ["train", "dev", "test", "hard"]


def peak(flac: bytes) -> float:
    wav, _ = sf.read(io.BytesIO(flac), dtype="float32")
    return float(np.abs(wav).max()) if len(wav) else 0.0


def video_split(video: str) -> str:
    u = int(hashlib.sha256(f"{SEED}:{video}".encode()).hexdigest(), 16) / 16 ** 64
    return next(s for s, edge in SPLITS.items() if u < edge)


def batches(path: Path, columns=None):
    for b in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
        yield from b.to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    data = Path(ap.parse_args().data_dir)
    src, out = data / "ViMedCSS", data / "ViMedCSS_vsplit"
    out.mkdir(exist_ok=True)

    # pass 1: assignment, de-duplication and silence, metadata only
    hard_ids = {r["id"] for r in batches(src / "hard.parquet", ["id"])}
    seen, assign, dropped_dup, dropped_silent = set(), {}, [], []
    for split in OFFICIAL:
        for r in batches(src / f"{split}.parquet", ["id", "group_id", "duration", "audio"]):
            if r["id"] in seen:
                dropped_dup.append(r["id"]); continue
            seen.add(r["id"])
            if not r["duration"] or r["duration"] <= 0 or peak(r["audio"]) < SILENCE_PEAK:  # empty or silent
                dropped_silent.append(r["id"]); continue
            assign[r["id"]] = video_split(r["group_id"])
        print(f"pass 1 {split}: kept {len(assign)} so far", flush=True)

    # pass 2: write each kept row once into its new split (+ hard slice of test); atomic like prepare_data.write
    tmp = {s: out / f"{s}.parquet.tmp" for s in [*SPLITS, "hard"]}
    writers = {s: pq.ParquetWriter(p, SCHEMA) for s, p in tmp.items()}
    buf = {s: [] for s in writers}

    def put(s, r):
        buf[s].append(r)
        if len(buf[s]) == 256:
            writers[s].write_table(pa.Table.from_pylist(buf[s], SCHEMA)); buf[s] = []

    written = set()
    for split in OFFICIAL:
        for r in batches(src / f"{split}.parquet"):
            new = assign.get(r["id"])
            if new is None or r["id"] in written:
                continue
            written.add(r["id"])
            r = r | {"source": "ViMedCSS_vsplit", "split": new}
            put(new, r)
            if new == "test" and r["id"] in hard_ids:
                put("hard", r | {"split": "hard"})
    for s, w in writers.items():
        if buf[s]:
            w.write_table(pa.Table.from_pylist(buf[s], SCHEMA))
        w.close()
        tmp[s].replace(out / f"{s}.parquet")

    report = {"seed": SEED, "fractions": SPLITS, "silence_peak": SILENCE_PEAK,
              "dropped_duplicate_ids": dropped_dup, "dropped_silent_ids": dropped_silent, "splits": {}}
    videos = {}
    for s in [*SPLITS, "hard"]:
        meta = pq.read_table(out / f"{s}.parquet", columns=["group_id", "duration"]).to_pylist()
        videos[s] = {m["group_id"] for m in meta}
        report["splits"][s] = {"clips": len(meta), "hours": round(sum(m["duration"] for m in meta) / 3600, 3),
                               "videos": len(videos[s])}
    overlap = {f"{a}&{b}": len(videos[a] & videos[b]) for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]}
    assert not any(overlap.values()), overlap
    report["video_overlap"] = overlap

    # TRAIN exclusion list over every source used by V3
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained("vinai/PhoWhisper-small")
    exclude, reasons = {}, Counter()
    for path in [data / s / "train.parquet" for s in ["VIVOS", "FLEURS", "VietMed"]] + [out / "train.parquet"]:
        for r in batches(path, ["id", "text", "duration", "audio"]):
            if not r["duration"] or r["duration"] <= 0:
                continue  # load_split already drops empty audio
            if peak(r["audio"]) < SILENCE_PEAK:
                exclude[r["id"]] = "silent"; reasons["silent"] += 1; continue
            n = max(len(tok(t, add_special_tokens=False).input_ids) for t in (r["text"], normalize_text(r["text"])))
            if n / r["duration"] > MAX_TOKENS_PER_SECOND:
                exclude[r["id"]] = f"{n / r['duration']:.1f} tokens/s"; reasons[path.parent.name] += 1
    (data / "v3_train_exclude.json").write_text(json.dumps(
        {"rule": f"TRAIN only: peak < {SILENCE_PEAK} or reference tokens/s > {MAX_TOKENS_PER_SECOND} (PhoWhisper tokenizer)",
         "counts": dict(reasons), "ids": exclude}, ensure_ascii=False, indent=1), encoding="utf-8")
    report["train_exclude_counts"] = dict(reasons)
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    write_registry(data)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("dropped")}, indent=1),
          "\ndropped duplicates:", len(dropped_dup), "| dropped silent:", len(dropped_silent))


if __name__ == "__main__":
    main()
