import json, os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from common.textnorm import normalize_text

ds = load_dataset("google/fleurs", "vi_vn", split="train", trust_remote_code=True)
adir = "cascade/data/audio/fleurs_train"
os.makedirs(adir, exist_ok=True)
rows = []
for i, ex in enumerate(tqdm(ds, desc="fleurs_train")):
    a = ex["audio"]
    p = f"{adir}/{i:04d}.wav"
    sf.write(p, a["array"], a["sampling_rate"])
    raw = ex.get("raw_transcription") or ex["transcription"]
    rows.append({"id": f"fleurs:train:{i:04d}", "source": "fleurs", "split": "train",
                 "fleurs_sentence_id": ex["id"], "gender": ex.get("gender"),
                 "audio_path": p, "duration": len(a["array"]) / a["sampling_rate"],
                 "text_raw": raw, "text_norm": normalize_text(raw), "en_ref": ""})
assert len({r["id"] for r in rows}) == len(rows)
with open("cascade/data/fleurs_train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rows", len(rows), "hours", sum(r["duration"] for r in rows) / 3600)
