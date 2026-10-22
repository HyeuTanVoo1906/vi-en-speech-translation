import json, re, unicodedata
from pathlib import Path
from datasets import load_dataset, Audio

def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

out_dir = Path("cascade/data")
out_dir.mkdir(parents=True, exist_ok=True)

# FLoRes+: vi <-> en cung id
flores = {}
for split in ["dev", "devtest"]:
    vi = load_dataset("openlanguagedata/flores_plus", "vie_Latn", split=split)
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    en_by_id = {r["id"]: r["text"] for r in en}
    for r in vi:
        flores.setdefault(norm(r["text"]), []).append((split, r["id"], r["text"], en_by_id[r["id"]]))
print("FLoRes+ vi sentences:", sum(len(v) for v in flores.values()),
      "| keys trung nhau:", sum(1 for v in flores.values() if len(v) > 1))

for fleurs_split, name in [("validation", "dev"), ("test", "test")]:
    ds = load_dataset("google/fleurs", "vi_vn", split=fleurs_split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows, miss = [], []
    for r in ds:
        hit = flores.get(norm(r["raw_transcription"])) or flores.get(norm(r["transcription"]))
        if not hit:
            miss.append(r["raw_transcription"])
            continue
        fs, fid, vi_text, en_text = hit[0]
        rows.append({
            "id": f"fleurs:{r['id']}", "source": "fleurs", "split": name,
            "audio_path": r["audio"]["path"], "duration": r["num_samples"] / 16000,
            "text_raw": r["raw_transcription"], "text_norm": norm(r["raw_transcription"]),
            "en_ref": en_text, "flores_split": fs, "flores_id": fid,
        })
    path = out_dir / f"fleurs_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    uniq = len({(r["flores_split"], r["flores_id"]) for r in rows})
    print(f"\n[{name}] audio: {len(ds)} | ghep duoc: {len(rows)} ({100*len(rows)/len(ds):.1f}%) | cau duy nhat: {uniq} | -> {path}")
    for m in miss[:5]:
        print("  KHONG GHEP:", m)
    for row in rows[:2]:
        print("  VI:", row["text_raw"]); print("  EN:", row["en_ref"])
