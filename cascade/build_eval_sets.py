import json, re, unicodedata
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

def norm(s):
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

base = Path("cascade/data")
base.mkdir(parents=True, exist_ok=True)

flores = {}
for split in ["dev", "devtest"]:
    vi = load_dataset("openlanguagedata/flores_plus", "vie_Latn", split=split)
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    en_by_id = {r["id"]: r["text"] for r in en}
    for r in vi:
        flores.setdefault(norm(r["text"]), []).append((split, r["id"], r["text"], en_by_id[r["id"]]))

for fleurs_split, name in [("validation", "dev"), ("test", "test")]:
    ds = load_dataset("google/fleurs", "vi_vn", split=fleurs_split, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    audio_dir = base / "audio" / f"fleurs_{name}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, miss = [], 0
    for idx, r in enumerate(tqdm(ds, desc=name)):
        hit = flores.get(norm(r["raw_transcription"])) or flores.get(norm(r["transcription"]))
        if not hit:
            miss += 1
            continue
        fs, fid, _, en_text = hit[0]
        a = r["audio"]
        p = audio_dir / f"{idx:04d}.wav"
        p.write_bytes(a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes())
        rows.append({
            "id": f"fleurs:{name}:{idx:04d}", "source": "fleurs", "split": name,
            "fleurs_sentence_id": r["id"], "gender": str(r.get("gender")),
            "audio_path": str(p), "duration": r["num_samples"] / 16000,
            "text_raw": r["raw_transcription"], "text_norm": norm(r["raw_transcription"]),
            "en_ref": en_text, "flores_split": fs, "flores_id": fid,
        })
    assert len({r["id"] for r in rows}) == len(rows), "id khong duy nhat"
    assert len({r["audio_path"] for r in rows}) == len(rows), "audio bi trung"
    path = base / f"fleurs_{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = sf.info(rows[0]["audio_path"])
    hours = sum(r["duration"] for r in rows) / 3600
    uniq = len({r["fleurs_sentence_id"] for r in rows})
    print(f"[{name}] rows={len(rows)} files={len(rows)} unique_sentences={uniq} miss={miss} | {hours:.2f} h | sr={info.samplerate} ch={info.channels}")
