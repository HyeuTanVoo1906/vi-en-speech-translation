import json, random, re
from collections import Counter
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from common.textnorm import normalize_text as norm

ROOT = Path("MT/data/phomt_raw/PhoMT/detokenization")
SEED, N_TRAIN, N_DEV = 42, 300_000, 2_000

# Cac cau KHONG duoc xuat hien trong du lieu huan luyen MT: FLEURS dev/test + toan bo FLoRes+
blocked_vi, blocked_en = set(), set()
for name in ["dev", "test"]:
    for line in open(f"cascade/data/fleurs_{name}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        blocked_vi.add(norm(r["text_raw"]))
        blocked_en.add(norm(r["en_ref"]))
for split in ["dev", "devtest"]:
    for lang, s in [("vie_Latn", blocked_vi), ("eng_Latn", blocked_en)]:
        for r in load_dataset("openlanguagedata/flores_plus", lang, split=split):
            s.add(norm(r["text"]))
print("blocked:", len(blocked_vi), "vi |", len(blocked_en), "en")

def basic_reason(vi, en):
    if not vi or not en:
        return "empty"
    wv, we = len(vi.split()), len(en.split())
    if not (3 <= wv <= 100 and 3 <= we <= 100):
        return "length"
    if not (0.6 <= len(vi) / len(en) <= 2.2):
        return "ratio"
    if vi == en:
        return "identical"
    return None

def scan(split):
    c, keep, seen = Counter(), [], set()
    with open(ROOT / split / f"{split}.vi", encoding="utf-8-sig") as fv, \
         open(ROOT / split / f"{split}.en", encoding="utf-8-sig") as fe:
        for i, (vi, en) in enumerate(tqdm(zip(fv, fe), desc=f"scan {split}")):
            vi, en = vi.strip(), en.strip()
            c["total"] += 1
            why = basic_reason(vi, en)
            if why is None:
                h = hash((vi, en))
                if h in seen:
                    why = "duplicate"
                else:
                    seen.add(h)
                    if norm(vi) in blocked_vi:
                        why = "blocked_vi"
                    elif norm(en) in blocked_en:
                        why = "blocked_en"
            if why:
                c[why] += 1
            else:
                keep.append(i)
    c["kept"] = len(keep)
    return keep, dict(c)

def fetch(split, idx):
    want, rows = set(idx), []
    with open(ROOT / split / f"{split}.vi", encoding="utf-8-sig") as fv, \
         open(ROOT / split / f"{split}.en", encoding="utf-8-sig") as fe:
        for i, (vi, en) in enumerate(zip(fv, fe)):
            if i in want:
                rows.append({"vi": vi.strip(), "en": en.strip()})
    return pd.DataFrame(rows)

audit = {"seed": SEED}
for split, n, out in [("train", N_TRAIN, "phomt_train_300k"), ("dev", N_DEV, "phomt_dev_2k")]:
    keep, counts = scan(split)
    chosen = sorted(random.Random(SEED).sample(keep, min(n, len(keep))))
    df = fetch(split, chosen)
    df["vi_norm"] = df["vi"].map(norm)
    df.to_parquet(f"MT/data/{out}.parquet", index=False)
    counts["sampled"] = len(df)
    audit[split] = counts
    print(split, counts)

# Kieu dat dau thanh: cu (hoa, huy) hay moi (hoa, huy)
OLD = re.compile(r"(oà|oá|oả|oã|oạ|oè|oé|oẻ|oẽ|oẹ)|(?<![qQ])(uỳ|uý|uỷ|uỹ|uỵ)")
NEW = re.compile(r"òa|óa|ỏa|õa|ọa|òe|óe|ỏe|õe|ọe|ùy|úy|ủy|ũy|ụy")
def tone(texts):
    s = " ".join(texts).lower()
    o, n = len(OLD.findall(s)), len(NEW.findall(s))
    return {"old_style": o, "new_style": n, "share_new": round(n / max(1, o + n), 3)}

def col(path, key):
    return [json.loads(l)[key] for l in open(path, encoding="utf-8")]

phomt = pd.read_parquet("MT/data/phomt_train_300k.parquet")
audit["tone_style"] = {
    "phomt_train_vi": tone(phomt["vi"].tolist()),
    "fleurs_dev_gold": tone(col("cascade/data/fleurs_dev.jsonl", "text_raw")),
    "fleurs_dev_B0_hyp": tone(col("runs/asr_public/fleurs_dev_B0.jsonl", "hyp_raw")),
    "fleurs_dev_B1_hyp": tone(col("runs/asr_public/fleurs_dev_B1.jsonl", "hyp_raw")),
}
print(json.dumps(audit["tone_style"], indent=2))
with open("reports/phomt_audit.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
