import argparse, json, random
import pandas as pd
from jiwer import wer
from tqdm import tqdm
from common.textnorm import normalize_text
from MT.mt.noise import NoiseModel, verbalize_numbers, make_noisy

ap = argparse.ArgumentParser()
ap.add_argument("--asr", default="runs/asr_public/fleurs_train_B0.jsonl")
ap.add_argument("--meta", default="cascade/data/fleurs_train.jsonl")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--p-noisy", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

raw = {r["id"]: r["text_raw"] for r in map(json.loads, open(args.meta, encoding="utf-8"))}
pairs = []
for r in map(json.loads, open(args.asr, encoding="utf-8")):
    ref = normalize_text(verbalize_numbers(raw[r["id"]])).split()
    if ref:
        pairs.append((ref, r["hyp_norm"].split()))
nm = NoiseModel.build(pairs)
nm.save("MT/data/noise_model.json")
print(f"pairs {len(pairs)} | WER after verbalizing numbers {nm.wer*100:.2f}% | span rules {len(nm.lex)} | insertion rate {nm.p_ins*100:.2f}%")

for name, path, off in [("train", "MT/data/phomt_train_300k.parquet", 0), ("dev", "MT/data/phomt_dev_2k.parquet", 1)]:
    df = pd.read_parquet(path)
    rng = random.Random(args.seed + off)
    df["vi_asr"] = [make_noisy(v, nm, rng, args.p_noisy, args.scale) for v in tqdm(df["vi"], desc=name)]
    df.to_parquet(path)
    print(name, len(df), "rows saved with vi_asr")

clean = [normalize_text(verbalize_numbers(v)) for v in df["vi"][:1000]]
sub = df["vi_asr"][:1000].tolist()
ok = [(c, s) for c, s in zip(clean, sub) if c.strip() and s.strip()]
print(f"measured noise level vs clean (dev, 30% sentences untouched): WER {wer([c for c,_ in ok], [s for _,s in ok])*100:.2f}%")
for v, a in list(zip(df["vi"], df["vi_asr"]))[:6]:
    print("RAW :", v); print("ASR~:", a); print()
