import random
import pandas as pd
from collections import Counter
from tqdm import tqdm
from jiwer import wer
from common.textnorm import normalize_text
from MT.mt.noise import verbalize_numbers, swap_tone

RATE, P_NOISY, SEED = 0.085, 0.8, 42
PATHS = [("train", "MT/data/phomt_train_300k.parquet", 0), ("dev", "MT/data/phomt_dev_2k.parquet", 1)]
tr = pd.read_parquet(PATHS[0][1])
vocab = [w for w, _ in Counter(w for s in tr["vi_norm"][:20000] for w in s.split()).most_common(5000)]

def rnd(words, rng):
    out = []
    for w in words:
        if rng.random() >= RATE:
            out.append(w)
            continue
        x = rng.random()
        if x < 0.25:
            continue
        elif x < 0.6:
            out.append(swap_tone(w, rng))
        elif x < 0.8:
            out.append(rng.choice(vocab))
        elif len(w) > 2:
            k = rng.randrange(len(w))
            out.append(w[:k] + w[k + 1:])
        else:
            out.append(w)
    return out

def make(raw, rng):
    clean = normalize_text(verbalize_numbers(raw, rng))
    if rng.random() < P_NOISY:
        noisy = " ".join(rnd(clean.split(), rng))
        if noisy.strip():
            return noisy
    return clean

for name, path, off in PATHS:
    df = pd.read_parquet(path)
    rng = random.Random(SEED + off)
    df["vi_rand"] = [make(v, rng) for v in tqdm(df["vi"], desc=name)]
    df.to_parquet(path)
clean = [normalize_text(verbalize_numbers(v)) for v in df["vi"][:1000]]
ok = [(c, s) for c, s in zip(clean, df["vi_rand"][:1000]) if c.strip() and s.strip()]
print(f"measured random-noise level (dev): WER {wer([c for c,_ in ok], [s for _,s in ok])*100:.2f}%  (M4 asr-style was 6.76%)")
for v, a in list(zip(df["vi"], df["vi_rand"]))[:3]:
    print("RAW :", v); print("RND~:", a); print()
