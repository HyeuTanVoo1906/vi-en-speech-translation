import random
import pandas as pd
from tqdm import tqdm
from common.textnorm import normalize_text
from MT.mt.noise import verbalize_numbers

for name, path, off in [("train", "MT/data/phomt_train_300k.parquet", 0), ("dev", "MT/data/phomt_dev_2k.parquet", 1)]:
    df = pd.read_parquet(path)
    rng = random.Random(42 + off)
    df["vi_num"] = [normalize_text(verbalize_numbers(v, rng)) for v in tqdm(df["vi"], desc=name)]
    df.to_parquet(path)
    print(name, "changed rows:", int((df["vi_num"] != df["vi_norm"]).sum()), "of", len(df))
