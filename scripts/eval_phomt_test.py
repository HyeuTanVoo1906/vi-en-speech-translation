"""Dịch PhoMT test (vi -> en, bản detokenization) bằng VinAI Translate để so với số trong paper (dòng "nhóm tái lập").

Chạy từ gốc repo:
  python -m scripts.eval_phomt_test --n 500                 # chạy thử nhỏ
  python -m scripts.eval_phomt_test --n 0                   # toàn bộ test (khoảng 19k câu)
  python -m scripts.eval_phomt_test --n 0 --adapter runs/mt/M4_s42/adapter --tag M4_s42
Cần GPU. Kết quả: reports/phomt_test_<tag>_<mt>.json
"""
import argparse
import json
import random
import time

from sacrebleu.metrics import BLEU, CHRF

from MT.mt.model import Translator

ap = argparse.ArgumentParser()
ap.add_argument("--mt", choices=["vinai", "nllb"], default="vinai")
ap.add_argument("--adapter", default=None)
ap.add_argument("--tag", default="M0")
ap.add_argument("--n", type=int, default=500, help="số câu (0 = toàn bộ test)")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

D = "MT/data/phomt_raw/PhoMT/detokenization/test"
vi = [l.rstrip("\n") for l in open(f"{D}/test.vi", encoding="utf-8")]
en = [l.rstrip("\n") for l in open(f"{D}/test.en", encoding="utf-8")]
assert len(vi) == len(en), (len(vi), len(en))
idx = list(range(len(vi)))
if args.n and args.n < len(idx):
    random.Random(args.seed).shuffle(idx)
    idx = sorted(idx[: args.n])
src, ref = [vi[i] for i in idx], [en[i] for i in idx]

tr = Translator(args.mt, adapter=args.adapter)
t0 = time.time()
hyp = tr.translate(src, batch_size=args.batch, desc=f"{args.mt} phomt_test")
dt = time.time() - t0

bleu, chrf = BLEU(), CHRF(word_order=2)
b, c = bleu.corpus_score(hyp, [ref]), chrf.corpus_score(hyp, [ref])
out = {
    "mt": args.mt, "tag": args.tag, "adapter": args.adapter, "set": "PhoMT test (detokenization)",
    "n_sentences": len(src), "full_test": len(src) == len(vi), "seed": args.seed, "seconds": round(dt, 1),
    "bleu": round(b.score, 2), "chrf++": round(c.score, 2),
    "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open(f"reports/phomt_test_{args.tag}_{args.mt}_n{len(src)}.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
