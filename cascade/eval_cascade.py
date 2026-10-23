import argparse, json, time
from sacrebleu.metrics import BLEU, CHRF
from MT.mt.model import Translator

ap = argparse.ArgumentParser()
ap.add_argument("--mt", choices=["vinai", "nllb"], default="vinai")
ap.add_argument("--set", dest="set_name", required=True)
ap.add_argument("--asr", nargs="+", default=["gold", "B0", "B1"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--tag", default="M0")
ap.add_argument("--unify-tone", action="store_true")
ap.add_argument("--adapter", default=None)
ap.add_argument("--verbalize", action="store_true")
args = ap.parse_args()
from common.textnorm import unify_tone
from MT.mt.noise import verbalize_numbers
prep = unify_tone if args.unify_tone else ((lambda s: verbalize_numbers(s)) if args.verbalize else (lambda s: s))

def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

inputs = {}
for a in args.asr:
    if a == "gold":
        rows, keys = read(f"cascade/data/{args.set_name}.jsonl"), {"raw": "text_raw", "norm": "text_norm"}
    else:
        rows, keys = read(f"runs/asr_public/{args.set_name}_{a}.jsonl"), {"raw": "hyp_raw", "norm": "hyp_norm"}
    if args.limit:
        rows = rows[: args.limit]
    for v, k in keys.items():
        inputs[(a, v)] = ([r["id"] for r in rows], [prep(r[k]) for r in rows], [r["en_ref"] for r in rows])

tr = Translator(args.mt, adapter=args.adapter)
flat = [s for ids, srcs, refs in inputs.values() for s in srcs]
t0 = time.time()
flat_out = tr.translate(flat, batch_size=args.batch, desc=f"{args.mt} {args.set_name}")
dt = time.time() - t0

bleu, chrf = BLEU(), CHRF(word_order=2)
scores, pos = {}, 0
print(f"\nMT={args.mt} tag={args.tag} set={args.set_name} | {len(set(flat))} cau duy nhat, dich {dt:.0f}s")
print(f"{'input':<12}{'dang':<7}{'BLEU':>7}{'chrF++':>9}")
for (a, v), (ids, srcs, refs) in inputs.items():
    hyps = flat_out[pos : pos + len(srcs)]
    pos += len(srcs)
    b, c = bleu.corpus_score(hyps, [refs]), chrf.corpus_score(hyps, [refs])
    scores[f"{a}/{v}"] = {"bleu": round(b.score, 2), "chrf++": round(c.score, 2)}
    print(f"{a:<12}{v:<7}{b.score:>7.2f}{c.score:>9.2f}")
    with open(f"runs/cascade/{args.set_name}_{args.tag}_{args.mt}_{a}_{v}.jsonl", "w", encoding="utf-8") as f:
        for i, s, h, r in zip(ids, srcs, hyps, refs):
            f.write(json.dumps({"id": i, "src": s, "hyp_en": h, "en_ref": r}, ensure_ascii=False) + "\n")
print("BLEU signature :", bleu.get_signature())
print("chrF signature :", chrf.get_signature())
with open(f"reports/cascade_{args.set_name}_{args.tag}_{args.mt}.json", "w", encoding="utf-8") as f:
    json.dump({"mt": args.mt, "tag": args.tag, "set": args.set_name, "seconds": round(dt, 1),
               "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature()),
               "scores": scores}, f, indent=2)
