import argparse, json, os, random
from collections import defaultdict
import sacrebleu

ap = argparse.ArgumentParser()
ap.add_argument("--set", dest="set_name", default="fleurs_dev")
ap.add_argument("--mt", default="vinai")
ap.add_argument("--a", default="M0")
ap.add_argument("--b", nargs="+", required=True)
ap.add_argument("--resamples", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

def load(tag, asr, var):
    p = f"runs/cascade/{args.set_name}_{tag}_{args.mt}_{asr}_{var}.jsonl"
    if not os.path.exists(p):
        return None
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}

def bleu(rows, ids):
    return sacrebleu.corpus_bleu([rows[i]["hyp_en"] for i in ids], [[rows[i]["en_ref"] for i in ids]]).score

rng = random.Random(args.seed)
cells = [(a, v) for a in ["gold", "B0", "B1"] for v in ["raw", "norm"]]
res = {}
print(f"{'input':<12}{'A':<12}{'B':<12}{'A BLEU':>8}{'B BLEU':>8}{'delta':>8}  95% CI          P(delta<=0)")
for asr, var in cells:
    A = load(args.a, asr, var)
    if A is None:
        continue
    groups = defaultdict(list)
    for i, r in A.items():
        groups[r["en_ref"]].append(i)
    keys = list(groups)
    samples = [[i for k in rng.choices(keys, k=len(keys)) for i in groups[k]] for _ in range(args.resamples)]
    a_full = bleu(A, list(A))
    a_boot = [bleu(A, s) for s in samples]
    for tag in args.b:
        B = load(tag, asr, var)
        if B is None:
            print(f"(thieu file: {tag} {asr} {var})")
            continue
        b_boot = [bleu(B, s) for s in samples]
        d = sorted(y - x for x, y in zip(a_boot, b_boot))
        lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
        p = sum(x <= 0 for x in d) / len(d)
        b_full = bleu(B, list(B))
        res[f"{asr}_{var}_{tag}"] = {"A": a_full, "B": b_full, "delta": b_full - a_full, "ci": [lo, hi], "p": p}
        print(f"{asr+' '+var:<12}{args.a:<12}{tag:<12}{a_full:8.2f}{b_full:8.2f}{b_full-a_full:+8.2f}  [{lo:+.2f},{hi:+.2f}]  {p:.3f}")
json.dump(res, open(f"reports/bootstrap_{args.set_name}_{args.a}_vs_{'_'.join(args.b)}.json", "w"), indent=1)
