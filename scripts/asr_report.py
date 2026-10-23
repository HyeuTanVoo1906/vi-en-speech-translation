import json, jiwer
from common.textnorm import normalize_text
from MT.mt.noise import verbalize_numbers

out = {}
print(f"{'set':<12}{'asr':<5}{'ref':<11}{'WER%':>7}{'CER%':>7}{'sub':>6}{'del':>6}{'ins':>6}{'ref words':>10}")
for st in ["fleurs_dev", "fleurs_test"]:
    raw = {r["id"]: r["text_raw"] for r in map(json.loads, open(f"cascade/data/{st}.jsonl", encoding="utf-8"))}
    for m in ["B0", "B1"]:
        R = [json.loads(l) for l in open(f"runs/asr_public/{st}_{m}.jsonl", encoding="utf-8")]
        for kind in ["digits", "verbalized"]:
            refs, hyps = [], []
            for r in R:
                ref = r["text_norm"] if kind == "digits" else normalize_text(verbalize_numbers(raw[r["id"]]))
                if ref.strip():
                    refs.append(ref); hyps.append(r["hyp_norm"])
            w = jiwer.process_words(refs, hyps)
            c = jiwer.cer(refs, hyps)
            n = w.hits + w.substitutions + w.deletions
            out[f"{st}_{m}_{kind}"] = {"wer": w.wer, "cer": c, "sub": w.substitutions, "del": w.deletions, "ins": w.insertions, "ref_words": n, "utts": len(refs)}
            print(f"{st:<12}{m:<5}{kind:<11}{w.wer*100:7.2f}{c*100:7.2f}{w.substitutions:6d}{w.deletions:6d}{w.insertions:6d}{n:10d}")
json.dump(out, open("reports/asr_public_wer.json", "w"), indent=1)
