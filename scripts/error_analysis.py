"""Phân tích lỗi cascade trên FLEURS dev với ASR B0: BLEU theo nhóm WER, ma trận nhầm thanh điệu, ví dụ dịch sai.

Chạy từ gốc repo: python -m scripts.error_analysis [--tags M0 M4_s42] [--set fleurs_dev]
Kết quả: reports/errors/*. Không cần GPU (chỉ đọc kết quả có sẵn trong runs/cascade và runs/asr_public).
"""
import argparse
import json
import re
import unicodedata

import jiwer
import matplotlib
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--set", dest="set_name", default="fleurs_dev")
ap.add_argument("--tags", nargs=2, default=["M0", "M4_s42"], help="tag cơ sở và tag cải tiến")
ap.add_argument("--mt", default="vinai")
args = ap.parse_args()
OUT = "reports/errors"
import os

os.makedirs(OUT, exist_ok=True)
base_tag, new_tag = args.tags


def read(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])


asr = read(f"runs/asr_public/{args.set_name}_B0.jsonl")


def casc(tag, a, v="raw"):
    return read(f"runs/cascade/{args.set_name}_{tag}_{args.mt}_{a}_{v}.jsonl").set_index("id")


# 1. WER từng câu của B0 (cùng chuẩn hoá với báo cáo ASR: text_norm vs hyp_norm)
asr = asr[asr["text_norm"].str.strip() != ""].reset_index(drop=True)
wo = jiwer.process_words(list(asr["text_norm"]), list(asr["hyp_norm"]))
utt_wer, S, D, I = [], 0, 0, 0
for chunks, ref in zip(wo.alignments, wo.references):
    s = sum(c.ref_end_idx - c.ref_start_idx for c in chunks if c.type == "substitute")
    d = sum(c.ref_end_idx - c.ref_start_idx for c in chunks if c.type == "delete")
    i = sum(c.hyp_end_idx - c.hyp_start_idx for c in chunks if c.type == "insert")
    S, D, I = S + s, D + d, I + i
    utt_wer.append(100 * (s + d + i) / max(1, len(ref)))
asr["wer"] = utt_wer
nref = sum(len(r) for r in wo.references)
sid = {"wer_pct": round(100 * wo.wer, 2), "thay_the_pct": round(100 * S / nref, 2),
       "xoa_pct": round(100 * D / nref, 2), "chen_pct": round(100 * I / nref, 2)}

# 2. Ghép kết quả dịch
g_new, b_base, b_new = casc(new_tag, "gold"), casc(base_tag, "B0"), casc(new_tag, "B0")
df = asr.set_index("id").join(
    g_new[["hyp_en"]].rename(columns={"hyp_en": "en_gold_input"})).join(
    b_base[["hyp_en"]].rename(columns={"hyp_en": f"en_b0_{base_tag}"})).join(
    b_new[["hyp_en"]].rename(columns={"hyp_en": f"en_b0_{new_tag}"})).reset_index()

BINS = [-1, 0.001, 10, 30, 1e9]
LABELS = ["WER = 0%", "0-10%", "10-30%", "> 30%"]
df["nhom_wer"] = pd.cut(df["wer"], BINS, labels=LABELS)
bleu = BLEU()
rows = []
for lab in LABELS:
    g = df[df["nhom_wer"] == lab]
    if g.empty:
        continue
    refs = [list(g["en_ref"])]
    rows.append({
        "Nhóm WER của câu": lab, "Số câu": len(g), "Tỷ lệ (%)": round(100 * len(g) / len(df), 1),
        "BLEU gold": round(bleu.corpus_score(list(g["en_gold_input"]), refs).score, 2),
        f"BLEU B0 ({base_tag})": round(bleu.corpus_score(list(g[f"en_b0_{base_tag}"]), refs).score, 2),
        f"BLEU B0 ({new_tag})": round(bleu.corpus_score(list(g[f"en_b0_{new_tag}"]), refs).score, 2),
    })
tab = pd.DataFrame(rows)
tab.to_csv(f"{OUT}/bleu_by_wer_group.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 3.8))
w = 0.26
cols = ["BLEU gold", f"BLEU B0 ({base_tag})", f"BLEU B0 ({new_tag})"]
for j, (c, col) in enumerate(zip(cols, ["#3a9d6e", "#e0833a", "#2a6fdb"])):
    ax.bar(np.arange(len(tab)) + (j - 1) * w, tab[c], w, label=c, color=col)
ax.set_xticks(range(len(tab)))
ax.set_xticklabels([f"{r['Nhóm WER của câu']}\n(n={r['Số câu']})" for _, r in tab.iterrows()])
ax.set_ylabel("BLEU"); ax.set_title("BLEU theo mức lỗi ASR của câu (FLEURS dev)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/bleu_by_wer_group.png", dpi=160); plt.close(fig)

# 3. Ma trận nhầm thanh điệu (chỉ các từ thế nhau mà chỉ khác dấu thanh)
TONES = ["ngang", "huyền", "sắc", "hỏi", "ngã", "nặng"]
MARK = {"̀": "huyền", "́": "sắc", "̉": "hỏi", "̃": "ngã", "̣": "nặng"}


def split_tone(word):
    nfd = unicodedata.normalize("NFD", word)
    tone = "ngang"
    base = []
    for ch in nfd:
        if ch in MARK:
            tone = MARK[ch]
        else:
            base.append(ch)
    return unicodedata.normalize("NFC", "".join(base)), tone


mat = pd.DataFrame(0, index=TONES, columns=TONES)
sub_total = same_base = 0
for chunks, ref, hyp in zip(wo.alignments, wo.references, wo.hypotheses):
    for c in chunks:
        if c.type != "substitute" or (c.ref_end_idx - c.ref_start_idx) != (c.hyp_end_idx - c.hyp_start_idx):
            continue
        for rw, hw in zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]):
            sub_total += 1
            rb, rt = split_tone(rw)
            hb, ht = split_tone(hw)
            if rb == hb and rt != ht:
                same_base += 1
                mat.loc[rt, ht] += 1
mat.to_csv(f"{OUT}/tone_confusion.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(5, 4.4))
im = ax.imshow(mat.values, cmap="Blues")
ax.set_xticks(range(6)); ax.set_xticklabels(TONES, rotation=30); ax.set_yticks(range(6)); ax.set_yticklabels(TONES)
ax.set_xlabel("Thanh điệu ASR nhận ra"); ax.set_ylabel("Thanh điệu đúng (tham chiếu)")
for i in range(6):
    for j in range(6):
        if mat.values[i, j]:
            ax.text(j, i, int(mat.values[i, j]), ha="center", va="center",
                    color="white" if mat.values[i, j] > mat.values.max() / 2 else "black", fontsize=8)
ax.set_title(f"Nhầm thanh điệu của B0 (n={same_base} từ)")
fig.tight_layout(); fig.savefig(f"{OUT}/tone_confusion.png", dpi=160); plt.close(fig)

# 4. Mười ví dụ giảm BLEU nhiều nhất khi chuyển từ gold sang đầu ra B0 (bản cải tiến)
def sbleu(h, r):
    return BLEU(effective_order=True).sentence_score(h, [r]).score


df["bleu_gold"] = [sbleu(h, r) for h, r in zip(df["en_gold_input"], df["en_ref"])]
df["bleu_b0"] = [sbleu(h, r) for h, r in zip(df[f"en_b0_{new_tag}"], df["en_ref"])]
df["giam"] = df["bleu_gold"] - df["bleu_b0"]


def cause(r):
    if re.search(r"\d", r["text"]):
        return "số/ký hiệu"
    if r["wer"] > 30:
        return "ASR sai nhiều từ"
    names = re.findall(r"(?<!^)(?<![.?!] )\b[A-ZÀ-Ỹ][a-zà-ỹ]+", r["text"])
    if names:
        return "tên riêng"
    return "ASR nhầm một vài từ"


top = df.sort_values("giam", ascending=False).head(10).copy()
top["nguyen_nhan_goi_y"] = top.apply(cause, axis=1)
top[["id", "wer", "text", "hyp_raw", "en_gold_input", f"en_b0_{new_tag}", "en_ref", "bleu_gold", "bleu_b0", "nguyen_nhan_goi_y"]] \
    .round(1).to_csv(f"{OUT}/failure_examples.csv", index=False, encoding="utf-8-sig")

summary = {
    "so_cau": len(df), "asr_b0_dev": sid,
    "nhan_xet": {
        "ti_le_cau_wer_bang_0_pct": float(tab.loc[tab["Nhóm WER của câu"] == "WER = 0%", "Tỷ lệ (%)"].sum()),
        "tu_thay_the_chi_khac_thanh_pct_tren_tu_thay_the": round(100 * same_base / max(1, sub_total), 1),
    },
}
json.dump(summary, open(f"{OUT}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("== Lỗi ASR B0 (dev) ==", sid)
print("\n== BLEU theo nhóm WER ==")
print(tab.to_string(index=False))
print(f"\n== Nhầm thanh điệu: {same_base}/{sub_total} từ thay thế ({summary['nhan_xet']['tu_thay_the_chi_khac_thanh_pct_tren_tu_thay_the']}%) chỉ khác dấu thanh ==")
print(mat.to_string())
print("\n== 10 ví dụ giảm nhiều nhất ==")
for _, r in top.iterrows():
    print(f"[{r['id']}] WER {r['wer']:.0f}% | {r['nguyen_nhan_goi_y']}")
    print(f"   gold : {r['text']}\n   ASR  : {r['hyp_raw']}\n   EN(gold)  : {r['en_gold_input']}\n   EN(ASR)   : {r[f'en_b0_{new_tag}']}\n   EN(ref)   : {r['en_ref']}")
print(f"\nĐã ghi vào {OUT}/")
