"""EDA cho Task 2: bảng nguồn dữ liệu, chia tập, thống kê định lượng, biểu đồ trước/sau xử lý.

Chạy từ gốc repo: python -m scripts.eda
Kết quả: reports/eda/*.csv, *.json, *.png. Không cần GPU.
"""
import json
import os
import re

import jiwer
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.textnorm import unify_tone

OUT = "reports/eda"
SEED = 42
B0_DEV_WER = 8.92  # reports/asr_public_wer.json, B0 dev, ref số đọc thành chữ (cùng chuẩn vi_num)
os.makedirs(OUT, exist_ok=True)

DIGIT = re.compile(r"\d")
PUNCT = re.compile(r"[^\w\s]")
GENDER = {0: "nam", 1: "nữ", 2: "khác"}


def pct(mask):
    return round(100 * float(mask.mean()), 2)


def words(s):
    return s.str.split().str.len()


def mb(path):
    return round(os.path.getsize(path) / 1e6, 2)


def audio_mb(df):
    total = 0
    for p in df["audio_path"]:
        p = p.replace("\\", os.sep)
        if os.path.exists(p):
            total += os.path.getsize(p)
    return round(total / 1e6, 1)


fl = {s: pd.read_json(f"cascade/data/fleurs_{s}.jsonl", lines=True) for s in ["train", "dev", "test"]}
phomt = pd.read_parquet("MT/data/phomt_train_300k.parquet")
phomt_dev = pd.read_parquet("MT/data/phomt_dev_2k.parquet")
audit = json.load(open("reports/phomt_audit.json", encoding="utf-8"))

stats = {"phomt_audit": audit}

# 1.1 Bảng nguồn dữ liệu
rows = []
for s, d in fl.items():
    rows.append({
        "Nguồn": f"FLEURS vi_vn ({s})", "Loại": "Online (google/fleurs)",
        "Số mẫu": len(d), "Số câu duy nhất": d["fleurs_sentence_id"].nunique(),
        "Số giờ audio": round(d["duration"].sum() / 3600, 2),
        "Định dạng": "wav 16 kHz mono + jsonl", "Dung lượng (MB)": audio_mb(d),
        "Trường dữ liệu": ", ".join(d.columns),
    })
rows.append({
    "Nguồn": "FLoRes+ eng (devtest)", "Loại": "Online (openlanguagedata/flores_plus)",
    "Số mẫu": int(sum(d["flores_id"].nunique() for d in [fl["dev"], fl["test"]])),
    "Số câu duy nhất": "", "Số giờ audio": "", "Định dạng": "jsonl (cột en_ref)", "Dung lượng (MB)": "",
    "Trường dữ liệu": "en_ref, flores_id, flores_split",
})
rows.append({
    "Nguồn": "PhoMT train (mẫu 300k)", "Loại": "Online (vinai/PhoMT)", "Số mẫu": len(phomt), "Số câu duy nhất": "",
    "Số giờ audio": "", "Định dạng": "parquet", "Dung lượng (MB)": mb("MT/data/phomt_train_300k.parquet"),
    "Trường dữ liệu": ", ".join(phomt.columns),
})
rows.append({
    "Nguồn": "PhoMT dev (mẫu 2k)", "Loại": "Online (vinai/PhoMT)", "Số mẫu": len(phomt_dev), "Số câu duy nhất": "",
    "Số giờ audio": "", "Định dạng": "parquet", "Dung lượng (MB)": mb("MT/data/phomt_dev_2k.parquet"),
    "Trường dữ liệu": ", ".join(phomt_dev.columns),
})
rows.append({
    "Nguồn": "Tự thu âm (Offline)", "Loại": "Offline", "Số mẫu": 0, "Số câu duy nhất": 0, "Số giờ audio": 0,
    "Định dạng": "", "Dung lượng (MB)": 0, "Trường dữ liệu": "chưa có",
})
src = pd.DataFrame(rows)
src.to_csv(f"{OUT}/source_table.csv", index=False, encoding="utf-8-sig")

# 1.3 Chia tập
tot = sum(len(d) for d in fl.values())
split_rows = []
for s, d in fl.items():
    split_rows.append({
        "Tập": f"FLEURS {s}", "Số mẫu": len(d), "Tỷ lệ (%)": round(100 * len(d) / tot, 1),
        "Số câu duy nhất": d["fleurs_sentence_id"].nunique(), "Số giờ": round(d["duration"].sum() / 3600, 2),
    })
pd.DataFrame(split_rows).to_csv(f"{OUT}/split_table.csv", index=False, encoding="utf-8-sig")
stats["fleurs_split_pct"] = {r["Tập"]: r["Tỷ lệ (%)"] for r in split_rows}

# Phân bố "lớp": giới tính người nói và nhóm độ dài câu
gender = {}
for s, d in fl.items():
    vc = d["gender"].map(GENDER).value_counts(normalize=True).mul(100).round(1)
    gender[s] = vc.to_dict()
stats["fleurs_gender_pct"] = gender
BINS = [0, 10, 20, 30, 1000]
LABELS = ["1-10 từ", "11-20 từ", "21-30 từ", ">30 từ"]
fl_len = pd.cut(words(fl["train"]["text_norm"]), BINS, labels=LABELS).value_counts(normalize=True).reindex(LABELS) * 100
pm_len = pd.cut(words(phomt["vi_norm"]), BINS, labels=LABELS).value_counts(normalize=True).reindex(LABELS) * 100
stats["length_bucket_pct"] = {"FLEURS train": fl_len.round(1).to_dict(), "PhoMT 300k": pm_len.round(1).to_dict()}

# Thống kê miền: PhoMT (train) so với FLEURS (nơi dùng)
tone_sample = phomt["vi"].sample(20000, random_state=SEED)
fl_all = pd.concat(fl.values())
dom = {
    "PhoMT": {
        "% câu có chữ số": pct(phomt["vi"].str.contains(DIGIT)),
        "% câu có chữ hoa": pct(phomt["vi"].map(lambda s: any(c.isupper() for c in s))),
        "% câu có dấu câu": pct(phomt["vi"].str.contains(PUNCT)),
        "% câu có dấu thanh kiểu mới": pct(tone_sample.map(lambda s: unify_tone(s) != s)),
        "số từ trung vị": float(words(phomt["vi_norm"]).median()),
        "số từ p95": float(words(phomt["vi_norm"]).quantile(0.95)),
    },
    "FLEURS (text gốc)": {
        "% câu có chữ số": pct(fl_all["text_raw"].str.contains(DIGIT)),
        "% câu có chữ hoa": pct(fl_all["text_raw"].map(lambda s: any(c.isupper() for c in s))),
        "% câu có dấu câu": pct(fl_all["text_raw"].str.contains(PUNCT)),
        "% câu có dấu thanh kiểu mới": pct(fl_all["text_raw"].map(lambda s: unify_tone(s) != s)),
        "số từ trung vị": float(words(fl_all["text_norm"]).median()),
        "số từ p95": float(words(fl_all["text_norm"]).quantile(0.95)),
    },
}
stats["domain_gap"] = dom
stats["duration_s"] = {s: d["duration"].describe().round(2).to_dict() for s, d in fl.items()}

# 1.4 Trước/sau xử lý trên PhoMT train
after = {
    "vi (gốc)": phomt["vi"], "vi_num (đọc số + chuẩn hoá)": phomt["vi_num"],
    "vi_asr (+ nhiễu kiểu ASR)": phomt["vi_asr"], "vi_rand (+ nhiễu ngẫu nhiên)": phomt["vi_rand"],
}
ba = {}
for name, col in after.items():
    ba[name] = {
        "% câu có chữ số": pct(col.str.contains(DIGIT)),
        "% câu có chữ hoa": pct(col.map(lambda s: any(c.isupper() for c in s))),
        "% câu có dấu câu": pct(col.str.contains(PUNCT)),
    }
stats["before_after_pct"] = ba
smp = phomt.sample(20000, random_state=SEED)
smp = smp[smp["vi_num"].str.strip() != ""]
sim = {
    "vi_asr": round(100 * jiwer.wer(list(smp["vi_num"]), list(smp["vi_asr"])), 2),
    "vi_rand": round(100 * jiwer.wer(list(smp["vi_num"]), list(smp["vi_rand"])), 2),
}
sim["% câu bị thay đổi vi_asr"] = pct(smp["vi_asr"] != smp["vi_num"])
sim["% câu bị thay đổi vi_rand"] = pct(smp["vi_rand"] != smp["vi_num"])
stats["simulated_wer_vs_vi_num"] = sim
stats["b0_real_dev_wer"] = B0_DEV_WER

json.dump(stats, open(f"{OUT}/eda_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# Biểu đồ
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#2a6fdb", "#e0833a", "#3a9d6e", "#8a6bbf"

fig, ax = plt.subplots(figsize=(6, 3.4))
for s, c in zip(["train", "dev", "test"], [C1, C2, C3]):
    ax.hist(fl[s]["duration"], bins=30, alpha=0.6, label=f"{s} (n={len(fl[s])})", color=c)
ax.set_xlabel("Độ dài clip (giây)"); ax.set_ylabel("Số clip"); ax.set_title("FLEURS: phân bố độ dài audio"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fleurs_duration.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.4))
x = range(len(LABELS))
ax.bar([i - 0.2 for i in x], fl_len.values, 0.4, label="FLEURS train", color=C1)
ax.bar([i + 0.2 for i in x], pm_len.values, 0.4, label="PhoMT 300k", color=C2)
for i, (a, b) in enumerate(zip(fl_len.values, pm_len.values)):
    ax.text(i - 0.2, a + 0.5, f"{a:.1f}%", ha="center", fontsize=8)
    ax.text(i + 0.2, b + 0.5, f"{b:.1f}%", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(LABELS); ax.set_ylabel("% câu"); ax.set_title("Độ dài câu: FLEURS và PhoMT"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/length_buckets.png", dpi=160); plt.close(fig)

keys = ["% câu có chữ số", "% câu có chữ hoa", "% câu có dấu câu"]
fig, ax = plt.subplots(figsize=(7, 3.8))
names = list(ba)
w = 0.2
for j, (n, c) in enumerate(zip(names, [C1, C3, C2, C4])):
    vals = [ba[n][k] for k in keys]
    ax.bar([i + (j - 1.5) * w for i in range(3)], vals, w, label=n, color=c)
ax.set_xticks(range(3)); ax.set_xticklabels(keys); ax.set_ylabel("% câu trong PhoMT train (300k)")
ax.set_title("Dữ liệu train trước và sau xử lý"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/before_after_train.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.4))
labs = ["vi_asr", "vi_rand", "B0 thật (dev)"]
vals = [sim["vi_asr"], sim["vi_rand"], B0_DEV_WER]
ax.bar(labs, vals, color=[C1, C2, C3])
for i, v in enumerate(vals):
    ax.text(i, v + 0.2, f"{v:.2f}%", ha="center")
ax.set_ylabel("WER (%)"); ax.set_title("Mức nhiễu mô phỏng so với lỗi ASR thật")
fig.tight_layout(); fig.savefig(f"{OUT}/noise_vs_real_wer.png", dpi=160); plt.close(fig)

# In tóm tắt định lượng để chép vào slide
d_pm, d_fl = dom["PhoMT"], dom["FLEURS (text gốc)"]
print("== Bảng nguồn ==")
print(src.drop(columns=["Trường dữ liệu"]).to_string(index=False))
print("\n== Chia tập FLEURS ==")
print(pd.DataFrame(split_rows).to_string(index=False))
print("\n== Giới tính FLEURS (%) ==", gender)
print("\n== Nhận xét định lượng (chép vào slide) ==")
print(f"- Chữ số: {d_pm['% câu có chữ số']}% câu PhoMT có chữ số, còn đầu ra ASR đọc số thành chữ; FLEURS gốc có {d_fl['% câu có chữ số']}%.")
print(f"- Chữ hoa: {d_pm['% câu có chữ hoa']}% câu PhoMT có chữ hoa; ASR trả về chữ thường, sau xử lý còn {ba['vi_num (đọc số + chuẩn hoá)']['% câu có chữ hoa']}%.")
print(f"- Dấu câu: {d_pm['% câu có dấu câu']}% câu PhoMT có dấu câu, sau xử lý còn {ba['vi_num (đọc số + chuẩn hoá)']['% câu có dấu câu']}%.")
print(f"- Dấu thanh kiểu mới: {d_pm['% câu có dấu thanh kiểu mới']}% (PhoMT, mẫu 20k) so với {d_fl['% câu có dấu thanh kiểu mới']}% (FLEURS).")
print(f"- Độ dài: trung vị {d_pm['số từ trung vị']} từ (PhoMT) so với {d_fl['số từ trung vị']} từ (FLEURS).")
print(f"- Nhiễu mô phỏng: vi_asr WER {sim['vi_asr']}% ({sim['% câu bị thay đổi vi_asr']}% câu đổi), vi_rand WER {sim['vi_rand']}% ({sim['% câu bị thay đổi vi_rand']}% câu đổi), B0 thật trên dev {B0_DEV_WER}%.")
print(f"\nĐã ghi vào {OUT}/")
