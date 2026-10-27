"""Xuất loss curve từ log TensorBoard của các lần train MT (runs/mt/*/tb).

Chạy từ gốc repo: python -m scripts.plot_curves
Kết quả: reports/curves/<run>_scalars.csv, <run>_loss.png, versions_train_loss.png. Không cần GPU.
"""
import glob
import os

import matplotlib
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs/mt"
OUT = "reports/curves"
os.makedirs(OUT, exist_ok=True)
NAMES = {
    "M1_s42": "V1 (M1: PhoMT gốc)",
    "M2_s42": "V2 (M2: chuẩn hoá)",
    "M2n_s42": "V3 (M2n: đọc số thành chữ)",
    "M4_s42": "V4 (M4: nhiễu kiểu ASR)",
    "M3_s42": "Đối chứng M3 (nhiễu ngẫu nhiên)",
}
ORDER = ["M1_s42", "M2_s42", "M2n_s42", "M4_s42", "M3_s42"]
COLORS = ["#2a6fdb", "#e0833a", "#3a9d6e", "#8a6bbf", "#888888"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(run):
    files = glob.glob(f"{RUNS}/{run}/tb/events*")
    if not files:
        return None
    ea = EventAccumulator(os.path.dirname(files[0]), size_guidance={"scalars": 0})
    ea.Reload()
    rows = []
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            rows.append({"tag": tag, "step": e.step, "value": e.value})
    return pd.DataFrame(rows)


def is_dev(tag):
    t = tag.lower()
    return any(k in t for k in ("dev", "eval", "val"))


curves = {}
for run in sorted(os.listdir(RUNS)):
    if run == "smoke":
        continue
    df = load(run)
    if df is None or df.empty:
        continue
    df.to_csv(f"{OUT}/{run}_scalars.csv", index=False)
    loss_tags = sorted(t for t in df["tag"].unique() if "loss" in t.lower())
    print(f"{run}: tags = {sorted(df['tag'].unique())}")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for tag in loss_tags:
        d = df[df["tag"] == tag].sort_values("step")
        if is_dev(tag):
            ax.plot(d["step"], d["value"], "o-", color="#e0833a", label=tag, ms=4)
        else:
            ax.plot(d["step"], d["value"], color="#2a6fdb", alpha=0.25, lw=1)
            sm = d["value"].ewm(span=20).mean()
            ax.plot(d["step"], sm, color="#2a6fdb", lw=1.8, label=f"{tag} (làm mượt)")
            curves[run] = (d["step"].values, sm.values)
    ax.set_xlabel("Bước huấn luyện"); ax.set_ylabel("Loss"); ax.set_title(f"{run}: loss theo bước")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/{run}_loss.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for run, c in zip(ORDER, COLORS):
    if run in curves:
        ax.plot(*curves[run], color=c, lw=1.8, label=NAMES[run])
ax.set_xlabel("Bước huấn luyện"); ax.set_ylabel("Train loss (làm mượt)")
ax.set_title("Train loss các phiên bản MT (cột nguồn khác nhau nên không so trực tiếp mức loss)")
ax.title.set_fontsize(9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/versions_train_loss.png", dpi=160); plt.close(fig)
print(f"\nĐã ghi vào {OUT}/")
