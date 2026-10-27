"""Đo tốc độ và kích thước hệ thống cascade (VAD -> ASR -> MT) cho kết luận khả thi của Task 2.

Chạy từ gốc repo: python -m scripts.benchmark [--n 20] [--mt-adapter runs/mt/M4_s42/adapter] [--asr-adapter DIR]
Cần GPU (đóng các chương trình khác dùng GPU trước khi chạy). Kết quả: reports/benchmark.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from pipeline.inference import SpeechTranslator

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="số clip FLEURS dev dùng để đo")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--mt-adapter", default="runs/mt/M4_s42/adapter")
ap.add_argument("--asr-adapter", default=None)
args = ap.parse_args()

dev = torch.cuda.is_available()
sync = torch.cuda.synchronize if dev else (lambda: None)
info = {
    "gpu": torch.cuda.get_device_name(0) if dev else "CPU",
    "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if dev else None,
    "torch": torch.__version__,
    "mt_adapter": args.mt_adapter, "asr_adapter": args.asr_adapter,
}

# Khởi động (nạp model)
if dev:
    torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
st = SpeechTranslator(asr_adapter=args.asr_adapter, mt_adapter=args.mt_adapter)
sync()
info["load_time_s"] = round(time.perf_counter() - t0, 1)
info["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3) if dev else None


# Kích thước model: tìm mọi nn.Module trong đối tượng (không phụ thuộc tên thuộc tính)
def find_modules(obj, prefix="st", depth=0, seen=None):
    seen = set() if seen is None else seen
    out = []
    if id(obj) in seen or depth > 2:
        return out
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        out.append((prefix, obj))
        return out
    for k, v in getattr(obj, "__dict__", {}).items():
        out += find_modules(v, f"{prefix}.{k}", depth + 1, seen)
    return out


sizes = {}
for name, m in find_modules(st):
    n = sum(p.numel() for p in m.parameters())
    b = sum(p.numel() * p.element_size() for p in m.parameters())
    if n:
        sizes[name] = {"tham_so_trieu": round(n / 1e6, 1), "dung_luong_mb": round(b / 1e6, 1)}
info["models"] = sizes
if os.path.isdir(args.mt_adapter):
    info["mt_adapter_disk_mb"] = round(sum(os.path.getsize(os.path.join(args.mt_adapter, f)) for f in os.listdir(args.mt_adapter)) / 1e6, 1)

# Độ trễ trên n clip FLEURS dev
rows = [json.loads(l) for l in open("cascade/data/fleurs_dev.jsonl", encoding="utf-8")]
rows = rows[: args.n + args.warmup]
for r in rows[: args.warmup]:
    st.translate_file(r["audio_path"].replace("\\", os.sep))
if dev:
    torch.cuda.reset_peak_memory_stats()
per = []
for r in rows[args.warmup:]:
    sync()
    t = time.perf_counter()
    res = st.translate_file(r["audio_path"].replace("\\", os.sep))
    sync()
    wall = time.perf_counter() - t
    s = res["stats"]
    per.append({"audio_s": s["audio_s"], "wall_s": wall, "vad_s": s["vad_s"], "asr_s": s["asr_s"], "mt_s": s["mt_s"]})

a = np.array([p["audio_s"] for p in per]); w = np.array([p["wall_s"] for p in per])
res = {
    "so_clip": len(per),
    "tong_audio_s": round(float(a.sum()), 1), "tong_thoi_gian_xu_ly_s": round(float(w.sum()), 1),
    "RTF_tong": round(float(w.sum() / a.sum()), 3),
    "RTF_tung_khau": {k: round(float(sum(p[k] for p in per) / a.sum()), 3) for k in ["vad_s", "asr_s", "mt_s"]},
    "do_tre_moi_clip_s": {"trung_binh": round(float(w.mean()), 2), "p50": round(float(np.percentile(w, 50)), 2),
                          "p95": round(float(np.percentile(w, 95)), 2), "max": round(float(w.max()), 2)},
    "audio_trung_binh_s": round(float(a.mean()), 1),
    "vram_dinh_gb": {"allocated": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "reserved": round(torch.cuda.max_memory_reserved() / 1e9, 2)} if dev else None,
}
info["do_tre"] = res
os.makedirs("reports", exist_ok=True)
json.dump(info, open("reports/benchmark.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False, indent=1))
print("\nĐã ghi reports/benchmark.json")
