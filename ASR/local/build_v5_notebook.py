"""Regenerate the UTF-8 V5 training notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 12 — V5: PhoWhisper-medium + LoRA (công thức V4, backbone lớn hơn)

- **V5** = công thức V4 (dữ liệu V3 chia theo video, LoRA r32, SpecAugment, seed) trên **PhoWhisper-medium** (769M tham số, small là 244M).
  Bằng chứng: bài PhoWhisper cho VIVOS small 6,33 → medium 4,97 → large 4,67; bài ViMedCSS zero-shot small 36,31 → large 31,24.
- Khác V4 (bắt buộc vì GPU 6 GB và thời gian): trọng số gốc fp16 (`local/models/PhoWhisper-medium-fp16`), lr 2e-4, **2 epoch** (~1870 bước),
  micro-batch 2 × tích lũy 16 (batch hiệu dụng vẫn 32), eval DEV-subset 4 lần (bước 467, 934, 1401, 1868).
- Đo trên pipeline thật (batch 2 × 16): ~37 s/bước → **khoảng 19 giờ train + ~2,7 giờ cho 4 lần eval (~40 phút/lần) ≈ 22 giờ**. VRAM tổng ~3,95 GB, RAM ~1,7 GB.
- **V5B** = V5 giải mã beam 5 (chỉ eval). Chọn hệ thống cuối bằng DEV, không nhìn TEST (TEST của V3 chia theo video đã được xem).

Mỗi lần chỉ một job GPU. Chạy lại cell train với cùng `RUN_ID` **và cùng `DUTY`** để tiếp tục từ checkpoint gần nhất.
Trước khi train: tắt kernel notebook 08–11 đang rảnh, đóng Chrome/Zalo, cắm sạc, tắt sleep. Ổ C: chỉ còn vài GB (pagefile nằm ở C:).
''')
code('''from pathlib import Path
import sys, json, ctypes
here = Path.cwd().resolve()
PROJECT = next((p for p in [here, *here.parents] if (p/'asr/config.py').is_file()), None)
if PROJECT is None:
    PROJECT = next((p/'ASR' for p in [here, *here.parents] if (p/'ASR/asr/config.py').is_file()), None)
assert PROJECT is not None, 'Mở notebook từ trong workspace dự án'
PYTHON = PROJECT.parent/'.venv-asr/Scripts/python.exe'
assert PYTHON.is_file(), 'Chạy ASR/local/setup.ps1 trước'
sys.path.insert(0, str(PROJECT/'local'))
from notebook_process import run_cli as _run_cli
def run_cli(arguments):
    return _run_cli(PROJECT, PYTHON, arguments)
RUNNER = PROJECT/'scripts/local_run.py'
RUNS = PROJECT/'local/runs'

EXP = 'V5_phowhisper_medium_lora_vsplit_specaug'
EXP_B = 'V5B_phowhisper_medium_lora_vsplit_specaug_beam5'
RUN_ID = 'local_v5'
DUTY = '1.0'
RUN = RUNS/EXP/RUN_ID
FINAL = RUN/'final'
V4_RUN = RUNS/'V4_phowhisper_small_lora_vsplit_specaug/local_v4'
assert (PROJECT/'data/ViMedCSS_vsplit/train.parquet').is_file(), 'Thiếu dữ liệu V3: chạy notebook 09 phần Dữ liệu V3'
assert (PROJECT/'local/models/PhoWhisper-medium-fp16/model.safetensors').is_file(), 'Thiếu trọng số medium fp16 (local/models)'

class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

def commit_free_gb():
    s = _MemStatus(); s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / 1e9

def gpu_job_running():
    """PID của job GPU local đang chạy (giữ local/gpu.lock), hoặc None."""
    import psutil
    lock = PROJECT/'local/gpu.lock'
    if not lock.exists():
        return None
    pid = int(lock.read_text())
    return pid if psutil.pid_exists(pid) else None

def latest_checkpoint(run):
    ckpts = sorted((run/'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    return ckpts[-1] if ckpts else None

free = commit_free_gb()
print('Run:', RUN)
print('Checkpoint gần nhất:', latest_checkpoint(RUN) or 'chưa có (train sẽ bắt đầu từ bước 0)')
print('Job GPU đang chạy:', gpu_job_running() or 'không')
print(f'Bộ nhớ commit trống: {free:.1f} GB', '' if free >= 4 else '-> THẤP: tắt kernel notebook cũ và Chrome rồi chạy lại cell này')
''')
code("run_cli([RUNNER, 'check', EXP, '--run-id', RUN_ID])\n")
md('''## 1. Train V5

Smoke 2 bước trước (đã đạt với cấu hình cuối, nên cell chỉ báo hoàn tất), rồi train đầy đủ 1868 bước ở DUTY 1,0.
''')
code("run_cli([RUNNER, 'train', EXP, '--run-id', 'smoke_v5', '--smoke', '--duty-cycle', DUTY])\n")
code('''pid = gpu_job_running()
if pid:
    print(f'Đang có job GPU khác (PID {pid}); không khởi chạy thêm. Đợi job đó xong hoặc dừng nó trước.')
elif commit_free_gb() < 4:
    print(f'Bộ nhớ commit chỉ còn {commit_free_gb():.1f} GB; tắt bớt ứng dụng/kernel rồi chạy lại cell.')
else:
    print('Train V5: bắt đầu mới hoặc tiếp tục từ', latest_checkpoint(RUN) or 'bước 0', flush=True)
    run_cli([RUNNER, 'train', EXP, '--run-id', RUN_ID, '--duty-cycle', DUTY])
    print('Adapter tốt nhất:', FINAL if (FINAL/'adapter_model.safetensors').is_file() else 'chưa có')
''')
md('''## Theo dõi: V5 so với V4 trên cùng DEV-subset

Chỉ đọc file. Cùng 600 clip DEV-subset; V4 tốt nhất 0,1191 (bước 2600). V5 eval ở bước 467, 934, 1401, 1868.
''')
code('''def eval_curve(run):
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        return {}
    state = json.loads((ckpt/'trainer_state.json').read_text(encoding='utf-8'))
    return {e['step']: e['eval_wer'] for e in state['log_history'] if 'eval_wer' in e}

v4, v5 = eval_curve(V4_RUN), eval_curve(RUN)
print('bước   V4_wer  V5_wer')
for step in sorted(v5):
    print(f"{step:5d}  {v4.get(step, float('nan')):.4f}  {v5[step]:.4f}")
if v5:
    print('Tốt nhất: V4', round(min(v4.values()), 4), '| V5', round(min(v5.values()), 4))
logs = sorted((PROJECT/'local/logs').glob('*.log'))
mine = [p for p in logs if f'mode=train experiment={EXP}' in p.read_text(encoding='utf-8', errors='replace')[:300]]
if mine:
    lines = mine[-1].read_text(encoding='utf-8', errors='replace').replace('\\r', '\\n').splitlines()
    progress = [l for l in lines if l.startswith('[TRAIN] ') and 'microbatch' not in l]
    print('\\nLog:', mine[-1])
    print(*(progress[-3:] or lines[-5:]), sep='\\n')
''')
md('''## 2. Đánh giá V5 trên DEV V3 (greedy, có guard)
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V5 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'smoke_local_v5', '--max-per-source', '2', '--checkpoint', FINAL,
             '--duty-cycle', DUTY])
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V5 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'dev_local_v5', '--checkpoint', FINAL, '--duty-cycle', DUTY])
''')
md('''## 3. Beam search 5 (V5B, chỉ giải mã lại)

Medium giải mã chậm hơn small khoảng 2 lần: dự kiến greedy ~3 giờ, beam ~5–6 giờ cho DEV.
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V5 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP_B, '--run-id', 'dev_beam5', '--duty-cycle', DUTY])
''')
md('''## So sánh trên DEV V3
''')
code('''import pandas as pd
rows = {}
for name, path in [('B0V', RUNS/'B0V_phowhisper_small_zeroshot_vsplit/dev_guard'),
                   ('V4', RUNS/'V4_phowhisper_small_lora_vsplit_specaug/dev_local_v4'),
                   ('V4B beam5', RUNS/'V4B_phowhisper_small_lora_vsplit_specaug_beam5/dev_beam5'),
                   ('V5', RUNS/EXP/'dev_local_v5'),
                   ('V5B beam5', RUNS/EXP_B/'dev_beam5')]:
    path = path/'eval/dev/metrics.json'
    if path.is_file():
        m = json.loads(path.read_text(encoding='utf-8'))
        rows[name] = {'overall': 100*m['overall']['wer'],
                      **{s: 100*v['wer'] for s, v in m['by_source'].items()},
                      'term recall': 100*m['overall'].get('cs_term_recall', float('nan')),
                      'truncated': m['overall']['n_truncated'],
                      'RTF': m['rtf_decode_only']}
display(pd.DataFrame(rows).round(3).rename_axis('WER % (DEV V3, guarded decoding)'))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'v5-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/12_V5.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
