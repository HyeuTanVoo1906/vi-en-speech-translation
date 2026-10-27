"""Regenerate the UTF-8 V4 training notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 10 — V4: V3 + SpecAugment, và beam search

- **V4** = công thức V3 giữ nguyên (dữ liệu V3, LR 3e-4, 3 epoch, LoRA, seed) + **SpecAugment** khi train: che ~5% khung thời gian
  (đoạn 100 ms) và đôi khi một dải 10 kênh mel. V3 chạm đáy ở bước 1800 rồi đi ngang; SpecAugment là regularization để vượt điểm đó.
  Chỉ đổi một biến nên V4 so V3 cho biết đúng tác dụng của SpecAugment. Eval không bị che.
- **Beam search 5** (V3B, V4B): chỉ giải mã lại, không train. Thường giảm WER so với greedy; tốn thời gian giải mã hơn.

Thứ tự gợi ý: train V4 (~16 giờ) → eval V4 → beam V3B/V4B (mỗi model ~2–3 giờ). Beam V3B chạy được bất cứ lúc nào GPU rảnh.
Mỗi lần chỉ một job GPU. Chạy lại cell train với cùng `RUN_ID` **và cùng `DUTY`** để tiếp tục từ checkpoint gần nhất.
Giữ VS Code mở, cắm sạc, tắt sleep, đóng bớt Chrome/Zalo (bộ nhớ commit của máy chỉ còn ~4 GB trống).
''')
code('''from pathlib import Path
import sys, json
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

EXP = 'V4_phowhisper_small_lora_vsplit_specaug'
RUN_ID = 'local_v4'
DUTY = '1.0'
RUN = RUNS/EXP/RUN_ID
FINAL = RUN/'final'
V3_RUN = RUNS/'V3_phowhisper_small_lora_vsplit/local_v3'
assert (PROJECT/'data/ViMedCSS_vsplit/train.parquet').is_file(), 'Thiếu dữ liệu V3: chạy notebook 09 phần Dữ liệu V3'

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

print('Run:', RUN)
print('Checkpoint gần nhất:', latest_checkpoint(RUN) or 'chưa có (train sẽ bắt đầu từ bước 0)')
print('Job GPU đang chạy:', gpu_job_running() or 'không')
''')
code("run_cli([RUNNER, 'check', EXP, '--run-id', RUN_ID])\n")
md('''## 1. Train V4

Smoke 2 bước trước (đã chạy đạt một lần thì cell báo đã hoàn tất), rồi train đầy đủ: ~2800 bước ở DUTY 1,0 → khoảng 16 giờ.
''')
code("run_cli([RUNNER, 'train', EXP, '--run-id', 'smoke_v4', '--smoke', '--duty-cycle', DUTY])\n")
code('''pid = gpu_job_running()
if pid:
    print(f'Đang có job GPU khác (PID {pid}); không khởi chạy thêm. Đợi job đó xong hoặc dừng nó trước.')
else:
    print('Train V4: bắt đầu mới hoặc tiếp tục từ', latest_checkpoint(RUN) or 'bước 0', flush=True)
    run_cli([RUNNER, 'train', EXP, '--run-id', RUN_ID, '--duty-cycle', DUTY])
    print('Adapter tốt nhất:', FINAL if (FINAL/'adapter_model.safetensors').is_file() else 'chưa có')
''')
md('''## Theo dõi: V4 so với V3 trên cùng DEV-subset

Chỉ đọc file. Cùng dữ liệu, cùng 600 clip DEV-subset, nên hai đường `eval_wer` so trực tiếp được (V3 tốt nhất 0,1199 ở bước 1800).
''')
code('''def eval_curve(run):
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        return {}
    state = json.loads((ckpt/'trainer_state.json').read_text(encoding='utf-8'))
    return {e['step']: e['eval_wer'] for e in state['log_history'] if 'eval_wer' in e}

v3, v4 = eval_curve(V3_RUN), eval_curve(RUN)
print('bước   V3_wer  V4_wer')
for step in sorted(v4):
    print(f"{step:5d}  {v3.get(step, float('nan')):.4f}  {v4[step]:.4f}")
if v4:
    print('Tốt nhất: V3', round(min(v3.values()), 4), '| V4', round(min(v4.values()), 4))
logs = sorted((PROJECT/'local/logs').glob('*.log'))
mine = [p for p in logs if f'mode=train experiment={EXP}' in p.read_text(encoding='utf-8', errors='replace')[:300]]
if mine:
    lines = mine[-1].read_text(encoding='utf-8', errors='replace').replace('\\r', '\\n').splitlines()
    progress = [l for l in lines if l.startswith('[TRAIN] ') and 'microbatch' not in l]
    print('\\nLog:', mine[-1])
    print(*(progress[-3:] or lines[-5:]), sep='\\n')
''')
md('''## 2. Đánh giá V4 trên DEV V3 (greedy, có guard)
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V4 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'smoke_local_v4', '--max-per-source', '2', '--checkpoint', FINAL,
             '--duty-cycle', DUTY])
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V4 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'dev_local_v4', '--checkpoint', FINAL, '--duty-cycle', DUTY])
''')
md('''## 3. Beam search 5 (chỉ giải mã lại)

V3B chạy được ngay khi GPU rảnh; V4B cần V4 train xong. Mỗi model ~2–3 giờ.
''')
code("run_cli([RUNNER, 'eval', 'V3B_phowhisper_small_lora_vsplit_beam5', '--run-id', 'dev_beam5', '--duty-cycle', DUTY])\n")
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V4 xong trước.')
else:
    run_cli([RUNNER, 'eval', 'V4B_phowhisper_small_lora_vsplit_specaug_beam5', '--run-id', 'dev_beam5', '--duty-cycle', DUTY])
''')
md('''## So sánh trên DEV V3
''')
code('''import pandas as pd
rows = {}
for name, path in [('B0V', RUNS/'B0V_phowhisper_small_zeroshot_vsplit/dev_guard'),
                   ('V3', RUNS/'V3_phowhisper_small_lora_vsplit/dev_local_v3'),
                   ('V3B beam5', RUNS/'V3B_phowhisper_small_lora_vsplit_beam5/dev_beam5'),
                   ('V4', RUNS/EXP/'dev_local_v4'),
                   ('V4B beam5', RUNS/'V4B_phowhisper_small_lora_vsplit_specaug_beam5/dev_beam5')]:
    path = path/'eval/dev/metrics.json'
    if path.is_file():
        m = json.loads(path.read_text(encoding='utf-8'))
        rows[name] = {'overall': 100*m['overall']['wer'],
                      **{s: 100*v['wer'] for s, v in m['by_source'].items()},
                      'term recall': 100*m['overall'].get('cs_term_recall', float('nan')),
                      'RTF': m['rtf_decode_only']}
display(pd.DataFrame(rows).round(3).rename_axis('WER % (DEV V3, guarded decoding)'))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'v4-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/10_V4.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
