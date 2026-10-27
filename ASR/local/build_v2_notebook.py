"""Regenerate the UTF-8 V2 training notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 08 — V2: train tiếp V1 (PhoWhisper-small LoRA, lr 1e-4)

Khởi tạo từ adapter tốt nhất của V1 (`local_v1/final`, checkpoint-2600), train thêm 2 epoch với LR 1e-4 thay vì 1e-3.
V1 bị nổ loss ở bước 660–900 vì LR cao và chỉ tốt lên khi LR đã giảm. Dữ liệu, LoRA và 600 clip DEV-subset giữ như V1,
nên `eval_wer` so trực tiếp được với V1 (best 0,1339). Eval trong lúc train dùng batch 8 (V1 dùng 1, tốn ~50% thời gian).

Chạy cell thiết lập và cell kiểm tra, rồi bấm ▶ đúng cell tác vụ. Mỗi lần chỉ một job GPU. Bấm Interrupt sẽ dừng tiến trình con;
chạy lại cell train với cùng `RUN_ID` sẽ tiếp tục từ checkpoint gần nhất (lưu mỗi 100 bước hoặc mỗi 10 phút).
Giữ VS Code mở trong suốt lúc train: kernel tắt thì tiến trình train cũng dừng. Config: `ASR/experiments/V2_phowhisper_small_lora_cont/config.yaml`.
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

EXP = 'V2_phowhisper_small_lora_cont'
RUN_ID = 'local_v2'
RUN = PROJECT/'local/runs'/EXP/RUN_ID
FINAL = RUN/'final'
INIT = PROJECT/'local/runs/V1_phowhisper_small_lora/local_v1/final'
V1_RUN = PROJECT/'local/runs/V1_phowhisper_small_lora/local_v1'
assert (INIT/'adapter_model.safetensors').is_file(), f'Thiếu adapter V1: {INIT}'

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

print('Project:', PROJECT)
print('Run:', RUN)
print('Khởi tạo từ:', INIT)
print('Checkpoint gần nhất:', latest_checkpoint(RUN) or 'chưa có (train sẽ bắt đầu từ bước 0)')
print('Job GPU đang chạy:', gpu_job_running() or 'không')
''')
code("run_cli([RUNNER, 'check', EXP, '--run-id', RUN_ID])\n")
md('''## 1. Train

Smoke 2 bước chỉ để kiểm tra (đã chạy đạt một lần; chạy lại sẽ báo đã hoàn tất). Cell train đầy đủ: khoảng 1816 bước,
~25 giây/bước với duty cycle 0,7, cộng 18 lần eval ~6 phút → khoảng 13–14 giờ. Cắm sạc và tắt sleep.
''')
code("run_cli([RUNNER, 'train', EXP, '--run-id', 'smoke_v2', '--smoke'])\n")
code('''pid = gpu_job_running()
if pid:
    print(f'Đang có job GPU khác (PID {pid}); không khởi chạy thêm. Dùng cell theo dõi bên dưới, hoặc dừng job đó trước.')
else:
    print('Train V2: bắt đầu mới hoặc tiếp tục từ', latest_checkpoint(RUN) or 'bước 0', flush=True)
    run_cli([RUNNER, 'train', EXP, '--run-id', RUN_ID])
    print('Adapter tốt nhất:', FINAL if (FINAL/'adapter_model.safetensors').is_file() else 'chưa có')
''')
md('''## Theo dõi

Chỉ đọc file. Dùng khi train đang chạy nền, hoặc sau khi bị gián đoạn. So `eval_wer` của V2 với V1 trên cùng 600 clip DEV-subset.
''')
code('''def eval_history(run):
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        return ckpt, []
    state = json.loads((ckpt/'trainer_state.json').read_text(encoding='utf-8'))
    return ckpt, [(e['step'], e['eval_wer']) for e in state['log_history'] if 'eval_wer' in e]

_, v1 = eval_history(V1_RUN)
ckpt, v2 = eval_history(RUN)
print('V1 best eval_wer:', min(w for _, w in v1) if v1 else None)
print('V2 checkpoint gần nhất:', ckpt)
for step, wer in v2:
    print(f'  bước {step:5d}  eval_wer={wer:.4f}')

logs = sorted((PROJECT/'local/logs').glob('*.log'))
mine = [p for p in logs if f'experiment={EXP}' in p.read_text(encoding='utf-8', errors='replace')[:300]]
if mine:
    lines = mine[-1].read_text(encoding='utf-8', errors='replace').replace('\\r', '\\n').splitlines()
    progress = [l for l in lines if l.startswith('[TRAIN] ') and 'microbatch' not in l]
    print('\\nLog:', mine[-1])
    print(*(progress[-3:] or lines[-5:]), sep='\\n')
''')
md('''## 2. Đánh giá

Chạy sau khi train xong (`local_v2/final` có adapter). Đánh giá thử 2 clip/nguồn, rồi toàn bộ DEV. TEST/HARD vẫn khóa.
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V2 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'smoke_local_v2', '--max-per-source', '2', '--checkpoint', FINAL])
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V2 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'dev_local_v2', '--checkpoint', FINAL])
''')
code('''import pandas as pd
rows = {}
for name, path in [('V1', PROJECT/'local/runs/V1_phowhisper_small_lora/dev_local_v1/eval/dev/metrics.json'),
                   ('V2', PROJECT/'local/runs'/EXP/'dev_local_v2/eval/dev/metrics.json')]:
    if path.is_file():
        m = json.loads(path.read_text(encoding='utf-8'))
        rows[name] = {'overall': 100*m['overall']['wer'],
                      **{s: 100*v['wer'] for s, v in m['by_source'].items()},
                      'term recall': 100*m['overall'].get('cs_term_recall', float('nan'))}
display(pd.DataFrame(rows).round(2).rename_axis('WER % (full DEV)'))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'v2-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/08_V2.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
