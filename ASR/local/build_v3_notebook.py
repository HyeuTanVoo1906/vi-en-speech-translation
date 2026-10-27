"""Regenerate the UTF-8 V3 training notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 09 — V3: train lại từ PhoWhisper gốc trên dữ liệu V3

- ViMedCSS chia lại theo video (`data/ViMedCSS_vsplit`, 0 video chung giữa TRAIN/DEV/TEST). Split chính thức có 81,7% clip TRAIN
  thuộc video cũng nằm trong DEV/TEST/HARD, nên số ViMedCSS của V1/V2 lạc quan. **So V3 với B0V**, không so ViMedCSS với V1/V2.
- Loại khỏi TRAIN: clip im lặng và nhãn dày hơn 16 token/giây (`data/v3_train_exclude.json`).
- LR 3e-4 (V1 dùng 1e-3 bị nổ loss), 3 epoch (~2800 bước), eval DEV-subset batch 8.
- `DUTY = 1.0`: GPU chạy liên tục, không nghỉ (V1/V2 dùng 0,7). Giới hạn VRAM 4,5 GB giữ nguyên.

Chạy cell thiết lập, cell dữ liệu, cell kiểm tra, rồi bấm ▶ đúng cell tác vụ. Mỗi lần chỉ một job GPU. Interrupt dừng tiến trình con;
chạy lại cell train với cùng `RUN_ID` **và cùng `DUTY`** sẽ tiếp tục từ checkpoint gần nhất (đổi DUTY thì run bị từ chối vì cấu hình khác).
Giữ VS Code mở trong lúc train. Config: `ASR/experiments/V3_phowhisper_small_lora_vsplit/config.yaml`.
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

EXP = 'V3_phowhisper_small_lora_vsplit'
BASELINE = 'B0V_phowhisper_small_zeroshot_vsplit'
RUN_ID = 'local_v3'
DUTY = '1.0'
RUN = PROJECT/'local/runs'/EXP/RUN_ID
FINAL = RUN/'final'
DATA = PROJECT/'data'

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
md('''## Dữ liệu V3

Chỉ dùng CPU. Nếu đã có `data/ViMedCSS_vsplit` thì cell chỉ in báo cáo; dữ liệu gốc `data/ViMedCSS` không bị sửa.
''')
code('''if not (DATA/'ViMedCSS_vsplit/split_report.json').is_file():
    run_cli([PROJECT/'scripts/build_v3_data.py'])
report = json.loads((DATA/'ViMedCSS_vsplit/split_report.json').read_text(encoding='utf-8'))
print('ViMedCSS_vsplit:', json.dumps(report['splits'], ensure_ascii=False))
print('Video chung giữa các split:', report['video_overlap'])
print('Bỏ: trùng id', len(report['dropped_duplicate_ids']), '| rỗng/im lặng', len(report['dropped_silent_ids']),
      '| loại khỏi TRAIN', report['train_exclude_counts'])
''')
code("run_cli([RUNNER, 'check', EXP, '--run-id', RUN_ID])\n")
md('''## 1. Train

Smoke 2 bước trước, rồi train đầy đủ: ~2800 bước, ~17–18 giây/bước ở DUTY 1,0, cộng ~28 lần eval → khoảng 16 giờ. Cắm sạc, tắt sleep.
''')
code("run_cli([RUNNER, 'train', EXP, '--run-id', 'smoke_v3', '--smoke', '--duty-cycle', DUTY])\n")
code('''pid = gpu_job_running()
if pid:
    print(f'Đang có job GPU khác (PID {pid}); không khởi chạy thêm. Đợi job đó xong hoặc dừng nó trước.')
else:
    print('Train V3: bắt đầu mới hoặc tiếp tục từ', latest_checkpoint(RUN) or 'bước 0', flush=True)
    run_cli([RUNNER, 'train', EXP, '--run-id', RUN_ID, '--duty-cycle', DUTY])
    print('Adapter tốt nhất:', FINAL if (FINAL/'adapter_model.safetensors').is_file() else 'chưa có')
''')
md('''## Theo dõi

Chỉ đọc file. DEV-subset của V3 khác V1/V2 (ViMedCSS mới), nên `eval_wer` ở đây không so trực tiếp với 0,1339 của V1.
''')
code('''ckpt = latest_checkpoint(RUN)
if ckpt:
    state = json.loads((ckpt/'trainer_state.json').read_text(encoding='utf-8'))
    print('Checkpoint gần nhất:', ckpt.name, '| bước', state['global_step'], '/', state['max_steps'])
    for e in state['log_history']:
        if 'eval_wer' in e:
            print(f"  bước {e['step']:5d}  eval_wer={e['eval_wer']:.4f}  eval_loss={e['eval_loss']:.4f}")
logs = sorted((PROJECT/'local/logs').glob('*.log'))
mine = [p for p in logs if f'mode=train experiment={EXP}' in p.read_text(encoding='utf-8', errors='replace')[:300]]
if mine:
    lines = mine[-1].read_text(encoding='utf-8', errors='replace').replace('\\r', '\\n').splitlines()
    progress = [l for l in lines if l.startswith('[TRAIN] ') and 'microbatch' not in l]
    print('\\nLog:', mine[-1])
    print(*(progress[-3:] or lines[-5:]), sep='\\n')
''')
md('''## 2. Đánh giá trên DEV mới

V3 và B0V đều giải mã với guard (bỏ clip im lặng, giới hạn token theo độ dài audio). TEST/HARD vẫn khóa.
B0V chạy được ngay cả khi V3 chưa xong (khi GPU rảnh).
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V3 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'smoke_local_v3', '--max-per-source', '2', '--checkpoint', FINAL,
             '--duty-cycle', DUTY])
''')
code('''if not (FINAL/'adapter_model.safetensors').is_file():
    print('Chưa có', FINAL, '- train V3 xong trước.')
else:
    run_cli([RUNNER, 'eval', EXP, '--run-id', 'dev_local_v3', '--checkpoint', FINAL, '--duty-cycle', DUTY])
''')
code("run_cli([RUNNER, 'eval', BASELINE, '--run-id', 'dev_guard', '--duty-cycle', DUTY])\n")
code('''import pandas as pd
rows = {}
for name, path in [('B0V', PROJECT/'local/runs'/BASELINE/'dev_guard/eval/dev/metrics.json'),
                   ('V3', RUN.parent/'dev_local_v3/eval/dev/metrics.json')]:
    if path.is_file():
        m = json.loads(path.read_text(encoding='utf-8'))
        rows[name] = {'overall': 100*m['overall']['wer'],
                      **{s: 100*v['wer'] for s, v in m['by_source'].items()},
                      'term recall': 100*m['overall'].get('cs_term_recall', float('nan'))}
display(pd.DataFrame(rows).round(2).rename_axis('WER % (DEV V3, guarded decoding)'))
''')
md('''## 3. (Tùy chọn) V1/V2/B0/B1 với guard trên DEV cũ

Không ảnh hưởng V3. Mỗi model ~1 giờ; chạy khi GPU rảnh. Cho biết guard (bỏ clip im lặng, giới hạn token) thay đổi số cũ bao nhiêu.
''')
code('''for exp in ['V1G_phowhisper_small_lora_guard', 'V2G_phowhisper_small_lora_cont_guard',
            'B0G_phowhisper_small_zeroshot_guard', 'B1G_whisper_small_zeroshot_guard']:
    run_cli([RUNNER, 'eval', exp, '--run-id', 'dev_guard', '--duty-cycle', DUTY])
''')
for i, cell in enumerate(cells):
    cell['id'] = f'v3-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/09_V3.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
