"""Regenerate the UTF-8 final TEST notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 11 — TEST cuối: hệ thống đã chốt V4B và các mốc so sánh

Hệ thống cuối **V4B** (V4 = V3 + SpecAugment, giải mã beam 5 + guard) đã được chốt trên DEV trước khi giải mã TEST:
`ASR/reports/final_freeze_20261027/freeze.json` (adapter sha256, cách giải mã, số liệu DEV của mọi ứng viên).

Mỗi ô giải mã TEST và HARD (lát ViMedCSS khó, nằm trong TEST) **một lần**: nếu đã có kết quả, ô chỉ in đường dẫn, không chạy lại.
Sau khi đọc TEST không chỉnh model hay cách giải mã nữa.

**Trước khi chạy:** tắt kernel của notebook 08, 09, 10 (mở notebook → Restart/Shut down kernel, hoặc đóng hẳn notebook) và đóng Chrome.
Mỗi kernel cũ giữ ~0,6 GB; máy chỉ còn ~3 GB bộ nhớ commit trống, không đủ để nạp PyTorch + model (hai lần chạy nền đã chết vì vậy).
Thời gian ước tính ở DUTY 1,0: V4B ~3 giờ, B0V/V3/V4 ~1,5 giờ mỗi model, V3B ~3 giờ (tùy chọn, để đủ bảng 2×2).
''')
code('''from pathlib import Path
import sys, json, ctypes
here = Path.cwd().resolve()
PROJECT = next((p for p in [here, *here.parents] if (p/'asr/config.py').is_file()), None)
if PROJECT is None:
    PROJECT = next((p/'ASR' for p in [here, *here.parents] if (p/'ASR/asr/config.py').is_file()), None)
assert PROJECT is not None, 'Mở notebook từ trong workspace dự án'
PYTHON = PROJECT.parent/'.venv-asr/Scripts/python.exe'
sys.path.insert(0, str(PROJECT/'local'))
from notebook_process import run_cli as _run_cli
def run_cli(arguments):
    return _run_cli(PROJECT, PYTHON, arguments)
RUNNER = PROJECT/'scripts/local_run.py'
RUNS = PROJECT/'local/runs'
DUTY = '1.0'

freeze = json.loads((PROJECT/'reports/final_freeze_20261027/freeze.json').read_text(encoding='utf-8'))
print('Hệ thống cuối:', freeze['final_system'], '| chốt lúc', freeze['frozen_at'])
print('Adapter sha256:', freeze['adapter']['adapter_model.safetensors'])
print('Giải mã:', freeze['decode'])
print('WER DEV (%):', {k: round(100*v['wer'], 2) for k, v in freeze['dev_evidence'].items()})

class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong), ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

def commit_free_gb():
    s = _MemStatus(); s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / 1e9

free = commit_free_gb()
print(f'Bộ nhớ commit trống: {free:.1f} GB', '' if free >= 5 else '-> THẤP: tắt kernel notebook cũ và Chrome rồi chạy lại cell này')

def gpu_job_running():
    import psutil
    lock = PROJECT/'local/gpu.lock'
    if not lock.exists():
        return None
    pid = int(lock.read_text())
    return pid if psutil.pid_exists(pid) else None

def decode_once(exp, split, run_id, *extra):
    """Decode one split once; skip if its metrics already exist (TEST is not re-run)."""
    out = RUNS/exp/run_id/'eval'/split/'metrics.json'
    if out.is_file():
        print('Đã có, không chạy lại:', out); return
    if gpu_job_running():
        print('Đang có job GPU khác; đợi nó xong.'); return
    if commit_free_gb() < 4:
        print(f'Bộ nhớ commit chỉ còn {commit_free_gb():.1f} GB; tắt bớt ứng dụng/kernel rồi chạy lại cell.'); return
    run_cli([RUNNER, 'eval', exp, '--split', split, '--allow-test', '--run-id', run_id, '--duty-cycle', DUTY, *extra])

V3_FINAL = RUNS/'V3_phowhisper_small_lora_vsplit/local_v3/final'
V4_FINAL = RUNS/'V4_phowhisper_small_lora_vsplit_specaug/local_v4/final'
''')
md('''## 1. Hệ thống cuối V4B
''')
code('''decode_once('V4B_phowhisper_small_lora_vsplit_specaug_beam5', 'test', 'test_beam5')
decode_once('V4B_phowhisper_small_lora_vsplit_specaug_beam5', 'hard', 'hard_beam5')
''')
md('''## 2. Mốc so sánh: B0V (zero-shot), V3, V4 (greedy)
''')
code('''decode_once('B0V_phowhisper_small_zeroshot_vsplit', 'test', 'test_guard')
decode_once('B0V_phowhisper_small_zeroshot_vsplit', 'hard', 'hard_guard')
''')
code('''decode_once('V3_phowhisper_small_lora_vsplit', 'test', 'test_local_v3', '--checkpoint', V3_FINAL)
decode_once('V3_phowhisper_small_lora_vsplit', 'hard', 'hard_local_v3', '--checkpoint', V3_FINAL)
''')
code('''decode_once('V4_phowhisper_small_lora_vsplit_specaug', 'test', 'test_local_v4', '--checkpoint', V4_FINAL)
decode_once('V4_phowhisper_small_lora_vsplit_specaug', 'hard', 'hard_local_v4', '--checkpoint', V4_FINAL)
''')
md('''## 3. (Tùy chọn) V3B — đủ bảng 2×2 SpecAugment × beam
''')
code('''decode_once('V3B_phowhisper_small_lora_vsplit_beam5', 'test', 'test_beam5')
decode_once('V3B_phowhisper_small_lora_vsplit_beam5', 'hard', 'hard_beam5')
''')
md('''## Kết quả TEST / HARD
''')
code('''import pandas as pd
SYSTEMS = {'B0V': ('B0V_phowhisper_small_zeroshot_vsplit', 'test_guard', 'hard_guard'),
           'V3': ('V3_phowhisper_small_lora_vsplit', 'test_local_v3', 'hard_local_v3'),
           'V3B': ('V3B_phowhisper_small_lora_vsplit_beam5', 'test_beam5', 'hard_beam5'),
           'V4': ('V4_phowhisper_small_lora_vsplit_specaug', 'test_local_v4', 'hard_local_v4'),
           'V4B (cuối)': ('V4B_phowhisper_small_lora_vsplit_specaug_beam5', 'test_beam5', 'hard_beam5')}
rows = {}
for name, (exp, test_run, hard_run) in SYSTEMS.items():
    test, hard = RUNS/exp/test_run/'eval/test/metrics.json', RUNS/exp/hard_run/'eval/hard/metrics.json'
    if not test.is_file():
        continue
    m = json.loads(test.read_text(encoding='utf-8'))
    row = {'TEST overall': 100*m['overall']['wer'], **{f'TEST {s}': 100*v['wer'] for s, v in m['by_source'].items()},
           'TEST term recall': 100*m['overall'].get('cs_term_recall', float('nan'))}
    if hard.is_file():
        h = json.loads(hard.read_text(encoding='utf-8'))
        row |= {'HARD WER': 100*h['overall']['wer'], 'HARD term recall': 100*h['overall'].get('cs_term_recall', float('nan'))}
    rows[name] = row
display(pd.DataFrame(rows).round(2))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'test-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/11_TEST.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
