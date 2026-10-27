"""Regenerate the UTF-8 context-gate notebook without executing its cells."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))


md('''# 07 — Huấn luyện context gate V1

Baseline KEEP/REPLACE với history ASR tự sinh. V1 được giữ nguyên; chỉ gate logistic được học.

Chạy cell thiết lập, sau đó chọn **một** cách: pipeline đầy đủ hoặc từng bước. Không chạy đồng thời với job GPU khác.

TRAIN/CAL/DEV nội bộ tách theo video từ official TRAIN, đã loại video trùng official DEV/TEST/HARD. Đây là shared-history diagnostic, chưa phải closed-loop streaming. Xem `ASR/local/CONTEXT_GATE.md`.
''')
code('''from pathlib import Path
import sys, json
here = Path.cwd().resolve()
PROJECT = next((p if p.name == 'ASR' else p / 'ASR' for p in [here, *here.parents]
                if (p / 'scripts/context_gate_pipeline.py').exists() or (p / 'ASR/scripts/context_gate_pipeline.py').exists()), None)
assert PROJECT is not None, 'Không tìm thấy thư mục ASR'
PYTHON = PROJECT.parent / '.venv-asr/Scripts/python.exe'
assert PYTHON.exists()
sys.path.insert(0, str(PROJECT / 'local'))
from notebook_process import run_cli
RUN = PROJECT / 'local/context/gate_v1'
BASE = ['scripts/context_gate_pipeline.py']
def task(mode, *extra):
    run_cli(PROJECT, PYTHON, BASE + [mode, '--run-dir', str(RUN), *extra])
print('Run:', RUN)
''')
md('''## Cách A: chạy đầy đủ / tiếp tục sau gián đoạn
Cell này sinh nhãn trên GPU, rồi tự train gate và đánh giá. Các clip hoàn thành được lưu ngay. Chạy lại tiếp tục phần thiếu. Nếu đang có tiến trình nền, chỉ dùng cell theo dõi bên dưới.
''')
code("task('run', '--offline')\n")
md('''## Cách B: chạy từng bước
Không cần chạy phần này nếu đã dùng Cách A. Prepare không dùng GPU; generate là phần tốn thời gian; train gate dùng CPU.
''')
for mode in ('prepare', 'generate', 'train', 'evaluate'):
    code(f"task('{mode}'" + (", '--offline'" if mode == 'generate' else '') + ')\n')
md('''## Theo dõi / đọc kết quả
Cell này chỉ đọc file, dùng được trong khi tiến trình nền đang chạy. Nhãn benefit/harm dựa trên lỗi từ, chỉ tạo sau khi đóng băng ứng viên và feature.
''')
code('''from collections import Counter
pairs = []
if (RUN / 'pairs.jsonl').exists():
    for line in (RUN / 'pairs.jsonl').read_text(encoding='utf-8').splitlines():
        try: pairs.append(json.loads(line))
        except json.JSONDecodeError: print('Dòng cuối đang được ghi; đọc lại sau.')
print('Đã sinh:', len(pairs), Counter(r['partition'] for r in pairs))
print('Cải thiện:', sum(r['benefit'] for r in pairs), 'Gây hại:', sum(r['harm'] for r in pairs))
for name in ['prepared.json', 'evaluation.json']:
    if (RUN / name).exists():
        print(name)
        print((RUN / name).read_text(encoding='utf-8'))
''')
for i, cell in enumerate(cells):
    cell['id'] = f'gate-{i:02d}'
nb = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python', 'version': '3.10'}}, nbformat=4, nbformat_minor=5)
path = Path(__file__).resolve().parents[1]/'notebooks/local/07_context_gate.ipynb'
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
