"""Generate small orchestration notebooks; model code runs in a fresh child process."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/local'
OUT.mkdir(parents=True, exist_ok=True)


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True))


common = '''from pathlib import Path
import sys, json
here = Path.cwd().resolve()
PROJECT = next((p for p in [here, *here.parents] if (p/'asr/config.py').is_file()), None)
if PROJECT is None:
    PROJECT = next((p/'ASR' for p in [here, *here.parents] if (p/'ASR/asr/config.py').is_file()), None)
assert PROJECT is not None, 'Open notebook from the project workspace'
PYTHON = PROJECT.parent/'.venv-asr/Scripts/python.exe'
DATA = PROJECT/'data'
RAW_DATA = PROJECT.parent/'dataset/ASR'
assert PYTHON.is_file(), 'Run ASR/local/setup.ps1 first'
sys.path.insert(0, str(PROJECT/'local'))
from notebook_process import run_cli as _run_cli
def run_cli(arguments):
    return _run_cli(PROJECT, PYTHON, arguments)
RUNNER = PROJECT/'scripts/local_run.py'
print('Project:', PROJECT)
print('Python:', PYTHON)
'''


def save(name, cells):
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}},
                    cells=cells)
    for i, cell in enumerate(cells):
        cell['id'] = f'cell-{i:03d}'
    (OUT/name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')


save('00_setup.ipynb', [
    md('# Local setup / data audit\nRun `powershell -ExecutionPolicy Bypass -File ASR/local/setup.ps1` from the workspace first. GPU work runs in a separate process and releases memory on exit. Default budget: 4.5 decimal GB total estimated VRAM; duty cycle 0.70 is pacing, not a hardware utilization cap.'),
    code(common), code("run_cli([RUNNER, 'check'])"),
    md('## Prepare all four datasets\nRUN_PREPARE defaults to False. Alternatively copy prepared parquet files and registry.json from Colab into ASR/data. Existing raw datasets are never moved or deleted.'),
    code("RUN_PREPARE = False\nif RUN_PREPARE:\n    run_cli([PROJECT/'scripts/prepare_data.py', '--raw-dir', RAW_DATA, '--out-dir', DATA])"),
    code("import pyarrow.parquet as pq\nfor source in ['VIVOS','FLEURS','VietMed','ViMedCSS']:\n    for split in ['train','dev','test']:\n        p = DATA/source/f'{split}.parquet'\n        print(source, split, pq.ParquetFile(p).metadata.num_rows if p.is_file() else 'MISSING')"),
    md('## Import Colab artifacts\nCopy the entire V1 `final` directory into `ASR/experiments/V1_phowhisper_small_lora/final`. Copy Colab reports into `ASR/local/imports/reports`, full audio into `ASR/local/imports/original_audio`. Paths embedded in imported CSV files still need remapping. `final` is an adapter for inference; optimizer resume requires checkpoint directories as well. Local runs never overwrite imported final.'),
])

for prefix, exp in [('01_B0','B0_phowhisper_small_zeroshot'), ('02_B1','B1_whisper_small_zeroshot'), ('03_V1','V1_phowhisper_small_lora')]:
    # V1's checkpoint-aware train/evaluate workflow is maintained in that notebook.
    # Preserve it (and any user run outputs) when refreshing the other templates.
    if prefix == '03_V1' and (OUT/'03_V1.ipynb').exists():
        continue
    cells = [md(f'# {exp}\nRun task cells individually: each starts when you press Run. B0/B1 are evaluation-only. In V1, run setup/configuration then choose evaluation OR training; do not Run All unless you intend both workflows.'), code(common),
             code(f"EXP = '{exp}'\nCHECKPOINT = None  # V1 defaults to experiments/<EXP>/final\n# To evaluate local training instead:\n# CHECKPOINT = PROJECT/'local/runs'/EXP/'local_v1/final'\nrun_cli([RUNNER, 'check', EXP])"),
             code("eval_options = ['--checkpoint', str(CHECKPOINT)] if CHECKPOINT else []\nprint('SMOKE EVALUATION: up to 2 clips per source', flush=True)\nrun_cli([RUNNER, 'eval', EXP, '--run-id', 'smoke_eval', '--max-per-source', '2', *eval_options])"),
             md('## Full DEV evaluation\nRun the following cell after smoke passes. Choose a new run-id when changing profile/checkpoint. TEST/HARD require an explicit final-evaluation command; do not tune on them.'),
             code("print('FULL DEV EVALUATION: all DEV clips', flush=True)\nrun_cli([RUNNER, 'eval', EXP, '--run-id', 'dev_local_v1', *eval_options])")]
    if prefix == '03_V1':
        cells += [md('## Optional local LoRA training\nFresh training or resume of this local run, NOT automatic continuation of Colab final. Batch 1 × accumulation 32. Two-step smoke uses a tiny subset; keep its checkpoints separate. Periodic save every 100 optimizer steps or 10 minutes at a step boundary; optimizer/scheduler/RNG/FP16 scaler included, best + latest retained. Hard termination loses steps since the last completed save.'),
                  code("print('TRAIN SMOKE: 2 optimizer steps, diagnostic only', flush=True)\nrun_cli([RUNNER, 'train', EXP, '--run-id', 'train_smoke', '--smoke'])"),
                  code("print('FULL V1 TRAINING: fresh base or resume this local run; Colab final is preserved', flush=True)\nrun_cli([RUNNER, 'train', EXP, '--run-id', 'local_v1'])")]
    save(prefix+'.ipynb', cells)

save('04_context_data.ipynb', [md('# Context / gate data preparation\nCurrent 30-clip panel is DEV only. Gate training needs independent TRAIN audio with validated timestamps. This notebook audits local prerequisites; it does not pretend a gate training dataset already exists.'), code(common),
    code("import pandas as pd\nfrom urllib.parse import urlparse, parse_qs\nmeta = RAW_DATA/'ViMedCSS/ViMedCSS-Metadata'\ntrain = pd.read_csv(meta/'train_set.csv')\ndev = pd.read_csv(meta/'valid_set.csv')\ndef vid(url):\n    p = urlparse(str(url))\n    return p.path.strip('/') if p.hostname in ['youtu.be','www.youtu.be'] else parse_qs(p.query).get('v',[''])[0]\ndef seconds(value):\n    total = 0.0\n    for part in str(value).split(':'): total = total*60+float(part)\n    return total\n# Conservative separation from ALL official DEV source videos.\ndev_ids = set(dev.original_video_link.map(vid))\ntrain['video_id'] = train.original_video_link.map(vid)\ntrain['start_s'] = train.start_time.map(seconds)\ntrain['end_s'] = train.end_time.map(seconds)\ntrain_panel = train[train.video_id.ne('') & ~train.video_id.isin(dev_ids) & train.start_s.ge(20) & (train.end_s-train.start_s).gt(0) & (train.end_s-train.start_s).le(30)].copy()\ntrain_panel['alignment_status'] = 'pending'\nout = PROJECT/'local/context/train_candidates.csv'\nout.parent.mkdir(parents=True, exist_ok=True)\ntrain_panel.sort_values(['video_id','start_s']).to_csv(out,index=False)\nprint('TRAIN candidates:', len(train_panel), '| Videos:', train_panel.video_id.nunique())\nprint('Saved:',out)\ndisplay(train_panel.head())"),
    md('## Next gate stage\nDownload/validate TRAIN original audio, generate reference-blind KEEP/context candidates, then create labels from TRAIN transcripts. Split gate TRAIN/CAL by source video. Never use reference WER, gold transcript, or oracle choice as input features. Export features once; small gate training can use CPU. First verify inference on local V1 in notebook 03. History 5s anomaly and EOS/truncation audit remain unresolved; do not freeze a history policy yet.'),
])

save('05_compare_runs.ipynb', [md('# Compare local experiments\nCompare matching split and subset sizes. Full DEV and smoke subsets must not be interpreted as the same benchmark.'),code(common),
    code("import pandas as pd\nrows = []\nfor path in (PROJECT/'local/runs').glob('*/*/eval/*/metrics.json'):\n    m = json.loads(path.read_text(encoding='utf-8'))\n    rel = path.relative_to(PROJECT/'local/runs')\n    if rel.parts[1].startswith('validation'): continue\n    rows.append({'experiment':rel.parts[0], 'run':rel.parts[1], 'split_or_subset':rel.parts[3], 'n':m['overall']['n_utts'], 'WER (%)':100*m['overall']['wer'], 'CER (%)':100*m['overall']['cer'], 'checkpoint':m['checkpoint']})\ncomparison = pd.DataFrame(rows)\ndisplay(comparison)\nif rows:\n    comparison.to_csv(PROJECT/'local/runs/comparison.csv',index=False)"),
    code("resources = []\nfor path in (PROJECT/'local/runs').glob('*/*/resources.jsonl'):\n    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]\n    if entries:\n        resources.append({'run':str(path.parent.relative_to(PROJECT)), 'max_total_gpu_GB':max(e['gpu_total_used_gb'] for e in entries), 'max_host_RSS_GB':max(e['host_rss_gb'] for e in entries)})\ndisplay(pd.DataFrame(resources))")])
print('Generated:', *sorted(p.name for p in OUT.glob('*.ipynb')), sep='\n')
