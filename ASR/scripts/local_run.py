"""Local resource-controlled entry point. Original Colab scripts/configs stay unchanged."""
import argparse
import atexit
import datetime
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / 'local/cache/datasets'))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('mode', choices=['check', 'eval', 'train'])
    ap.add_argument('exp_id', nargs='?', default='B0_phowhisper_small_zeroshot')
    ap.add_argument('--run-id', default='local_v1')
    ap.add_argument('--data-dir', default=str(ROOT / 'data'))
    ap.add_argument('--checkpoint')
    ap.add_argument('--split', default='dev', choices=['dev', 'test', 'hard'])
    ap.add_argument('--allow-test', action='store_true')
    ap.add_argument('--max-per-source', type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--duty-cycle', type=float, help='override profile gpu_duty_cycle for this run (kept in manifest)')
    args = ap.parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        ap.error('run-id must be one directory name')
    profile = json.loads((ROOT / 'local/profile.json').read_text())
    if args.duty_cycle is not None:
        profile['gpu_duty_cycle'] = args.duty_cycle
    print(f'[INIT] mode={args.mode} experiment={args.exp_id}; importing PyTorch/ASR libraries...', flush=True)
    import psutil
    import torch
    import transformers
    import asr.config as config
    import asr.data as data

    original_load = config.load_experiment
    cfg, _ = original_load(args.exp_id, args.data_dir)
    run = ROOT / 'local/runs' / args.exp_id / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    required = ['train', 'dev'] if args.mode == 'train' else [args.split]
    sources = list(cfg['data']['train_sources']) if args.mode == 'train' else cfg['data']['eval_sources']
    if args.mode == 'eval' and args.split == 'hard':  # only ViMedCSS sources define a HARD split
        sources = [s for s in sources if (Path(args.data_dir) / s / 'hard.parquet').is_file()]
    missing =[str(Path(args.data_dir) / source / f'{split}.parquet')
               for source in sources for split in required
               if not (Path(args.data_dir) / source / f'{split}.parquet').is_file()]
    import importlib.metadata
    status = dict(python=sys.executable, cuda=torch.cuda.is_available(),
                  versions={p: importlib.metadata.version(p) for p in
                            ['torch', 'transformers', 'peft', 'accelerate', 'datasets']},
                  missing_data=missing, profile=profile,
                  local_v1_final=(ROOT / 'experiments/V1_phowhisper_small_lora/final/adapter_config.json').exists())
    status['system_ram_total_gb'] = psutil.virtual_memory().total/1e9
    status['system_ram_available_gb'] = psutil.virtual_memory().available/1e9
    if torch.cuda.is_available():
        gpu_free, gpu_total = torch.cuda.mem_get_info()
        status['gpu_total_gb'] = gpu_total/1e9
        status['gpu_current_used_gb'] = (gpu_total-gpu_free)/1e9
    print(json.dumps(status, indent=2), flush=True)
    if args.mode == 'check':
        if torch.cuda.is_available():
            print('GPU:', torch.cuda.get_device_name(0))
        return
    if missing:
        raise SystemExit('Missing prepared data; run 00_setup first. No partial-source benchmark allowed.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; not starting a CPU training job.')
    if args.mode == 'train' and not cfg['train']:
        raise SystemExit('B0/B1 are evaluation-only; use V1 for training.')
    if args.mode == 'eval' and cfg['train'] and not args.checkpoint:
        args.checkpoint = str(ROOT / 'experiments' / args.exp_id / 'final')
    if args.checkpoint and not Path(args.checkpoint).is_dir():
        raise SystemExit(f'Missing checkpoint: {args.checkpoint}')

    lock = ROOT / 'local/gpu.lock'
    if lock.exists():
        owner = int(lock.read_text())
        if psutil.pid_exists(owner):
            raise SystemExit(f'Another local GPU run is active (PID {owner}).')
        lock.unlink()  # stale lock from an exited child; no user data is removed
    try:
        with lock.open('x') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit('Another GPU run started concurrently.')

    def release_lock():
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()

    atexit.register(release_lock)

    # Decimal GB. Budget includes estimated existing GPU use plus a 250 MB reserve.
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocator_budget = profile['total_gpu_budget_gb'] * 1e9 - used - 250_000_000
    if allocator_budget < 1_000_000_000:
        raise SystemExit('Too little GPU headroom; close other GPU jobs before retrying.')
    torch.cuda.set_per_process_memory_fraction(min(1., allocator_budget / total))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    telemetry = run / 'resources.jsonl'

    def sample(stage):
        free_now, total_now = torch.cuda.mem_get_info()
        row = dict(time=datetime.datetime.now().isoformat(), stage=stage,
                   gpu_total_used_gb=(total_now-free_now)/1e9,
                   allocator_reserved_gb=torch.cuda.memory_reserved()/1e9,
                   allocator_peak_gb=torch.cuda.max_memory_reserved()/1e9,
                   host_rss_gb=process.memory_info().rss/1e9)
        with telemetry.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        host_limit = profile.get('host_ram_budget_gb', 0)
        if host_limit and row['host_rss_gb'] > host_limit:
            raise MemoryError('Host RAM budget exceeded; periodic checkpoint remains available.')
        if row['gpu_total_used_gb'] > profile['total_gpu_budget_gb']:
            raise MemoryError('Total GPU memory budget exceeded; close competing GPU applications.')
        return row

    # Duty cycle is deliberate idle time, NOT a GPU utilization or power limit.
    duty = profile['gpu_duty_cycle']
    if not 0 < duty <= 1:
        raise ValueError('gpu_duty_cycle must be in (0,1]')

    def paced_generate(original):
        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = original(*a, **kw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            sample('generate')
            time.sleep(elapsed*(1/duty-1))
            return result
        return wrapped

    import asr.model as model_module
    original_model = model_module.load_model

    def local_model(*a, **kw):
        print('[MODEL] Loading processor and weights; first use may download model files...', flush=True)
        model, processor = original_model(*a, **kw)
        print('[MODEL] Weights loaded.', flush=True)
        model.generate = paced_generate(model.generate)
        return model, processor

    model_module.load_model = local_model

    def local_config(exp_id, data_dir=None):
        conf, _ = original_load(exp_id, data_dir or args.data_dir)
        conf['eval']['batch_size'] = 1
        if conf['train']:
            # train.local_eval_steps (optional): eval+save interval for slow models; default = profile save_steps
            steps = conf['train'].get('local_eval_steps', profile['save_steps'])
            # train.local_batch_size (optional): keeps the effective batch (profile batch x accum) with fewer, larger micro-batches
            batch = conf['train'].get('local_batch_size', profile['batch_size'])
            accum = profile['grad_accum'] * profile['batch_size'] // batch
            conf['train'].update(batch_size=batch, grad_accum=accum,
                                 num_workers=0, fp16=True, gradient_checkpointing=True,
                                 save_steps=steps, eval_steps=steps,
                                 save_total_limit=profile['save_total_limit'], logging_steps=5)
        return conf, run

    config.load_experiment = local_config
    resolved, _ = local_config(args.exp_id)
    signature = dict(config=resolved, profile=profile, smoke=args.smoke,
                     source_script=args.mode, checkpoint=args.checkpoint,
                     split=args.split if args.mode == 'eval' else None,
                     max_per_source=args.max_per_source)
    # Refuse silently resuming a run after its configuration has changed.
    manifest = run / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != signature:
        raise SystemExit('Run configuration changed. Choose a new --run-id.')
    manifest.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')

    class ResourceCallback(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.last_saved = time.perf_counter()
            self.train_started = time.perf_counter()
            self.initial_step = state.global_step
            print(f'[TRAIN] Start/resume step {state.global_step}/{state.max_steps} | '
                  f'batch={args.per_device_train_batch_size} | accumulation={args.gradient_accumulation_steps}', flush=True)

        def on_save(self, args, state, control, **kw):
            self.last_saved = time.perf_counter()
            print(f'[CHECKPOINT] Saved: {args.output_dir}/checkpoint-{state.global_step}', flush=True)

        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.microbatch = 0

        def on_substep_end(self, args, state, control, **kw):
            self.microbatch += 1
            if self.microbatch % 4 == 0:
                print(f'[TRAIN] step {state.global_step+1}/{state.max_steps} | '
                      f'microbatch {self.microbatch}/{args.gradient_accumulation_steps}', flush=True)

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-self.started
            usage = sample(f'train_step:{state.global_step}')
            time.sleep(elapsed*(1/duty-1))
            done = state.global_step-self.initial_step
            seconds = time.perf_counter()-self.train_started
            eta = seconds/done*(state.max_steps-state.global_step) if done else 0
            print(f'[TRAIN] {state.global_step}/{state.max_steps} '
                  f'({100*state.global_step/max(1,state.max_steps):.1f}%) | epoch={state.epoch:.3f} | '
                  f'ETA~{eta/60:.1f} min | RAM={usage["host_rss_gb"]:.2f} GB | '
                  f'VRAM(total)={usage["gpu_total_used_gb"]:.2f} GB', flush=True)
            if time.perf_counter()-self.last_saved >= profile['checkpoint_interval_seconds']:
                control.should_save = True

        def on_log(self, args, state, control, logs=None, **kw):
            if logs:
                print('[METRICS]', json.dumps(logs, ensure_ascii=False, default=str), flush=True)

    start = time.perf_counter()
    script_args = [args.exp_id, '--data-dir', args.data_dir]
    if args.mode == 'eval':
        script_args += ['--split', args.split]
        if args.checkpoint:
            script_args += ['--checkpoint', args.checkpoint]
        if args.max_per_source:
            script_args += ['--max-per-source', str(args.max_per_source)]
        if args.allow_test:
            script_args += ['--allow-test']
    sys.argv = [str(ROOT / 'scripts' / f'{args.mode if args.mode == "train" else "evaluate"}.py'), *script_args]
    try:
        if args.mode == 'train':
            from local_train import run_training
            run_training(resolved, run, ResourceCallback(), args.smoke)
        else:
            runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sample('finished_or_interrupted')
        print(f'Elapsed: {time.perf_counter()-start:.1f}s | Output: {run}', flush=True)


if __name__ == '__main__':
    main()
