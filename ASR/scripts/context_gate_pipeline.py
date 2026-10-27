"""Prepare -> generate frozen ASR pairs -> train/calibrate -> evaluate a context gate.

Track B diagnostic: native Whisper history prompting, not a streaming editor.
All gate partitions come from official TRAIN, separated by source video.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from asr.context_gate import FEATURES, features, label, choose, probability, words

DEFAULT_MANIFEST = ROOT/'local/context/alignment_audit_20261020/train_audio_waveform_verified_20s.csv'
DEFAULT_CHECKPOINT = ROOT/'local/runs/V1_phowhisper_small_lora/local_v1/final'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def immutable_jsonl(path, rows):
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Refusing changed inputs in existing run: {path}; use a new --run-dir')
    if not path.exists():
        path.write_text(text, encoding='utf-8')


def video_id(value):
    if '://' not in value:
        return value
    url = urlparse(value)
    return parse_qs(url.query).get('v', [url.path.rstrip('/').split('/')[-1]])[0]


def partition(video, seed=13):
    n = int(hashlib.sha256(f'{seed}:{video}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if n < 70 else 'cal' if n < 85 else 'dev'


def prepare(args):
    import pyarrow.parquet as pq
    import soundfile as sf
    data = ROOT/'data/ViMedCSS'
    excluded = set()
    for split in ('dev', 'test', 'hard'):
        path = data/f'{split}.parquet'
        if not path.exists():
            raise FileNotFoundError(f'Cannot audit video separation without {path}')
        excluded.update(video_id(g) for g in pq.read_table(path, columns=['group_id'])['group_id'].to_pylist())
    metadata = {r['id'].split(':', 1)[-1]: r for r in
                pq.read_table(data/'train.parquet', columns=['id', 'text', 'group_id']).to_pylist()}
    with args.manifest.open(encoding='utf-8-sig', newline='') as f:
        source = list(csv.DictReader(f))
    rows, skipped, seen, audio_info = [], {}, set(), {}
    for r in source:
        sid, vid = r['segment_id'], r['video_id']
        if sid in seen:
            raise ValueError(f'Duplicate segment: {sid}')
        seen.add(sid)
        if vid in excluded:
            skipped['official_eval_video_overlap'] = skipped.get('official_eval_video_overlap', 0)+1
            continue
        meta = metadata.get(sid)
        if meta is None or video_id(meta['group_id']) != vid:
            raise ValueError(f'TRAIN ID/video mismatch: {sid}')
        start, end = float(r['start_s']), float(r['end_s'])
        reason = None
        if r['alignment_status'] != 'waveform_verified':
            reason = 'unverified'
        elif not (0 < end-start <= 30 and start >= args.history_seconds):
            reason = 'duration_or_history'
        elif not words(meta['text']):
            reason = 'empty_reference'
        if reason:
            skipped[reason] = skipped.get(reason, 0)+1
            continue
        path = Path(r['audio_path']).resolve()
        if str(path) not in audio_info:
            info = sf.info(str(path))
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f'Expected mono 16 kHz: {path}')
            stat = path.stat()
            audio_info[str(path)] = dict(duration=info.duration, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if end > audio_info[str(path)]['duration']+.001:
            raise ValueError(f'Audio ends before segment: {sid}')
        rows.append(dict(id=sid, video_id=vid, partition=partition(vid), audio_path=str(path),
                         start=start, end=end, history_start=start-args.history_seconds,
                         reference=meta['text'], audio_identity=audio_info[str(path)]))
    rows.sort(key=lambda r: (r['video_id'], r['start'], r['id']))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    immutable_jsonl(args.run_dir/'segments.jsonl', rows)
    summary = dict(rows=len(rows), partitions={p: dict(clips=sum(r['partition']==p for r in rows),
                    videos=len({r['video_id'] for r in rows if r['partition']==p})) for p in ('train','cal','dev')},
                   skipped=skipped, manifest_sha256=digest(args.manifest), history_seconds=args.history_seconds,
                   user_reports_listening_complete=True,
                   protocol='shared-history native-prompting diagnostic; official TRAIN only',
                   limitation='V1 backbone saw official TRAIN; gate dev is not unseen by backbone or an official DEV score')
    save(args.run_dir/'prepared.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def generate(args):
    import atexit
    import psutil
    import torch
    import soundfile as sf
    import transformers
    from asr.config import load_experiment
    from asr.model import load_model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for candidate generation')
    rows = read_jsonl(args.run_dir/'segments.jsonl')
    if not rows:
        raise ValueError('Run prepare first')
    checkpoint = args.checkpoint.resolve()
    weights = sorted(checkpoint.glob('adapter_model.*'))
    if not weights:
        raise FileNotFoundError(f'Missing adapter weights: {checkpoint}')
    profile = json.loads((ROOT/'local/profile.json').read_text())
    identity = dict(schema=1, segments_sha256=digest(args.run_dir/'segments.jsonl'),
                    checkpoint=str(checkpoint), weights={p.name:digest(p) for p in weights},
                    adapter_config=digest(checkpoint/'adapter_config.json'),
                    script_sha256=digest(Path(__file__)), features_sha256=digest(ROOT/'asr/context_gate.py'),
                    torch=torch.__version__, transformers=transformers.__version__,
                    max_new_tokens=args.max_new_tokens, prompt_max_tokens=128, profile=profile)
    meta_path = args.run_dir/'generation.json'
    if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8')) != identity:
        raise ValueError('Generation identity changed; use a new run directory')
    save(meta_path, identity)
    output = args.run_dir/'pairs.jsonl'
    completed = read_jsonl(output)
    done = {r['id'] for r in completed}
    if len(done) != len(completed) or not done <= {r['id'] for r in rows}:
        raise ValueError('Duplicate/unknown completed pair IDs')
    # A limit is per partition, with deterministic hash sampling to avoid video-order bias.
    if args.limit:
        selected = []
        for split in ('train', 'cal', 'dev'):
            subset = [r for r in rows if r['partition']==split]
            subset.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
            selected.extend(subset[:args.limit])
        rows = selected
    pending = [r for r in rows if r['id'] not in done]
    if not pending:
        print('All requested pairs already generated.', flush=True)
        return
    lock = ROOT/'local/gpu.lock'
    if lock.exists():
        owner = int(lock.read_text())
        if psutil.pid_exists(owner):
            raise RuntimeError(f'GPU job active: PID {owner}')
        lock.unlink()
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    def unlock():
        if lock.exists() and lock.read_text()==str(os.getpid()):
            lock.unlink()
    atexit.register(unlock)
    try:
        free, total = torch.cuda.mem_get_info()
        budget = profile['total_gpu_budget_gb']*1e9
        allowance = budget-(total-free)-250e6
        if allowance < 1.2e9:
            raise RuntimeError('Insufficient free GPU budget; close other GPU jobs')
        torch.cuda.set_per_process_memory_fraction(min(.95, allowance/total))
        torch.manual_seed(13)
        cfg, _ = load_experiment('V1_phowhisper_small_lora')
        print('[MODEL] Loading frozen V1...', flush=True)
        model, processor = load_model(cfg, checkpoint=checkpoint)
        model.to('cuda', torch.float16).eval()
        model.generation_config.return_dict_in_generate = True
        def resource_check(elapsed):
            free_now, total_now = torch.cuda.mem_get_info()
            if total_now-free_now > budget:
                raise RuntimeError('Sampled total GPU budget exceeded')
            ram = psutil.Process().memory_info().rss/1e9
            if profile.get('host_ram_budget_gb', 0)>0 and ram>profile['host_ram_budget_gb']:
                raise RuntimeError('Process RAM budget exceeded')
            duty = profile['gpu_duty_cycle']
            if not 0 < duty <= 1:
                raise ValueError('gpu_duty_cycle must be in (0,1]')
            time.sleep(elapsed*(1/duty-1))

        def tensor(audio):
            batch = processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            return batch.input_features.to('cuda', torch.float16), batch.attention_mask.to('cuda')

        def decode(audio, prompt=None):
            x, mask = tensor(audio)
            kw = {}
            if prompt and words(prompt):
                prompt_ids = processor.get_prompt_ids(prompt, return_tensors='pt')
                # Preserve start-of-previous token while bounding history token length.
                if len(prompt_ids)>129:
                    prompt_ids = torch.cat([prompt_ids[:1], prompt_ids[-128:]])
                kw['prompt_ids'] = prompt_ids.to('cuda')
            tick = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(x, attention_mask=mask, language='vi', task='transcribe', num_beams=1,
                                        max_new_tokens=args.max_new_tokens, return_dict_in_generate=True,
                                        output_scores=True, **kw)
                scores = model.compute_transition_scores(result.sequences.to(result.scores[0].device),
                                                          result.scores, normalize_logits=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            seq = result.sequences[0]
            truncated = len(result.scores)>=args.max_new_tokens and int(seq[-1])!=processor.tokenizer.eos_token_id
            # Short-form return_dict includes the decoder prompt. Decode ONLY generated
            # tokens, or history would be copied into the current hypothesis and labels.
            text = processor.batch_decode(result.sequences[:, -len(result.scores):], skip_special_tokens=True)[0].strip()
            confidence = float(scores.mean()) if scores.numel() else -30.
            del result, x, mask, scores
            resource_check(elapsed)
            return text, confidence, truncated

        def acoustic_score(audio, hypothesis):
            # Same audio, language/task prefix and no history prompt for both candidates.
            ids = processor.tokenizer(hypothesis, return_tensors='pt').input_ids.to('cuda')
            if ids.shape[1]>model.config.max_target_positions:
                raise ValueError('Candidate exceeds teacher-forcing token budget')
            if int(ids[0,0])==model.config.decoder_start_token_id:
                ids = ids[:,1:]
            x, mask = tensor(audio)
            tick = time.perf_counter()
            with torch.inference_mode():
                result = model(input_features=x, attention_mask=mask, labels=ids)
                score = -float(result.loss)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            del result, x, mask, ids
            resource_check(elapsed)
            return score

        for index, row in enumerate(pending, 1):
            start_time = time.perf_counter()
            path = Path(row['audio_path'])
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns)!=(row['audio_identity']['size'], row['audio_identity']['mtime_ns']):
                raise ValueError(f'Audio changed: {path}')
            with sf.SoundFile(path) as f:
                def read(a, b):
                    f.seek(round(a*16000))
                    return f.read(round(b*16000)-round(a*16000), dtype='float32')
                # Strictly preceding history: never includes current/future samples.
                history_audio = read(row['history_start'], row['start'])
                audio = read(row['start'], row['end'])
            history, confidence, ht = decode(history_audio)
            base, _, bt = decode(audio)
            context, _, ct = decode(audio, history)
            ba = acoustic_score(audio, base)
            ca = ba if base==context else acoustic_score(audio, context)
            x = features(base, context, history, ba, ca, confidence, row['end']-row['start'], bt, ct, ht)
            # Only now, after candidates and features are frozen, access the reference.
            record = dict(id=row['id'], video_id=row['video_id'], partition=row['partition'],
                          base=base, context=context, history=history, features=x,
                          base_truncated=bt, context_truncated=ct, history_truncated=ht,
                          reference=row['reference'], **label(row['reference'], base, context),
                          elapsed_seconds=time.perf_counter()-start_time)
            with output.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
            print(f'[PAIRS] {index}/{len(pending)} {row["partition"]} {row["id"]} '
                  f'errors={record["base_errors"]}->{record["context_errors"]} '
                  f'{record["elapsed_seconds"]:.1f}s', flush=True)
    finally:
        unlock()


def fit_head(rows, target):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x, y = [r['features'] for r in rows], [r[target] for r in rows]
    if len(set(y))<2:
        return {'constant': float(y[0]), 'warning': f'Only one class for {target}'}
    scale = StandardScaler().fit(x)
    model = LogisticRegression(C=1, max_iter=2000, random_state=13).fit(scale.transform(x), y)
    return dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))


def score(rows, model):
    decisions = [choose(model, r)[0]=='REPLACE' for r in rows]
    count = len(rows)
    denom = sum(r['ref_words'] for r in rows)
    selected_errors = sum(r['context_errors'] if d else r['base_errors'] for r,d in zip(rows, decisions))
    return dict(n=count, baseline_wer=sum(r['base_errors'] for r in rows)/denom,
                always_context_wer=sum(r['context_errors'] for r in rows)/denom,
                oracle_pair_wer=sum(min(r['base_errors'],r['context_errors']) for r in rows)/denom,
                gate_wer=selected_errors/denom, coverage=sum(decisions)/count,
                helped=sum(d and r['benefit'] for r,d in zip(rows,decisions)),
                harmed=sum(d and r['harm'] for r,d in zip(rows,decisions)),
                harm_rate=sum(d and r['harm'] for r,d in zip(rows,decisions))/count,
                selected_errors=selected_errors,
                benefit_brier=sum((probability(model['benefit'],r['features'])-r['benefit'])**2 for r in rows)/count,
                harm_brier=sum((probability(model['harm'],r['features'])-r['harm'])**2 for r in rows)/count)


def validate_pairs(run, allow_partial=False):
    rows = read_jsonl(run/'pairs.jsonl')
    source = {r['id']:r for r in read_jsonl(run/'segments.jsonl')}
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate pairs')
    if not allow_partial and {r['id'] for r in rows}!=set(source):
        raise ValueError('Candidate generation incomplete; resume generate or explicitly use --allow-partial for pilot')
    videos = {p:set() for p in ('train','cal','dev')}
    for row in rows:
        original = source[row['id']]
        if row['partition']!=original['partition'] or row['video_id']!=original['video_id']:
            raise ValueError('Pair identity/partition mismatch')
        if len(row['features'])!=len(FEATURES):
            raise ValueError('Feature schema mismatch')
        videos[row['partition']].add(row['video_id'])
    if any(videos[a]&videos[b] for a,b in [('train','cal'),('train','dev'),('cal','dev')]):
        raise ValueError('Cross-partition video leakage')
    if any(not v for v in videos.values()):
        raise ValueError('Need generated examples in train, cal and dev')
    return rows


def train(args):
    rows = validate_pairs(args.run_dir, args.allow_partial)
    output = args.run_dir/'gate.json'
    if output.exists():
        raise ValueError('gate.json already frozen; use a new run directory for retraining')
    train_rows = [r for r in rows if r['partition']=='train']
    cal = [r for r in rows if r['partition']=='cal']
    if not any(r['benefit'] for r in train_rows):
        raise ValueError('No beneficial TRAIN candidates. Candidate/context utility bottleneck; do not train a fake gate.')
    model = dict(schema=1, features=FEATURES, benefit=fit_head(train_rows,'benefit'),
                 harm=fit_head(train_rows,'harm'), benefit_threshold=1.01, harm_threshold=0,
                 pairs_sha256=digest(args.run_dir/'pairs.jsonl'),
                 generation_sha256=digest(args.run_dir/'generation.json'),
                 allow_partial=args.allow_partial, calibration_harm_budget=args.max_harm_rate,
                 protocol='Track B shared-history prompt gate; internal TRAIN video holdout')
    curve = []
    best_key = (sum(r['base_errors'] for r in cal), 0, 0)
    best = (1.01, 0)
    for benefit_threshold in [i/20 for i in range(21)]+[1.01]:
        for harm_threshold in [0, .01, .025, .05, .1, .2, .5, 1]:
            model.update(benefit_threshold=benefit_threshold, harm_threshold=harm_threshold)
            result = score(cal, model)
            curve.append(dict(benefit_threshold=benefit_threshold, harm_threshold=harm_threshold, **result))
            key = (result['selected_errors'], result['harmed'], result['coverage'])
            if result['harm_rate']<=args.max_harm_rate and key<best_key:
                best_key, best = key, (benefit_threshold, harm_threshold)
    model.update(benefit_threshold=best[0], harm_threshold=best[1])
    model['calibration'] = score(cal, model)
    model['counts'] = {p:sum(r['partition']==p for r in rows) for p in ('train','cal','dev')}
    model['operating_point'] = 'KEEP_ONLY' if best[0]>1 else 'CAL_SELECTED'
    save(args.run_dir/'calibration_curve.json', curve)
    save(output, model)
    print(json.dumps(model, indent=2), flush=True)


def evaluate(args):
    model = json.loads((args.run_dir/'gate.json').read_text(encoding='utf-8'))
    if digest(args.run_dir/'pairs.jsonl')!=model['pairs_sha256']:
        raise ValueError('Pairs changed after training; frozen evaluation refused')
    rows = [r for r in validate_pairs(args.run_dir, model['allow_partial']) if r['partition']=='dev']
    report = dict(protocol=model['protocol'], operating_point=model['operating_point'],
                  dev=score(rows,model), calibration=model['calibration'],
                  limitations=['Internal split of official TRAIN; backbone may have trained on all these clips.',
                               'History decoded independently from preceding audio; not closed-loop streaming.',
                               'Harm budget is empirical on CAL, not a statistical guarantee.',
                               'Prompting changes candidate generation (Track B); not a fixed-pool Track A ablation.'])
    immutable_jsonl(args.run_dir/'dev_decisions.jsonl',
                    [dict(id=r['id'], decision=choose(model,r)[0], benefit_probability=choose(model,r)[1],
                          harm_probability=choose(model,r)[2], hypothesis=r['context'] if choose(model,r)[0]=='REPLACE' else r['base']) for r in rows])
    save(args.run_dir/'evaluation.json', report)
    print(json.dumps(report, indent=2), flush=True)


def run(args):
    """Sequential full workflow. Stop on failure; never train on incomplete pairs."""
    prepare(args)
    generate(args)
    if not (args.run_dir/'gate.json').exists():
        train(args)
    evaluate(args)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('mode', choices=['prepare','generate','train','evaluate','run'])
    ap.add_argument('--run-dir', type=Path, default=ROOT/'local/context/gate_v1')
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument('--history-seconds', type=float, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=225)
    ap.add_argument('--limit', type=int, help='pilot: deterministic max clips PER partition; resume with no limit for full run')
    ap.add_argument('--allow-partial', action='store_true', help='explicitly train a pilot on incomplete generation')
    ap.add_argument('--max-harm-rate', type=float, default=.01, help='CAL harmed utterances / all CAL utterances')
    ap.add_argument('--offline', action='store_true', help='use already-cached V1 base model without network requests')
    args = ap.parse_args()
    if not 0 < args.history_seconds <= 20 or not 0<=args.max_harm_rate<=1:
        ap.error('history must be in (0,20], harm rate in [0,1]')
    if args.limit is not None and args.limit<1:
        ap.error('--limit must be positive')
    if not 1<=args.max_new_tokens<=225:
        ap.error('--max-new-tokens must be in [1,225]')
    args.run_dir = args.run_dir.resolve()
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    globals()[args.mode](args)


if __name__=='__main__':
    main()
