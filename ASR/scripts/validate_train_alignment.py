"""Read-only audio alignment audit; writes separate reports, never changes source manifests."""
import argparse
import csv
import datetime
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import correlate, resample_poly

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
MATCH_SR = 4000


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ncc(source, clip):
    """Zero-mean, locally energy-normalized waveform correlation, valid lags only."""
    source = np.asarray(source, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    if len(source) < len(clip) or len(clip) < 2:
        return np.empty(0)
    clip = clip - clip.mean()
    energy = np.dot(clip, clip)
    if energy <= 1e-12:
        return np.zeros(len(source) - len(clip) + 1)
    sums = np.concatenate(([0.0], np.cumsum(source)))
    squares = np.concatenate(([0.0], np.cumsum(source * source)))
    n = len(clip)
    window_sum = sums[n:] - sums[:-n]
    variance = np.maximum(squares[n:] - squares[:-n] - window_sum**2 / n, 0)
    numerator = correlate(source, clip, mode="valid", method="fft")
    # FFT/cumulative-sum roundoff in silent source windows must not become a
    # spurious perfect match after division by a near-zero denominator.
    valid = variance > max(n * 1e-10, float(np.var(source)) * n * 1e-8)
    scores = np.zeros_like(numerator)
    np.divide(numerator, np.sqrt(np.maximum(variance * energy, 1e-24)),
              out=scores, where=valid)
    return np.clip(scores, -1, 1)


def best_match(source, clip, sr=MATCH_SR):
    scores = np.abs(ncc(source, clip))
    if not len(scores):
        return None
    index = int(np.argmax(scores))
    # Ignore the same local correlation peak when evaluating alternative matches.
    mask = np.ones(len(scores), dtype=bool)
    radius = int(0.25 * sr)
    mask[max(0, index-radius):index+radius+1] = False
    alternative = float(scores[mask].max()) if mask.any() else 0.0
    return index, float(scores[index]), alternative


def load_range(path, start, end):
    with sf.SoundFile(str(path)) as f:
        if f.samplerate != SR or f.channels != 1:
            raise ValueError("Source must be mono 16 kHz")
        start_frame = max(0, round(start * SR))
        f.seek(min(start_frame, len(f)))
        return f.read(max(0, round(end * SR) - start_frame), dtype="float32")


def find_alignment(path, clip, nominal_start, nominal_end, global_cache=None):
    info = sf.info(str(path))
    duration = len(clip) / SR
    small = resample_poly(clip, 1, 4)
    lo = max(0.0, nominal_start - 3.0)
    hi = min(info.duration, nominal_start + duration + 3.0)
    search = resample_poly(load_range(path, lo, hi), 1, 4)
    match = best_match(search, small)
    scope = "local_3s"
    if match is None or match[1] < 0.85:
        # Full-video search also catches intro edits or grossly wrong metadata.
        key = str(path)
        if global_cache is not None and global_cache.get("key") == key:
            search = global_cache["audio"]
        else:
            search = resample_poly(load_range(path, 0, info.duration), 1, 4)
            if global_cache is not None:
                global_cache.clear()
                global_cache.update(key=key, audio=search)
        match = best_match(search, small)
        lo = 0.0
        scope = "full_video"
    if match is None:
        return dict(status="exclude", reason="source_shorter_than_clip")
    offset, score, alternative = match
    matched_start = lo + offset / MATCH_SR
    # Independently align first/middle/last thirds; partial matches cannot pass.
    piece_scores, piece_shifts = [], []
    for a, b in zip(np.linspace(0, len(clip), 4, dtype=int)[:-1],
                    np.linspace(0, len(clip), 4, dtype=int)[1:]):
        piece = clip[a:b]
        expected = matched_start + a / SR
        piece_lo = max(0, expected - 0.10)
        segment = load_range(path, piece_lo, min(info.duration, expected + len(piece)/SR + 0.10))
        found = best_match(resample_poly(segment, 1, 2), resample_poly(piece, 1, 2), 8000)
        if found is None:
            piece_scores.append(0.0)
            piece_shifts.append(999.0)
        else:
            piece_scores.append(found[1])
            piece_shifts.append(piece_lo + found[0]/8000 - (nominal_start+a/SR))
    shift = matched_start - nominal_start
    spread = max(piece_shifts) - min(piece_shifts)
    margin = score - alternative
    duration_delta = duration - (nominal_end - nominal_start)
    confident = score >= .80 and min(piece_scores) >= .65 and spread <= .04 and margin >= .15
    history_ok = matched_start >= 20.0
    if confident:
        if abs(shift) <= .08 and abs(duration_delta) <= .05:
            status, reason = "pass", "strong_consistent_waveform_match"
        else:
            status, reason = "adjust", "strong_match_timestamp_correction"
    else:
        status, reason = "review", "uncertain_or_inconsistent_waveform_match"
    return dict(
        status=status, reason=reason, search_scope=scope,
        clip_duration_s=round(duration, 6), source_duration_s=round(info.duration, 6),
        matched_start_s=round(matched_start, 6),
        matched_end_s=round(matched_start+duration, 6),
        offset_s=round(shift, 6), duration_delta_s=round(duration_delta, 6),
        correlation=round(score, 6), alternative_correlation=round(alternative, 6),
        peak_margin=round(margin, 6),
        first_third_correlation=round(piece_scores[0], 6),
        middle_third_correlation=round(piece_scores[1], 6),
        last_third_correlation=round(piece_scores[2], 6),
        third_offset_spread_s=round(spread, 6),
        first_third_offset_s=round(piece_shifts[0], 6),
        middle_third_offset_s=round(piece_shifts[1], 6),
        last_third_offset_s=round(piece_shifts[2], 6),
        history_20s_ready=history_ok,
        listening_status="not_listened",
    )


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--recheck-review", action="store_true",
                    help="Recheck uncertain rows only, retaining completed measurements")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = {r["segment_id"]: r for r in read_csv(ROOT/"local/context/train_candidates.csv")}
    available = read_csv(ROOT/"local/context/train_audio_available.csv")
    requested = {r["segment_id"]: r for r in available}
    retained = []
    if args.recheck_review:
        with (args.output/"progress.jsonl").open(encoding="utf-8") as f:
            previous = [json.loads(line) for line in f if line.strip()]
        retained = [r for r in previous if r["status"] != "review"]
        uncertain = {r["segment_id"] for r in previous if r["status"] == "review"}
        requested = {sid: r for sid, r in requested.items() if sid in uncertain}
    if args.limit:
        # Spread pilot samples across the manifest, instead of only the first video.
        ids = list(requested)
        selected = [ids[i] for i in np.linspace(0, len(ids)-1, min(args.limit, len(ids)), dtype=int)]
        requested = {k: requested[k] for k in selected}
    print(f"Requested {len(requested)} clips; CPU waveform matching only", flush=True)
    parquet = pq.ParquetFile(ROOT/"data/ViMedCSS/train.parquet")
    results, seen, cache = retained.copy(), set(), {}
    started = time.monotonic()
    journal_name = "recheck_progress.jsonl" if args.recheck_review else "progress.jsonl"
    with (args.output/journal_name).open("w", encoding="utf-8") as journal:
        for group in range(parquet.num_row_groups):
            ids = parquet.read_row_group(group, columns=["id"])["id"].to_pylist()
            if not any(x.removeprefix("vimedcss:") in requested for x in ids):
                continue
            table = parquet.read_row_group(group, columns=["id", "audio"])
            for entry in table.to_pylist():
                sid = entry["id"].removeprefix("vimedcss:")
                if sid not in requested:
                    continue
                row = requested[sid]
                metadata = candidates[sid]
                result = dict(segment_id=sid, video_id=row["video_id"],
                              nominal_start_s=float(row["start_s"]),
                              nominal_end_s=float(row["end_s"]),
                              transcript=metadata["segment_text"],
                              audio_path=row["audio_path"])
                try:
                    clip, sr = sf.read(io.BytesIO(entry["audio"]), dtype="float32")
                    if sr != SR or clip.ndim != 1 or not len(clip):
                        raise ValueError("Dataset clip must be nonempty mono 16 kHz")
                    if np.sqrt(np.mean(clip**2)) < 1e-5:
                        result.update(status="review", reason="silent_or_near_silent_dataset_clip")
                    else:
                        result.update(find_alignment(Path(row["audio_path"]), clip,
                                      float(row["start_s"]), float(row["end_s"]), cache))
                except Exception as exc:
                    result.update(status="review", reason="read_or_analysis_error", error=str(exc))
                results.append(result)
                seen.add(sid)
                journal.write(json.dumps(result, ensure_ascii=False)+"\n")
                journal.flush()
                if len(results) % 25 == 0 or args.limit:
                    counts = {s: sum(r["status"] == s for r in results)
                              for s in ("pass", "adjust", "review", "exclude")}
                    print(f"{len(results)}/{len(requested)+len(retained)} | {counts} | elapsed={time.monotonic()-started:.0f}s", flush=True)
        for sid in sorted(set(requested)-seen):
            results.append(dict(segment_id=sid, video_id=requested[sid]["video_id"],
                                status="exclude", reason="dataset_clip_not_found"))
    results.sort(key=lambda r: (r["video_id"], r["segment_id"]))
    fields = list(dict.fromkeys(k for row in results for k in row))
    write_csv(args.output/"alignment_all.csv", results, fields)
    for status in ("pass", "adjust", "review", "exclude"):
        write_csv(args.output/f"alignment_{status}.csv", [r for r in results if r["status"] == status], fields)
    summary = dict(created_at=datetime.datetime.now().isoformat(),
                   requested=len(requested)+len(retained), analyzed=len(results),
                   counts={s: sum(r["status"] == s for r in results) for s in ("pass", "adjust", "review", "exclude")},
                   elapsed_seconds=round(time.monotonic()-started, 2),
                   listening_performed=False,
                   thresholds=dict(full_ncc=.80, minimum_third_ncc=.65,
                                   maximum_third_shift_spread_s=.04, minimum_peak_margin=.15,
                                   pass_max_offset_s=.08, pass_max_duration_delta_s=.05),
                   scope="Prepared TRAIN clip waveform versus downloaded source; no transcript recognition or model inference")
    summary["silent_source_windows_excluded_from_ncc"] = True
    summary["rechecked_uncertain_rows"] = len(requested) if args.recheck_review else 0
    if args.recheck_review and (args.output/"summary.json").exists():
        previous_summary = json.loads((args.output/"summary.json").read_text(encoding="utf-8"))
        summary["initial_scan_elapsed_seconds"] = previous_summary["elapsed_seconds"]
        summary["recheck_elapsed_seconds"] = summary["elapsed_seconds"]
        summary["elapsed_seconds"] += previous_summary["elapsed_seconds"]
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
