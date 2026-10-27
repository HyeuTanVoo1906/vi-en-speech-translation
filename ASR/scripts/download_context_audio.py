"""User-run TRAIN audio downloader. No model/GPU imports; no download on import."""
import argparse
import atexit
import csv
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'local'))


def video_id(url):
    p = urlparse(str(url))
    return p.path.strip('/') if p.hostname in ('youtu.be', 'www.youtu.be') else parse_qs(p.query).get('v', [''])[0]


def folder_bytes(folder):
    total = 0
    for path in folder.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def inspect_audio(path):
    import soundfile as sf
    try:
        info = sf.info(str(path))
        if info.samplerate == 16000 and info.channels == 1 and info.frames > 0:
            return info.duration
    except (RuntimeError, OSError):
        pass
    return None


def native_audio_files(work):
    extensions = {'.webm', '.m4a', '.mp4', '.mp3', '.opus', '.ogg', '.aac', '.wav', '.flac'}
    return [p for p in work.glob('source.*') if p.is_file() and p.suffix.lower() in extensions]


def tools():
    deno = ROOT / 'local/tools/deno/deno.exe'
    deno = str(deno) if deno.is_file() else shutil.which('deno')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    try:
        version = importlib.metadata.version('yt-dlp')
    except importlib.metadata.PackageNotFoundError:
        version = None
    return deno, ffmpeg, version


def install_deno():
    # Official Windows x64 release; extract only deno.exe, never arbitrary zip paths.
    existing = shutil.which('deno')
    if existing:
        print('Using Deno from PATH:', existing, flush=True); return
    folder = ROOT / 'local/tools/deno'
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / 'deno.exe'
    if dest.is_file():
        print('Deno already present:', dest, flush=True); return
    url = 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip'
    archive = folder / 'deno.zip.part'
    print('Downloading official portable Deno:', url, flush=True)
    with urllib.request.urlopen(url, timeout=30) as response, archive.open('wb') as f:
        received = 0
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > 150_000_000:
                raise RuntimeError('Unexpectedly large Deno archive')
            f.write(chunk)
            print(f'Deno archive: {received/1e6:.1f} MB', flush=True)
    with zipfile.ZipFile(archive) as z:
        member = z.getinfo('deno.exe')
        if member.file_size > 250_000_000:
            raise RuntimeError('Unexpectedly large executable')
        with z.open(member) as source, dest.with_suffix('.tmp').open('wb') as target:
            shutil.copyfileobj(source, target)
    dest.with_suffix('.tmp').replace(dest)
    archive.unlink()
    print('Installed portable Deno:', dest, flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('action', choices=['check', 'download', 'install-deno'])
    ap.add_argument('--batch-videos', type=int, default=30)
    ap.add_argument('--all-videos', action='store_true', help='Try the entire pending manifest within disk limits')
    ap.add_argument('--pause-seconds', type=float, default=8, help='Pause between attempted videos')
    ap.add_argument('--audio-budget-gb', type=float, default=10)
    ap.add_argument('--min-free-gb', type=float, default=20)
    ap.add_argument('--max-video-minutes', type=float, default=60)
    ap.add_argument('--retry-failed', action='store_true')
    auth = ap.add_mutually_exclusive_group()
    auth.add_argument('--cookies-from-browser', help='e.g. chrome; read locally only when user runs download')
    auth.add_argument('--cookies', help='Optional local Netscape cookie file; never print its contents')
    args = ap.parse_args()
    if args.action == 'install-deno':
        install_deno(); return
    if args.batch_videos < 1 or args.audio_budget_gb <= .5 or args.min_free_gb < 0 or args.pause_seconds < 0:
        ap.error('Invalid batch or disk limits')
    cookie_options = []
    if args.cookies_from_browser:
        cookie_options = ['--cookies-from-browser', args.cookies_from_browser]
    elif args.cookies:
        cookie_file = Path(args.cookies).expanduser().resolve()
        if not cookie_file.is_file():
            ap.error('Local cookie file is missing; no download started')
        cookie_options = ['--cookies', str(cookie_file)]
    audio_root = ROOT / 'local/imports/original_audio/train'
    audio_root.mkdir(parents=True, exist_ok=True)
    context = ROOT / 'local/context'
    manifest = context / 'train_candidates.csv'
    if not manifest.is_file():
        raise SystemExit('Run notebook 04 first: train_candidates.csv is missing.')
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    dev_csv = ROOT.parent / 'dataset/ASR/ViMedCSS/ViMedCSS-Metadata/valid_set.csv'
    with dev_csv.open(encoding='utf-8-sig', newline='') as f:
        dev_videos = {video_id(r['original_video_link']) for r in csv.DictReader(f)}
    groups = {}
    for row in rows:
        vid = row['video_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', vid) or vid in dev_videos:
            raise ValueError(f'Invalid video ID or DEV overlap: {vid}')
        start, end = float(row['start_s']), float(row['end_s'])
        if start < 20 or not 0 < end-start <= 30:
            raise ValueError(f'Invalid timestamps: {row["segment_id"]}')
        groups.setdefault(vid, []).append(row)
    if args.all_videos:
        args.batch_videos = len(groups)
    deno, ffmpeg, ytdlp_version = tools()
    disk = shutil.disk_usage(audio_root)
    print(json.dumps(dict(
        candidate_clips=len(rows), candidate_videos=len(groups),
        audio_folder=str(audio_root), audio_folder_gb=folder_bytes(audio_root)/1e9,
        drive_free_gb=disk.free/1e9, audio_budget_gb=args.audio_budget_gb,
        min_free_gb=args.min_free_gb, yt_dlp=ytdlp_version, deno=deno, ffmpeg=ffmpeg,
        authentication=('browser session: '+args.cookies_from_browser if args.cookies_from_browser
                        else 'local cookie file' if args.cookies else 'none'),
        mode='all pending videos' if args.all_videos else 'batch',
        max_attempts=args.batch_videos, pause_seconds=args.pause_seconds,
        device='CPU only; no model or GPU used'), ensure_ascii=False, indent=2), flush=True)
    if args.action == 'check':
        return
    if not all([deno, ffmpeg, ytdlp_version]):
        raise SystemExit('Missing tools. Run the notebook dependency and portable-Deno cells first.')

    import psutil
    from notebook_process import stop_process_tree
    lock = audio_root / 'download.lock'
    if lock.exists():
        if psutil.pid_exists(int(lock.read_text())):
            raise SystemExit('Another audio download is running.')
        lock.unlink()
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    def unlock():
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
    atexit.register(unlock)
    journal = context / 'train_audio_downloads.jsonl'
    last_status = {}
    if journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            try:
                entry = json.loads(line)
                last_status[entry['video_id']] = entry['status']
            except (ValueError, KeyError):
                continue
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = context / 'download_logs' / stamp
    log_dir.mkdir(parents=True, exist_ok=True)

    def disk_reason(reserve=0):
        used = folder_bytes(audio_root)
        free = shutil.disk_usage(audio_root).free
        if used + reserve >= args.audio_budget_gb*1e9:
            return f'audio folder budget reached ({used/1e9:.2f} GB)'
        if free - reserve <= args.min_free_gb*1e9:
            return f'drive free-space floor reached ({free/1e9:.2f} GB free)'
        return None

    def monitored(command, log_path):
        print('[PROCESS]', subprocess.list2cmdline(list(map(str, command))), flush=True)
        started = time.monotonic()
        with log_path.open('a', encoding='utf-8') as log:
            child = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding='utf-8', errors='replace')
            def drain():
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            reason = None
            try:
                while child.poll() is None:
                    reason = disk_reason(100_000_000)
                    if time.monotonic()-started > 1200:
                        reason = 'per-video process timeout (20 minutes)'
                    if reason:
                        stop_process_tree(child.pid); break
                    time.sleep(1)
                code = child.wait()
            except BaseException:
                stop_process_tree(child.pid); child.wait(); raise
            finally:
                reader.join(timeout=5)
        return code, reason

    def export_available():
        ready = []
        for vid, clips in sorted(groups.items()):
            path = audio_root / f'{vid}.flac'
            duration = inspect_audio(path) if path.is_file() else None
            if duration is None:
                continue
            for row in clips:
                if float(row['end_s']) <= duration:
                    ready.append(dict(segment_id=row['segment_id'], video_id=vid,
                                      audio_path=str(path), history_start_s=float(row['start_s'])-20,
                                      start_s=row['start_s'], end_s=row['end_s'],
                                      alignment_status='pending'))
        save_csv(context/'train_audio_available.csv', ready,
                 ['segment_id', 'video_id', 'audio_path', 'history_start_s', 'start_s', 'end_s', 'alignment_status'])
        return ready

    attempts = succeeded = skipped = 0
    batch_rows = []
    for vid in sorted(groups):
        dest = audio_root / f'{vid}.flac'
        if dest.is_file() and inspect_audio(dest) is not None:
            skipped += 1; continue
        if last_status.get(vid) in ('failed', 'out_of_bounds') and not args.retry_failed:
            continue
        if attempts >= args.batch_videos:
            break
        reason = disk_reason(500_000_000)
        if reason:
            print('[STOP]', reason, flush=True); break
        if attempts and args.pause_seconds:
            print(f'[PAUSE] Waiting {args.pause_seconds:g}s before the next video...', flush=True)
            time.sleep(args.pause_seconds)
        attempts += 1
        work = audio_root / '.work' / vid
        work.mkdir(parents=True, exist_ok=True)
        log = log_dir / f'{vid}.log'
        print(f'\n[VIDEO {attempts}/{args.batch_videos}] {vid} | clips={len(groups[vid])}', flush=True)
        record = dict(time=datetime.datetime.now().isoformat(), video_id=vid,
                      status='failed', error='', log=str(log), audio_path=str(dest))
        stop_batch = False
        try:
            native = native_audio_files(work)
            if not native:
                code, reason = monitored([
                    sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist',
                    '--no-cache-dir', '--no-overwrites', '--newline', '--progress', '--progress-delta', '2',
                    '--js-runtimes', f'deno:{deno}', '--socket-timeout', '20', '--retries', '2',
                    '--fragment-retries', '2', '--concurrent-fragments', '1', '--limit-rate', '2M',
                    '--max-filesize', '512M', '--match-filters', f'!is_live & duration <= {args.max_video_minutes*60:g}',
                    '-f', 'bestaudio', '-o', str(work/'source.%(ext)s'),
                    *cookie_options,
                    f'https://www.youtube.com/watch?v={vid}',
                ], log)
                if reason:
                    stop_batch = 'budget' in reason or 'floor' in reason
                if code or reason:
                    detail = log.read_text(encoding='utf-8', errors='replace').lower()
                    if any(marker in detail for marker in (
                        'could not copy chrome cookie database', 'failed to decrypt',
                        'unable to decrypt', 'could not find chrome cookies database',
                        'could not copy edge cookie database',
                    )):
                        record['status'] = 'cookie_error'
                        stop_batch = True
                        raise RuntimeError('Browser cookie access/decryption failed. Batch stopped. '
                                           'Close Chrome normally and retry one video; if decryption '
                                           'still fails, use a locally exported Netscape cookie file. '
                                           'Do not disable browser/Windows protections or upload cookies.')
                    if any(marker in detail for marker in (
                        'http error 429', 'too many requests',
                        'sign in to confirm', 'not a bot',
                    )):
                        record['status'] = 'blocked'
                        stop_batch = True
                        raise RuntimeError('YouTube rate/authentication block. Batch stopped; '
                                           'do not retry all videos. Resolve access normally in '
                                           'your browser and test a single video later. See log.')
                    raise RuntimeError(reason or f'yt-dlp exit={code}; see log')
                native = native_audio_files(work)
            if len(native) != 1:
                raise RuntimeError('No single downloaded audio file; video may be unavailable or filtered.')
            temp = work/'converted.flac'
            code, reason = monitored([ffmpeg, '-hide_banner', '-nostdin', '-loglevel', 'error',
                                      '-threads', '2', '-y', '-i', native[0], '-vn', '-ar', '16000',
                                      '-ac', '1', '-c:a', 'flac', '-sample_fmt', 's16', temp], log)
            if reason:
                stop_batch = 'budget' in reason or 'floor' in reason
            if code or reason or inspect_audio(temp) is None:
                raise RuntimeError(reason or f'FLAC conversion/validation failed; ffmpeg exit={code}')
            duration = inspect_audio(temp)
            # Publish only completely converted and inspected audio, never a partial download.
            temp.replace(dest)
            record.update(status='ok', duration_s=duration, size_mb=dest.stat().st_size/1e6)
            if not any(float(r['end_s']) <= duration for r in groups[vid]):
                record['status'] = 'out_of_bounds'
            else:
                succeeded += 1
            # Delete only this job's temporary source files after a valid final FLAC exists.
            if not work.resolve().is_relative_to((audio_root/'.work').resolve()):
                raise RuntimeError('Temporary path escaped managed work directory')
            for path in work.iterdir():
                if path.is_file() and (path.name.startswith('source.') or path.name == 'converted.flac'):
                    path.unlink()
            if not any(work.iterdir()):
                work.rmdir()
        except KeyboardInterrupt:
            record.update(status='interrupted', error='User interrupted; partial files retained for resume')
            stop_batch = True
        except Exception as exc:
            record['error'] = str(exc)
        with journal.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
        batch_rows.append(record)
        save_csv(context/'train_audio_last_batch.csv', batch_rows,
                 ['time', 'video_id', 'status', 'duration_s', 'size_mb', 'audio_path', 'error', 'log'])
        ready = export_available()
        print(f'[{record["status"].upper()}] {vid} | available clips={len(ready)} | '
              f'audio folder={folder_bytes(audio_root)/1e9:.2f} GB | free={shutil.disk_usage(audio_root).free/1e9:.1f} GB', flush=True)
        if record['error']:
            print('[DETAIL]', record['error'], flush=True)
        if stop_batch:
            break
    ready = export_available()
    print(f'\n[DONE] attempted={attempts} succeeded={succeeded} existing_skipped={skipped} '
          f'| available clips={len(ready)} | videos={len({r["video_id"] for r in ready})}', flush=True)
    print('Alignment is still PENDING. Verify dataset clips vs original cuts before gate training.', flush=True)


if __name__ == '__main__':
    main()
