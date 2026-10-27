"""Notebook child-process cleanup only; importing this module starts no jobs."""


def run_cli(project, python, arguments):
    """Stream CR/newline progress and show a heartbeat while a child is quiet."""
    import datetime
    import os
    import queue
    import re
    import subprocess
    import threading
    import time

    logs = project / 'local/logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log = logs / f'{stamp}.log'
    command = [str(python), '-u', *map(str, arguments)]
    print('[START]', subprocess.list2cmdline(command), flush=True)
    print('[LOG]', log, flush=True)
    started = last_output = time.monotonic()
    events = queue.Queue()
    with log.open('w', encoding='utf-8') as handle:
        with subprocess.Popen(
            command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1', 'USE_TF': '0'},
        ) as child:
            def read_output():
                buffer = []
                try:
                    while True:
                        char = child.stdout.read(1)
                        if not char:
                            break
                        if char in '\r\n':
                            if buffer:
                                events.put(''.join(buffer)); buffer.clear()
                        else:
                            buffer.append(char)
                    if buffer:
                        events.put(''.join(buffer))
                finally:
                    events.put(None)

            threading.Thread(target=read_output, daemon=True).start()
            try:
                while True:
                    try:
                        line = events.get(timeout=1)
                        if line is None:
                            break
                        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
                        print(line, flush=True)
                        handle.write(line + '\n'); handle.flush()
                        last_output = time.monotonic()
                    except queue.Empty:
                        if time.monotonic() - last_output >= 10:
                            message = f'[RUNNING] {time.monotonic()-started:.0f}s elapsed; waiting for the next model/data update.'
                            print(message, flush=True)
                            handle.write(message + '\n'); handle.flush()
                            last_output = time.monotonic()
                result = child.wait()
            except KeyboardInterrupt:
                stop_process_tree(child.pid)
                child.wait()
                print('[STOPPED] Resume from the last completed checkpoint.', flush=True)
                raise
    print(f'[END] exit={result} | elapsed={time.monotonic()-started:.1f}s | log={log}', flush=True)
    if result:
        raise RuntimeError(f'Child process failed ({result}); read the error above. Log: {log}')


def stop_process_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
