import threading
import time
import traceback
import numpy as np
import torch
import torchaudio
from pipeline.inference import SR

GPU_LOCK = threading.Lock()


class StreamSession:
    """Nhận âm thanh từng khúc nhỏ; luồng nền nhận dạng và dịch phần đang nói (bản tạm) và chốt câu khi có khoảng lặng."""

    def __init__(self, st, min_silence=0.6, max_len=15.0, pad=0.15, partial_every=0.8):
        self.st, self.min_silence, self.max_len, self.pad, self.partial_every = st, min_silence, max_len, pad, partial_every
        self.lock = threading.Lock()
        self.buf = torch.zeros(0)
        self.t0 = 0.0
        self.segments = []
        self.speaking = False
        self.partial_vi = ""
        self.partial_en = ""
        self.partial_start = 0.0
        self.last_infer = 0.0
        self.last_push = time.monotonic()
        self.stop_flag = False
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def push(self, sr, chunk):
        x = np.asarray(chunk)
        x = x.astype(np.float32) / 32768.0 if x.dtype.kind in "iu" else x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        t = torch.from_numpy(x)
        if sr != SR:
            t = torchaudio.functional.resample(t, sr, SR)
        with self.lock:
            self.buf = torch.cat([self.buf, t])
            self.last_push = time.monotonic()

    def close(self):
        self.stop_flag = True
        self.worker.join(timeout=60)
        try:
            self._step(force=True)
        except Exception:
            traceback.print_exc()

    def _loop(self):
        while not self.stop_flag:
            if time.monotonic() - self.last_push > 20:
                break
            try:
                did = self._step(force=False)
            except Exception:
                traceback.print_exc()
                did = False
                time.sleep(1.0)
            if not did:
                time.sleep(0.15)

    def _step(self, force):
        from silero_vad import get_speech_timestamps
        with self.lock:
            buf = self.buf.clone()
            t0 = self.t0
        n = len(buf) / SR
        ts = []
        if len(buf) >= SR // 4:
            ts = get_speech_timestamps(buf, self.st.vad, sampling_rate=SR, min_silence_duration_ms=300,
                                       speech_pad_ms=100, min_speech_duration_ms=100)
        if not ts:
            self.speaking = False
            self.partial_vi = self.partial_en = ""
            if n > 3.0:
                cut = len(buf) - SR
                with self.lock:
                    self.buf = self.buf[cut:]
                    self.t0 += cut / SR
            return False
        s, e = ts[0]["start"], ts[-1]["end"]
        silence = (len(buf) - e) / SR
        self.speaking = silence < self.min_silence
        a = max(0, s - int(self.pad * SR))
        if force or silence >= self.min_silence or n >= self.max_len:
            b = min(len(buf), e + int(self.pad * SR))
            with GPU_LOCK:
                res = self.st.translate_audio(buf[a:b])
            new = [dict(g, start=round(t0 + a / SR + g["start"], 2), end=round(t0 + a / SR + g["end"], 2)) for g in res["segments"]]
            with self.lock:
                self.segments.extend(new)
                self.buf = self.buf[b:]
                self.t0 += b / SR
            self.partial_vi = self.partial_en = ""
            return True
        if time.monotonic() - self.last_infer >= self.partial_every:
            with GPU_LOCK:
                res = self.st.translate_audio(buf[a:])
            self.partial_vi, self.partial_en = res["vi"], res["en"]
            self.partial_start = round(t0 + a / SR, 2)
            self.last_infer = time.monotonic()
            return True
        return False
