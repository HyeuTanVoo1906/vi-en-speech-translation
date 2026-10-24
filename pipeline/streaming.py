import numpy as np
import torch
import torchaudio
from pipeline.inference import SR


class StreamSession:
    def __init__(self, st, min_silence=1.0, max_len=20.0, pad=0.15):
        self.st, self.min_silence, self.max_len, self.pad = st, min_silence, max_len, pad
        self.buf = torch.zeros(0)
        self.t0 = 0.0
        self.segments = []
        self.speaking = False

    def push(self, sr, chunk):
        x = np.asarray(chunk)
        x = x.astype(np.float32) / 32768.0 if x.dtype.kind in "iu" else x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        t = torch.from_numpy(x)
        if sr != SR:
            t = torchaudio.functional.resample(t, sr, SR)
        self.buf = torch.cat([self.buf, t])
        return self._flush(force=False)

    def flush(self):
        return self._flush(force=True)

    def _flush(self, force):
        from silero_vad import get_speech_timestamps
        new = []
        ts = []
        if len(self.buf) >= SR // 4:
            ts = get_speech_timestamps(self.buf, self.st.vad, sampling_rate=SR,
                                       min_silence_duration_ms=300, speech_pad_ms=100, min_speech_duration_ms=100)
        n = len(self.buf) / SR
        if not ts:
            self.speaking = False
            if n > 3.0:
                keep = self.buf[-SR:]
                self.t0 += (len(self.buf) - len(keep)) / SR
                self.buf = keep
            return new
        s, e = ts[0]["start"], ts[-1]["end"]
        silence = (len(self.buf) - e) / SR
        self.speaking = silence < self.min_silence
        if force or silence >= self.min_silence or n >= self.max_len:
            a = max(0, s - int(self.pad * SR))
            b = min(len(self.buf), e + int(self.pad * SR))
            res = self.st.translate_audio(self.buf[a:b])
            for g in res["segments"]:
                g = dict(g, start=round(self.t0 + a / SR + g["start"], 2), end=round(self.t0 + a / SR + g["end"], 2))
                self.segments.append(g)
                new.append(g)
            self.buf = self.buf[b:]
            self.t0 += b / SR
            self.speaking = False
        return new
