import time
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from common.textnorm import normalize_text
import os
from pathlib import Path
from MT.mt.model import Translator
from MT.mt.noise import verbalize_numbers

SR = 16000
DEFAULT_MT_ADAPTER = "runs/mt/M4_s42/adapter"
ROOT = Path(__file__).resolve().parents[1]


def load_audio(path):
    x, s = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[0] == 0:
        raise ValueError("File âm thanh rỗng")
    t = torch.from_numpy(x.mean(axis=1))
    if s != SR:
        t = torchaudio.functional.resample(t, s, SR)
    return t


def make_chunks(ts, max_s=25.0, max_gap=0.8):
    chunks = []
    for t in ts:
        s, e = t["start"] / SR, t["end"] / SR
        if chunks and s - chunks[-1][1] <= max_gap and e - chunks[-1][0] <= max_s:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    return chunks


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class SpeechTranslator:
    def __init__(self, asr_name="vinai/PhoWhisper-small", asr_adapter=None, mt_adapter=DEFAULT_MT_ADAPTER, device="cuda"):
        from silero_vad import load_silero_vad
        self.dev = device
        self.vad = load_silero_vad()
        self.proc = WhisperProcessor.from_pretrained(asr_name)
        asr = WhisperForConditionalGeneration.from_pretrained(asr_name)
        if asr_adapter:
            from peft import PeftModel
            asr = PeftModel.from_pretrained(asr, asr_adapter).merge_and_unload()
        asr.generation_config.forced_decoder_ids = None
        self.asr = asr.half().to(device).eval()
        if mt_adapter:
            if not os.path.isabs(mt_adapter):
                mt_adapter = str(ROOT / mt_adapter)
            if not os.path.isdir(mt_adapter):
                raise FileNotFoundError(f"Không tìm thấy adapter MT: {mt_adapter}")
        self.mt = Translator("vinai", device=device, adapter=mt_adapter)

    @torch.no_grad()
    def transcribe(self, wavs):
        feats = self.proc(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(self.dev, torch.float16)
        ids = self.asr.generate(feats, language="vi", task="transcribe", num_beams=1, max_new_tokens=225)
        return [t.strip() for t in self.proc.batch_decode(ids, skip_special_tokens=True)]

    def translate_audio(self, wav):
        from silero_vad import get_speech_timestamps
        dur = len(wav) / SR
        _sync(); t0 = time.perf_counter()
        ts = get_speech_timestamps(wav, self.vad, sampling_rate=SR, min_silence_duration_ms=300,
                                   max_speech_duration_s=25, speech_pad_ms=100)
        chunks = make_chunks(ts)
        _sync(); t1 = time.perf_counter()
        vi = []
        for i in range(0, len(chunks), 8):
            vi += self.transcribe([wav[int(s * SR):int(e * SR)].numpy() for s, e in chunks[i:i + 8]])
        _sync(); t2 = time.perf_counter()
        en = self.mt.translate([normalize_text(verbalize_numbers(v)) for v in vi], batch_size=8, desc="mt")
        _sync(); t3 = time.perf_counter()
        segs = [{"start": round(c[0], 2), "end": round(c[1], 2), "vi": v, "en": e}
                for c, v, e in zip(chunks, vi, en) if v.strip()]
        return {"segments": segs,
                "vi": " ".join(s["vi"] for s in segs), "en": " ".join(s["en"] for s in segs),
                "stats": {"audio_s": round(dur, 2), "vad_s": round(t1 - t0, 2), "asr_s": round(t2 - t1, 2),
                          "mt_s": round(t3 - t2, 2), "rtf": round((t3 - t0) / max(dur, 1e-6), 3),
                          "segments": len(segs)}}

    def translate_file(self, path):
        return self.translate_audio(load_audio(path))
