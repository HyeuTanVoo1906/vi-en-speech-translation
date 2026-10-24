import sys, time
import numpy as np
from pipeline.inference import SpeechTranslator, load_audio, SR
from pipeline.streaming import StreamSession

st = SpeechTranslator()
gap = np.zeros(int(1.5 * SR), dtype=np.float32)
parts = []
for p in sys.argv[1:]:
    parts += [load_audio(p).numpy(), gap]
pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
sess = StreamSession(st)
step = SR // 2
worst = 0.0
for i in range(0, len(pcm), step):
    t = time.perf_counter()
    new = sess.push(SR, pcm[i:i + step])
    worst = max(worst, time.perf_counter() - t)
    for g in new:
        print(f"[audio t={(i + step) / SR:.1f}s] {g['start']:.1f}-{g['end']:.1f}s\n   VI: {g['vi']}\n   EN: {g['en']}")
for g in sess.flush():
    print(f"[flush] {g['start']:.1f}-{g['end']:.1f}s\n   VI: {g['vi']}\n   EN: {g['en']}")
print(f"segments {len(sess.segments)}, audio {len(pcm) / SR:.1f}s, slowest push {worst:.2f}s")
