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
seen, last = 0, ""
t_wall = time.perf_counter()
for i in range(0, len(pcm), step):
    sess.push(SR, pcm[i:i + step])
    time.sleep(0.5)
    now = (i + step) / SR
    with sess.lock:
        segs = list(sess.segments)
    for g in segs[seen:]:
        print(f"[t={now:.1f}s] CHOT {g['start']:.1f}-{g['end']:.1f}s\n    VI: {g['vi']}\n    EN: {g['en']}")
    seen = len(segs)
    if sess.partial_en and sess.partial_en != last:
        print(f"[t={now:.1f}s] tam: {sess.partial_en}")
        last = sess.partial_en
sess.close()
with sess.lock:
    segs = list(sess.segments)
for g in segs[seen:]:
    print(f"[flush] CHOT {g['start']:.1f}-{g['end']:.1f}s\n    VI: {g['vi']}\n    EN: {g['en']}")
print(f"segments {len(segs)}, audio {len(pcm) / SR:.1f}s, wall {time.perf_counter() - t_wall:.1f}s")
