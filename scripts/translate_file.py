import json, sys
from pipeline.inference import SpeechTranslator

st = SpeechTranslator()
r = st.translate_file(sys.argv[1])
for s in r["segments"]:
    print(f"[{s['start']:.1f}-{s['end']:.1f}s] VI: {s['vi']}")
    print(f"              EN: {s['en']}")
print(json.dumps(r["stats"], ensure_ascii=False))
