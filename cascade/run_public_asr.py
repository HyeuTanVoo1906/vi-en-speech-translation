import argparse, json, time
from pathlib import Path
import jiwer, soundfile as sf, torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from common.textnorm import normalize_text

MODELS = {"B0": "vinai/PhoWhisper-small", "B1": "openai/whisper-small"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=MODELS, required=True)
ap.add_argument("--set", dest="set_name", required=True, help="vd: fleurs_dev")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out", default="")
args = ap.parse_args()

rows = [json.loads(l) for l in open(f"cascade/data/{args.set_name}.jsonl", encoding="utf-8")]
if args.limit:
    rows = rows[: args.limit]
out_path = Path(args.out or f"runs/asr_public/{args.set_name}_{args.model}.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

name = MODELS[args.model]
proc = WhisperProcessor.from_pretrained(name)
model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
model.generation_config.forced_decoder_ids = None

order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])  # gom cau dai gan nhau
hyps = [""] * len(rows)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t0 = time.time()
for b in tqdm(range(0, len(order), args.batch), desc=f"{args.model} {args.set_name}"):
    idx = order[b : b + args.batch]
    wavs = [sf.read(rows[i]["audio_path"], dtype="float32")[0] for i in idx]
    feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to("cuda", torch.float16)
    with torch.no_grad():
        out = model.generate(feats, language="vi", task="transcribe", num_beams=1, max_new_tokens=225)
    for i, t in zip(idx, proc.batch_decode(out, skip_special_tokens=True)):
        hyps[i] = t.strip()
torch.cuda.synchronize()
dt = time.time() - t0

with out_path.open("w", encoding="utf-8") as f:
    for r, h in zip(rows, hyps):
        f.write(json.dumps({
            "id": r["id"], "source": r["source"], "split": r["split"], "asr_model": args.model,
            "hyp_raw": h, "hyp_norm": normalize_text(h), "text": r["text_raw"],
            "text_norm": r["text_norm"], "en_ref": r["en_ref"], "duration": r["duration"],
        }, ensure_ascii=False) + "\n")

refs = [r["text_norm"] for r in rows]
hn = [normalize_text(h) for h in hyps]
audio_s = sum(r["duration"] for r in rows)
print(f"{args.model} {args.set_name}: n={len(rows)} WER={100*jiwer.wer(refs, hn):.2f}% CER={100*jiwer.cer(refs, hn):.2f}% "
      f"RTF={dt/audio_s:.3f} peakVRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB -> {out_path}")
for r, h in list(zip(rows, hyps))[:3]:
    print("  GOLD:", r["text_raw"]); print("  HYP :", h)
