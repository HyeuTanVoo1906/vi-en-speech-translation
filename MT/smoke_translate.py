import time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NAME = "vinai/vinai-translate-vi2en-v2"
tok = AutoTokenizer.from_pretrained(NAME, src_lang="vi_VN")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME).half().cuda().eval()
en_id = getattr(tok, "lang_code_to_id", {}).get("en_XX") or tok.convert_tokens_to_ids("en_XX")

CASES = [
    ("raw",       "Tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng, hỗ trợ kích thích sự phát triển của nang trứng."),
    ("normalized","tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng hỗ trợ kích thích sự phát triển của nang trứng"),
    ("asr-error", "tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng hỗ trợ kích thích sự phát triển của tế bào"),
    ("raw",       "Thông thường, hormone này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp."),
    ("asr-error", "thông thường hocmon này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp"),
    ("asr-error", "ester dion e2 các chuyên gia đánh giá ester dion là loại estrogen mạnh nhất"),
]

texts = [t for _, t in CASES]
batch = tok(texts, padding=True, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**batch, decoder_start_token_id=en_id, num_beams=5, early_stopping=True, max_length=128)
torch.cuda.synchronize()
dt = time.time() - t0
for (kind, vi), en in zip(CASES, tok.batch_decode(out, skip_special_tokens=True)):
    print(f"[{kind}] {vi}\n    -> {en}\n")
print(f"time {dt:.2f}s | peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB | params {sum(p.numel() for p in model.parameters())/1e6:.0f}M")
