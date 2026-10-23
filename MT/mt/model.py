import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SPECS = {
    "vinai": {"name": "vinai/vinai-translate-vi2en-v2", "src": "vi_VN", "tgt": "en_XX"},
    "nllb": {"name": "facebook/nllb-200-distilled-600M", "src": "vie_Latn", "tgt": "eng_Latn"},
}

class Translator:
    def __init__(self, kind, device="cuda", adapter=None):
        spec = SPECS[kind]
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(spec["name"], src_lang=spec["src"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(spec["name"]).half().to(device).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload().half().eval()
        tgt_id = getattr(self.tok, "lang_code_to_id", {}).get(spec["tgt"]) or self.tok.convert_tokens_to_ids(spec["tgt"])
        self.gen_kwargs = {"decoder_start_token_id": tgt_id} if kind == "vinai" else {"forced_bos_token_id": tgt_id}

    @torch.no_grad()
    def translate(self, texts, batch_size=32, num_beams=5, max_length=256, desc="translate"):
        uniq = sorted({t for t in texts if t.strip()}, key=lambda t: (len(t), t))
        out = {}
        for b in tqdm(range(0, len(uniq), batch_size), desc=desc):
            chunk = uniq[b : b + batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            gen = self.model.generate(**enc, num_beams=num_beams, max_length=max_length, early_stopping=True, **self.gen_kwargs)
            for s, h in zip(chunk, self.tok.batch_decode(gen, skip_special_tokens=True)):
                out[s] = h.strip()
        return [out.get(t, "") for t in texts]
