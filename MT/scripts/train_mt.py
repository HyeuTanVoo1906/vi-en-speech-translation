import argparse, json, os, random, time
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

NAME = "vinai/vinai-translate-vi2en-v2"
ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--src-col", default="vi")
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--save-every", type=int, default=500)
ap.add_argument("--eval-n", type=int, default=300)
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

out = os.path.join("runs", "mt", args.exp)
os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "DONE")):
    print("DONE marker exists:", out); raise SystemExit
random.seed(args.seed); torch.manual_seed(args.seed)
logf = open(os.path.join(out, "train.log"), "a", encoding="utf-8")
def log(s):
    print(s); logf.write(s + "\n"); logf.flush()
log(f"args: {vars(args)}")

tok = AutoTokenizer.from_pretrained(NAME, src_lang="vi_VN", tgt_lang="en_XX")
pad, eos = tok.pad_token_id, tok.eos_token_id
dec_start = tok.convert_tokens_to_ids("en_XX")
L = args.max_len

def load(path, n):
    df = pd.read_parquet(path)
    df = df[df[args.src_col].str.strip().ne("") & df["en"].str.strip().ne("")]
    if n < len(df):
        df = df.sample(n=n, random_state=args.seed)
    return df.reset_index(drop=True)

def encode(df):
    src = tok(df[args.src_col].tolist(), truncation=True, max_length=L)["input_ids"]
    tgt = tok(text_target=df["en"].tolist(), add_special_tokens=False, truncation=True, max_length=L - 1)["input_ids"]
    return src, tgt

log("tokenizing...")
tr = load("MT/data/phomt_train_300k.parquet", args.n)
dv = load("MT/data/phomt_dev_2k.parquet", args.eval_n)
tr_src, tr_tgt = encode(tr)
dv_src, dv_tgt = encode(dv)
log(f"train {len(tr)} | dev {len(dv)} | src-col {args.src_col}")

def collate(ids, S, T):
    ls = max(len(S[i]) for i in ids); lt = max(len(T[i]) + 1 for i in ids)
    x = torch.full((len(ids), ls), pad); m = torch.zeros((len(ids), ls), dtype=torch.long)
    di = torch.full((len(ids), lt), pad); y = torch.full((len(ids), lt), -100)
    for j, i in enumerate(ids):
        s, t = S[i], T[i]
        x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = 1
        di[j, :len(t) + 1] = torch.tensor([dec_start] + t)
        y[j, :len(t) + 1] = torch.tensor(t + [eos])
    return x.cuda(), m.cuda(), di.cuda(), y.cuda()

def make_batches(epoch):
    rng = random.Random(args.seed + epoch)
    idx = list(range(len(tr_src))); rng.shuffle(idx)
    g = args.bs * args.accum * 20
    mbs = []
    for k in range(0, len(idx), g):
        chunk = sorted(idx[k:k + g], key=lambda i: len(tr_src[i]))
        mbs += [chunk[j:j + args.bs] for j in range(0, len(chunk), args.bs)]
    rng.shuffle(mbs)
    return mbs

ck = os.path.join(out, "ckpt")
base = AutoModelForSeq2SeqLM.from_pretrained(NAME).cuda()
base.config.use_cache = False
resume = args.resume and os.path.exists(os.path.join(ck, "state.pt"))
if resume:
    model = PeftModel.from_pretrained(base, ck, is_trainable=True)
else:
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_2_SEQ_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
    model = get_peft_model(base, cfg)
model.print_trainable_parameters()

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
all_mbs = []
for e in range(args.epochs):
    all_mbs += make_batches(e)
total = len(all_mbs) // args.accum
sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total)), total)
scaler = torch.cuda.amp.GradScaler()
step = 0
if resume:
    st = torch.load(os.path.join(ck, "state.pt"), map_location="cpu")
    opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"]); step = st["step"]
    log(f"resumed at step {step}")
try:
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(out, "tb"))
except Exception:
    tb = None

def lm_loss(batch):
    x, m, di, y = batch
    with torch.autocast("cuda", dtype=torch.float16):
        lg = model(input_ids=x, attention_mask=m, decoder_input_ids=di).logits
    return F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.view(-1), ignore_index=-100)

@torch.no_grad()
def dev_loss():
    model.eval(); tot, cnt = 0.0, 0
    order = sorted(range(len(dv_src)), key=lambda i: len(dv_src[i]))
    for k in range(0, len(order), 16):
        b = collate(order[k:k + 16], dv_src, dv_tgt)
        tot += lm_loss(b).item() * len(order[k:k + 16]); cnt += len(order[k:k + 16])
    model.train(); return tot / cnt

def save():
    model.save_pretrained(ck)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, os.path.join(ck, "state.pt"))

dl = None
model.train(); t0 = time.time()
bar = tqdm(total=total, initial=step, desc=args.exp, dynamic_ncols=True)
while step < total:
    opt.zero_grad(set_to_none=True); la = 0.0
    for a in range(args.accum):
        b = collate(all_mbs[step * args.accum + a], tr_src, tr_tgt)
        loss = lm_loss(b) / args.accum
        scaler.scale(loss).backward(); la += loss.item()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); sched.step(); step += 1
    bar.update(1); bar.set_postfix(loss=f"{la:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
    if tb: tb.add_scalar("train/loss", la, step)
    if step % 50 == 0: log(f"step {step}/{total} loss {la:.4f} elapsed {time.time()-t0:.0f}s")
    if step % args.save_every == 0 or step == total:
        dl = dev_loss(); log(f"== step {step} dev_loss {dl:.4f}")
        if tb: tb.add_scalar("dev/loss", dl, step)
        save()
bar.close()
model.save_pretrained(os.path.join(out, "adapter"))
json.dump({"exp": args.exp, "steps": total, "dev_loss": dl, "train_seconds": time.time() - t0,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9, "args": vars(args)},
          open(os.path.join(out, "summary.json"), "w"), indent=1)
open(os.path.join(out, "DONE"), "w").write("ok")
log(f"finished, peak VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
