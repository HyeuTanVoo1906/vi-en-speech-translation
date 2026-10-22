import zipfile, pathlib
from huggingface_hub import hf_hub_download

p = hf_hub_download("vinai/PhoMT", "PhoMT.zip", repo_type="dataset", local_dir="MT/data/phomt_dl")
dest = pathlib.Path("MT/data/phomt_raw")
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(p) as z:
    for n in z.namelist():
        assert ".." not in n and not n.startswith("/"), n
    z.extractall(dest)

files = sorted(f for f in dest.rglob("*") if f.is_file())
for f in files:
    with open(f, "rb") as fh:
        n = sum(1 for _ in fh)
    print(f"{f.stat().st_size/1e6:8.1f} MB {n:>9} lines  {f.relative_to(dest)}")

print()
for f in files[:12]:
    if f.stat().st_size < 600e6:
        with open(f, encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "").rstrip("\n")[:160] for _ in range(2)]
        print(f.relative_to(dest), "->", head)
