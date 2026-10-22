from datasets import load_dataset, get_dataset_config_names

print("== 1) openlanguagedata/flores_plus ==")
try:
    vi = load_dataset("openlanguagedata/flores_plus", "vie_Latn", split="dev")
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="dev")
    print("OK", len(vi), len(en), vi.column_names)
    print(vi[0]); print(en[0])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])

print("\n== 2) Muennighoff/flores200 (mirror) ==")
try:
    names = get_dataset_config_names("Muennighoff/flores200", trust_remote_code=True)
    cand = [n for n in names if "vie_Latn" in n and "eng_Latn" in n]
    print("configs co vie+eng:", cand[:5], "| tong so config:", len(names))
    if cand:
        ds = load_dataset("Muennighoff/flores200", cand[0], trust_remote_code=True)
        print(ds)
        split = list(ds.keys())[0]
        print(ds[split][0])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])
