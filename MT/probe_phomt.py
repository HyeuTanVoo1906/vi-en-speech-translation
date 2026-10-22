from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("vinai/PhoMT", files_metadata=True)
for s in info.siblings:
    print(f"{(s.size or 0)/1e6:9.1f} MB  {s.rfilename}")
