import importlib.metadata as md, json, sys
import torch

pkgs = ["torch", "transformers", "peft", "accelerate", "sentencepiece", "numpy",
        "sacrebleu", "tensorboard", "tqdm", "soundfile", "librosa", "pandas",
        "pyarrow", "datasets", "pytest"]
env = {"python": sys.version.split()[0]}
for p in pkgs:
    try:
        env[p] = md.version(p)
    except md.PackageNotFoundError:
        env[p] = "MISSING"
env["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    env["gpu"] = torch.cuda.get_device_name(0)
    env["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
print(json.dumps(env, indent=2))
