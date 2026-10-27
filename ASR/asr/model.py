"""Build Whisper model/processor from config, optionally with LoRA."""
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asr.config import ROOT


def load_model(cfg: dict, for_training: bool = False, checkpoint: Path | None = None):
    m = cfg["model"]
    local = ROOT / m["name"]  # local weights dir (relative to ASR/) or HF id
    name = str(local) if local.is_dir() else m["name"]
    processor = WhisperProcessor.from_pretrained(name, language=m["language"], task=m["task"])

    # model.dtype: float16 loads the frozen base in half precision (medium does not fit 6 GB in fp32); peft keeps
    # the trainable LoRA weights in fp32 (autocast_adapter_dtype), so fp16 AMP + GradScaler still works.
    kw = {"torch_dtype": getattr(torch, m["dtype"]), "low_cpu_mem_usage": True} if m.get("dtype") else {}
    if checkpoint is not None and m.get("lora") is None:  # full fine-tune: checkpoint holds all weights
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint, **kw)
    else:
        if local.is_dir():  # pread, not the default mmap: mmap loads intermittently access-violate here (low commit memory)
            kw["state_dict"] = load_file(str(local / "model.safetensors"), backend="pread")
        model = WhisperForConditionalGeneration.from_pretrained(name, **kw)
    model.generation_config.language = m["language"]
    model.generation_config.task = m["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None

    lora = m.get("lora")
    if checkpoint is not None and lora is not None:
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    elif for_training:
        model.config.use_cache = False
        if m.get("spec_augment"):  # Whisper's built-in SpecAugment on log-mel; masks only in train mode
            model.config.update({"apply_spec_augment": True, **m["spec_augment"]})
        if m.get("init_adapter"):  # continue training an existing LoRA adapter (path relative to ASR/)
            model = PeftModel.from_pretrained(model, ROOT / m["init_adapter"], is_trainable=True)
            model.print_trainable_parameters()
        elif lora is not None:
            model = get_peft_model(model, LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"], bias="none",
            ))
            model.print_trainable_parameters()
    if m.get("dtype"):  # Trainer's in-training eval feeds fp32 features to generate(); the half-precision base needs matching input
        half_dtype, generate = getattr(torch, m["dtype"]), model.generate

        def generate_half(*args, **kwargs):
            if kwargs.get("input_features") is not None:
                kwargs["input_features"] = kwargs["input_features"].to(half_dtype)
            return generate(*args, **kwargs)
        model.generate = generate_half
    return model, processor
