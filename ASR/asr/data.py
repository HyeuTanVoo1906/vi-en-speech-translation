"""Load prepared parquet splits and turn them into Whisper training batches."""
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, concatenate_datasets, load_dataset

from asr.metrics import normalize_text

SR = 16000


def decode_audio(flac_bytes: bytes) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(flac_bytes), dtype="float32")
    assert sr == SR, f"expected {SR} Hz, got {sr} (re-run prepare_data.py)"
    return wav


def _cap_hours(ds: Dataset, max_hours: float, seed: int) -> Dataset:
    ds = ds.shuffle(seed=seed)
    n = int(np.searchsorted(np.cumsum(ds["duration"]), max_hours * 3600))
    return ds.select(range(n))


def load_split(cfg: dict, split: str, max_per_source: int | None = None) -> Dataset:
    """train: data.train_sources {source: max_hours|null}; other splits: data.eval_sources (missing files skipped)."""
    d, seed = cfg["data"], cfg["seed"]
    sources = d["train_sources"] if split == "train" else dict.fromkeys(d["eval_sources"])
    exclude = set()
    if split == "train" and d.get("train_exclude"):  # json {"ids": {id: reason}} next to the parquet sources
        exclude = set(json.loads((Path(d["dir"]) / d["train_exclude"]).read_text(encoding="utf-8"))["ids"])
    parts = []
    for source, max_hours in sources.items():
        path = Path(d["dir"]) / source / f"{split}.parquet"
        if not path.exists():
            if split == "train":
                raise FileNotFoundError(path)
            continue
        ds = load_dataset("parquet", data_files=str(path), split="train")
        if exclude:
            n = len(ds)
            ds = ds.filter(lambda i: i not in exclude, input_columns="id")
            print(f"Exclude {source}/{split}: {n - len(ds)} rows listed in {d['train_exclude']}", flush=True)
        empty_indices = [i for i, dur in enumerate(ds["duration"]) if dur <= 0]
        if empty_indices:
            print(f"Exclude {source}/{split}: {len(empty_indices)} empty-audio rows "
                  f"(duration <= 0): {ds.select(empty_indices)['id']}", flush=True)
        ds = ds.filter(lambda dur: 0 < dur <= d["max_duration"], input_columns="duration")
        if max_hours is not None:
            ds = _cap_hours(ds, max_hours, seed)
        if max_per_source and len(ds) > max_per_source:
            ds = ds.shuffle(seed=seed).select(range(max_per_source))
        parts.append(ds)
    if not parts:
        raise FileNotFoundError(f"No '{split}' parquet for {list(sources)} in {d['dir']}")
    return concatenate_datasets(parts)


class Featurizer:
    """On-the-fly transform (ds.with_transform): audio -> log-mel, text -> label ids."""

    def __init__(self, processor, normalize: bool):
        self.processor, self.normalize = processor, normalize

    def __call__(self, batch: dict) -> dict:
        feats = self.processor.feature_extractor(
            [decode_audio(b) for b in batch["audio"]], sampling_rate=SR
        ).input_features
        texts = [normalize_text(t) if self.normalize else t for t in batch["text"]]
        return {"input_features": feats, "labels": self.processor.tokenizer(texts).input_ids}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all():  # model prepends it itself
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def to_device_features(processor, audios: list[bytes], device: str, dtype: torch.dtype) -> torch.Tensor:
    feats = processor.feature_extractor([decode_audio(a) for a in audios], sampling_rate=SR, return_tensors="pt")
    return feats.input_features.to(device, dtype)
