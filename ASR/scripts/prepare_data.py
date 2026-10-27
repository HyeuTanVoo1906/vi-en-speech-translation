"""Convert raw ASR corpora into one uniform format: data/<SOURCE>/<split>.parquet (16 kHz mono FLAC bytes).

Run once (locally or on Colab), then upload ASR/data/ to Drive.
    python scripts/prepare_data.py --raw-dir ../dataset/ASR

Not converted on purpose:
- VietMed-NER: audio likely overlaps VietMed -> leakage risk if mixed.
- TED2020: text only, not an ASR corpus.
"""
import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

SR = 16000
SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("split", pa.string()),
    ("group_id", pa.string()),   # unit that must stay in one split (speaker / recording / video / sentence)
    ("speaker_id", pa.string()), ("text", pa.string()),
    ("cs_terms", pa.string()),   # ';'-separated code-switch terms ('' if not annotated)
    ("duration", pa.float32()), ("audio", pa.binary()),
])


def to_flac(data: bytes | str | Path) -> tuple[bytes, float]:
    src = io.BytesIO(data) if isinstance(data, bytes) else data
    wav, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1, 1), SR, format="FLAC")
    return buf.getvalue(), len(wav) / SR


def row(id, source, split, group_id, speaker_id, text, audio, cs_terms=""):
    flac, dur = to_flac(audio)
    return dict(id=id, source=source, split=split, group_id=group_id or "", speaker_id=speaker_id or "",
                text=text.strip(), cs_terms=cs_terms or "", duration=dur, audio=flac)


def write(path: Path, rows, chunk: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    n, buf = 0, []
    with pq.ParquetWriter(tmp, SCHEMA) as w:
        for r in rows:
            buf.append(r)
            if len(buf) == chunk:
                w.write_table(pa.Table.from_pylist(buf, SCHEMA)); n += len(buf); buf = []
                print(f"\r  {path.parent.name}/{path.name}: {n}", end="", flush=True)
        if buf:
            w.write_table(pa.Table.from_pylist(buf, SCHEMA)); n += len(buf)
    tmp.replace(path)  # atomic: a crashed run never leaves a half-written split
    print(f"\r  {path.parent.name}/{path.name}: {n} rows")


# ---------------------------------------------------------------- sources
def vivos(raw: Path):
    """Official train/test. DEV = ~5% of TRAIN speakers (deterministic hash) -> speaker-disjoint."""
    def items(part):
        for line in (raw / part / "prompts.txt").read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            spk = uid.split("_")[0]
            yield uid, spk, text.lower(), raw / part / "waves" / spk / f"{uid}.wav"

    def is_dev(spk):
        return int(hashlib.md5(spk.encode()).hexdigest(), 16) % 20 == 0

    def split_rows(part, split, keep):
        for uid, spk, text, wav in items(part):
            if keep(spk):
                yield row(f"vivos:{uid}", "VIVOS", split, spk, spk, text, wav)

    return {
        "train": lambda: split_rows("train", "train", lambda s: not is_dev(s)),
        "dev": lambda: split_rows("train", "dev", is_dev),
        "test": lambda: split_rows("test", "test", lambda s: True),
    }


def fleurs(raw: Path):
    """TSV: id, file, raw text, normalized, chars, n_samples, gender. Audio inside audio/<split>.tar.gz."""
    base = raw / "data" / "vi_vn"

    def split_rows(split):
        meta = {}
        for line in (base / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            meta[cols[1]] = cols
        with tarfile.open(base / "audio" / f"{split}.tar.gz") as tar:
            for m in tar:
                cols = meta.get(Path(m.name).name)
                if m.isfile() and cols:
                    # group = sentence id: several speakers may read the same sentence
                    yield row(f"fleurs:{split}:{Path(m.name).stem}", "FLEURS", split, cols[0], "",
                              cols[2], tar.extractfile(m).read())

    return {s: (lambda s=s: split_rows(s)) for s in ("train", "dev", "test")}


def parquet_rows(files, convert):
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            for r in batch.to_pylist():
                yield convert(r)


def vietmed(raw: Path):
    def split_rows(split):
        return parquet_rows(sorted((raw / "data").glob(f"{split}-*.parquet")), lambda r: row(
            f"vietmed:{r['utterance_id']}", "VietMed", split, r["audio_name"], r["speaker_name"],
            r["text"], r["audio"]["bytes"]))

    return {s: (lambda s=s: split_rows(s)) for s in ("train", "dev", "test")}


def vimedcss(raw: Path):
    """Official splits; 'validation' -> dev. group = source video. 'hard' kept as its own split."""
    def split_rows(split, file_prefix):
        return parquet_rows(sorted((raw / "data").glob(f"{file_prefix}-*.parquet")), lambda r: row(
            f"vimedcss:{r['segment_id']}", "ViMedCSS", split, r["original_video_link"], "",
            r["segment_text"], r["audio"]["bytes"], r["cs_terms_list"]))

    names = {"train": "train", "dev": "validation", "test": "test", "hard": "hard"}
    return {s: (lambda s=s, p=p: split_rows(s, p)) for s, p in names.items()}


SOURCES = {"VIVOS": ("VIVOS", vivos), "FLEURS": ("FLEURS_vi_vn", fleurs),
           "VietMed": ("VietMed", vietmed), "ViMedCSS": ("ViMedCSS", vimedcss)}


# ---------------------------------------------------------------- registry
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_registry(out: Path) -> None:
    reg = {}
    for p in sorted(out.glob("*/*.parquet")):
        meta = pq.read_table(p, columns=["duration", "group_id"])
        reg[f"{p.parent.name}/{p.stem}"] = {
            "rows": meta.num_rows,
            "hours": round(float(np.sum(meta["duration"].to_numpy())) / 3600, 3),
            "groups": len(set(meta["group_id"].to_pylist())),
            "sha256": sha256(p),
        }
    (out / "registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(json.dumps(reg, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="../dataset/ASR")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)

    for name in args.sources:
        folder, builder = SOURCES[name]
        print(f"[{name}]")
        for split, rows in builder(raw / folder).items():
            path = out / name / f"{split}.parquet"
            if path.exists() and not args.overwrite:
                print(f"  {name}/{split}.parquet exists, skip")
                continue
            write(path, rows())
    write_registry(out)


if __name__ == "__main__":
    main()
