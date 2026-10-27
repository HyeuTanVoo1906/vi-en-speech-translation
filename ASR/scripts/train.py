"""Train one experiment. Checkpoints go to experiments/<ID>/checkpoints and resume automatically.

    python scripts/train.py V1_phowhisper_small_lora [--data-dir /content/data]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr.config import load_experiment, log_run
from asr.data import Collator, Featurizer, load_split
from asr.metrics import compute_metrics
from asr.model import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--data-dir", help="override data.dir (e.g. local copy on Colab)")
    args = ap.parse_args()

    cfg, exp_dir = load_experiment(args.exp_id, args.data_dir)
    t, ev = cfg["train"], cfg["eval"]
    if not t:
        sys.exit(f"{args.exp_id} has no 'train' section -> evaluation-only, use scripts/evaluate.py")
    set_seed(cfg["seed"])

    model, processor = load_model(cfg, for_training=True)
    featurize = Featurizer(processor, cfg["data"]["normalize_text"])
    train_ds = load_split(cfg, "train").with_transform(featurize)
    dev_ds = load_split(cfg, "dev", cfg["data"]["dev_max_per_source"]).with_transform(featurize)
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    pad_id = processor.tokenizer.pad_token_id

    def metrics_fn(pred):
        preds = np.where(pred.predictions == -100, pad_id, pred.predictions)
        labels = np.where(pred.label_ids == -100, pad_id, pred.label_ids)
        m = compute_metrics(processor.batch_decode(labels, skip_special_tokens=True),
                            processor.batch_decode(preds, skip_special_tokens=True))
        return {"wer": m["wer"], "cer": m["cer"]}

    ckpt_dir = exp_dir / "checkpoints"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        seed=cfg["seed"], data_seed=cfg["seed"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=ev["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"], warmup_ratio=t["warmup_ratio"], lr_scheduler_type="linear",
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=t["eval_steps"],
        save_strategy="steps", save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=ev["max_new_tokens"], generation_num_beams=1,
        logging_steps=t["logging_steps"], report_to="none",
        remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=t["num_workers"],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=metrics_fn, processing_class=processor.feature_extractor,
    )

    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    log_run(cfg, exp_dir, "train", resume_from=last, n_train=len(train_ds), n_dev=len(dev_ds))
    print(f"resume_from_checkpoint={last}")
    trainer.train(resume_from_checkpoint=last)

    final = exp_dir / "final"  # best checkpoint by dev WER (LoRA: adapter only)
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print(f"saved best model -> {final}")


if __name__ == "__main__":
    main()
