"""Explicit local Trainer construction, independent of the Colab entry point."""
import numpy as np
import json
import random
import torch
from pathlib import Path
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from asr.config import log_run
from asr.data import Collator, Featurizer, load_split
from asr.metrics import compute_metrics
from asr.model import load_model


def run_training(cfg, run, callback, smoke=False):
    set_seed(cfg['seed'])
    t, ev = cfg['train'], cfg['eval']
    model, processor = load_model(cfg, for_training=True)
    featurize = Featurizer(processor, cfg['data']['normalize_text'])
    print('[DATA] Loading TRAIN parquet and applying duration filters...', flush=True)
    train = load_split(cfg, 'train', 2 if smoke else None).with_transform(featurize)
    print(f'[DATA] TRAIN: {len(train)} clips. Loading DEV...', flush=True)
    dev = load_split(cfg, 'dev', 2 if smoke else cfg['data']['dev_max_per_source']).with_transform(featurize)
    print(f'[DATA] DEV: {len(dev)} clips.', flush=True)
    pad = processor.tokenizer.pad_token_id

    def metrics(pred):
        hypotheses = np.where(pred.predictions == -100, pad, pred.predictions)
        labels = np.where(pred.label_ids == -100, pad, pred.label_ids)
        m = compute_metrics(processor.batch_decode(labels, skip_special_tokens=True),
                            processor.batch_decode(hypotheses, skip_special_tokens=True))
        return {k: m[k] for k in ['wer', 'cer']}

    ckpt = run / 'checkpoints'
    args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt), seed=cfg['seed'], data_seed=cfg['seed'],
        num_train_epochs=t['epochs'], max_steps=2 if smoke else -1,
        per_device_train_batch_size=1 if smoke else t['batch_size'],
        per_device_eval_batch_size=t.get('eval_batch_size', 1),
        gradient_accumulation_steps=1 if smoke else t['grad_accum'],
        learning_rate=t['lr'], warmup_ratio=t['warmup_ratio'], lr_scheduler_type='linear',
        fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=1 if smoke else t['eval_steps'],
        save_strategy='steps', save_steps=1 if smoke else t['save_steps'],
        save_total_limit=t['save_total_limit'],
        load_best_model_at_end=True, metric_for_best_model='wer', greater_is_better=False,
        predict_with_generate=True, generation_max_length=ev['max_new_tokens'], generation_num_beams=1,
        logging_steps=1 if smoke else t['logging_steps'], logging_first_step=True,
        disable_tqdm=False, report_to='none',
        remove_unused_columns=False, label_names=['labels'],
        dataloader_num_workers=0, dataloader_pin_memory=False, eval_accumulation_steps=1,
    )
    class LocalTrainer(Seq2SeqTrainer):
        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            scaler = self.accelerator.scaler
            if scaler is not None:
                torch.save(scaler.state_dict(),
                           ckpt / f'checkpoint-{self.state.global_step}' / 'scaler.pt')

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            scaler = self.accelerator.scaler
            if checkpoint is not None and scaler is not None:
                path = Path(checkpoint) / 'scaler.pt'
                if not path.is_file():
                    raise FileNotFoundError(f'Incomplete FP16 resume checkpoint: {path}')
                scaler.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

        def _load_rng_state(self, checkpoint):
            # Transformers 4.46 predates torch 2.6's weights_only default.
            # Only deserialize RNG state produced inside this local run.
            if checkpoint is None:
                return
            path = Path(checkpoint).resolve()
            if not path.is_relative_to(ckpt.resolve()):
                raise ValueError('RNG checkpoint must belong to this local run')
            rng_file = path / 'rng_state.pth'
            if not rng_file.is_file():
                raise FileNotFoundError(f'Incomplete resume checkpoint: {rng_file}')
            state = torch.load(rng_file, map_location='cpu', weights_only=False)
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.random.set_rng_state(state['cpu'])
            if torch.cuda.is_available() and 'cuda' in state:
                torch.cuda.random.set_rng_state(state['cuda'])

    trainer = LocalTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                            data_collator=Collator(processor, model.config.decoder_start_token_id),
                            compute_metrics=metrics, processing_class=processor.feature_extractor,
                            callbacks=[callback])
    last = get_last_checkpoint(str(ckpt)) if ckpt.exists() else None
    log_run(cfg, run, 'local_train', resume_from=last, n_train=len(train), n_dev=len(dev), smoke=smoke)
    print(f'train={len(train)} dev={len(dev)} resume_from_checkpoint={last}', flush=True)
    if last:
        saved = json.loads((Path(last)/'trainer_state.json').read_text())
        if smoke and saved['global_step'] >= 2:
            # Load states through Trainer only if there are unfinished steps.
            print('Two-step smoke already completed. Resume checkpoint verified on disk.', flush=True)
            return
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(run/'final'))
    processor.save_pretrained(str(run/'final'))
    print('Saved local best adapter:', run/'final', flush=True)
