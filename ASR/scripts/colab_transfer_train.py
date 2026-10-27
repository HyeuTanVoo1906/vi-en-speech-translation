import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--grad-accum',type=int,default=8)
    a=p.parse_args()
    import torch
    from transformers import TrainerCallback
    from asr.config import load_experiment
    from local_train import run_training
    if not torch.cuda.is_available():raise SystemExit('Select a GPU runtime first')
    cfg,_=load_experiment('V1_phowhisper_small_lora',a.data_dir)
    cfg['exp_id']='V1_colab_retrain_original_splits'
    cfg['train'].update(batch_size=a.batch_size,grad_accum=a.grad_accum,num_workers=0)
    cfg['eval']['batch_size']=1
    run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True)
    manifest=run/'resolved_config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=cfg:
        raise SystemExit('Config changed: use a new run-dir')
    manifest.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    # Reuse the tested scaler/RNG resume implementation. No local GPU pacing.
    run_training(cfg,run,TrainerCallback())

if __name__=='__main__':main()
