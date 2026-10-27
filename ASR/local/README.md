# Local ASR on RTX 4050 6 GB

Start with `ASR/notebooks/local/00_setup.ipynb`, then B0, B1, V1. B0/B1 are zero-shot evaluations, not training runs. `04_context_data` prepares a TRAIN manifest; it does not train an unimplemented gate. `05_compare_runs` compares results and memory measurements.

After the notebook 04 manifest is ready, `06_download_train_audio.ipynb` downloads TRAIN original audio in batches of at most 30 attempted videos. It is CPU-only and user-run. Managed storage: `local/imports/original_audio/train`, including `.work` source/conversion files. Defaults: 10 decimal GB audio budget, 20 GB free-space floor, 60-minute/video filter, 512 MiB source-file filter, one download at a time, 2 MiB/s rate limit. A one-second disk monitor stops the current child process with spare headroom; limits are sampled safeguards, not filesystem quotas. Interrupted partial files stay resumable. Converted native source files are removed only after a valid mono 16 kHz FLAC has been published. No raw dataset or training checkpoint cleanup is performed.

The downloader rechecks DEV source-video exclusion, validates timestamps and IDs, journals successes/failures, skips existing valid FLAC, and exports an available-audio manifest with pending alignment. It does not check the acoustic alignment or create gate labels. Error retries are opt-in. Tools are installed/downloaded only when the user executes the notebook setup cells; portable Deno is extracted from the official Windows x64 release. The script and notebook were prepared without executing a video download or model job.

## Environment

From the workspace root: `powershell -ExecutionPolicy Bypass -File ASR/local/setup.ps1`.
`.venv-asr` has pinned Colab ASR dependencies and reuses the existing CUDA PyTorch through system-site-packages. Global packages are not uninstalled or upgraded. This is not a fully hermetic environment; the run log records exact versions. Do not install unrelated packages into this venv.

## Resource profile

`ASR/local/profile.json` uses decimal GB. Default total GPU memory target is 4.5 GB, including observed pre-existing GPU use and 250 MB headroom. The PyTorch allocator receives the remaining budget. CUDA context, libraries, Windows WDDM accounting and other apps are outside the allocator cap. The total budget is checked after each generate/optimizer step; transient or external allocations can exceed it. This cannot guarantee all-device usage never crosses 4.5 GB.

`gpu_duty_cycle=0.7` inserts idle time equal to 3/7 of each measured compute interval. This is NOT a hardware GPU-utilization cap, does not set a power limit, and permits temporary 100% utilization. Lower it for more idle time. Training batch 1, accumulation 32, FP16, gradient checkpointing, zero loader workers, no pinned host memory, evaluation accumulation 1. No claim of equivalent bitwise results to Colab.

`host_ram_budget_gb=0` means monitor host RAM without a cap, pending clarification whether 4.5 GB means GPU or host RAM. A positive value enables post-step RSS checks; this is a sampled stop condition, not an OS hard memory cap. The notebook kernel and other apps are not included in child-process RSS.

## Storage and checkpoint policy

```
ASR/data/<source>/*.parquet                  # one prepared dataset shared by all models
ASR/experiments/<ID>/final/                  # imported Colab final, preserved
ASR/local/runs/<ID>/<run-id>/manifest.json   # resolved local config
ASR/local/runs/<ID>/<run-id>/checkpoints/    # trainer resume incl optimizer/scheduler/RNG/FP16-scaler
ASR/local/runs/<ID>/<run-id>/final/          # local best adapter + processor
ASR/local/runs/<ID>/<run-id>/eval/           # metrics and predictions
ASR/local/runs/<ID>/<run-id>/resources.jsonl # sampled GPU and process RSS
ASR/local/logs/                             # timestamped child process console logs
ASR/local/imports/reports/                  # copied Colab reports
ASR/local/imports/original_audio/           # copied full audio, one file/video
ASR/local/context/                         # manifests/features for later gate work
```

Normal training saves every 100 optimizer steps, or after 600 seconds at the next completed optimizer step, retaining best/latest through Trainer's save_total_limit=2. Smoke saves every step and runs 2 optimizer steps on tiny subsets with accumulation 1. Smoke weights/results are diagnostics only, not comparable to full runs. Same run-id resumes the latest completed checkpoint; do not delete optimizer or trainer files. Changed config/profile requires a new run-id. Hard termination/power loss can lose work since the last complete checkpoint. No partial checkpoint is guaranteed recoverable. Full final output is written only on successful completion.

Evaluation currently restarts from the beginning after interruption; its final predictions are saved at completion by the existing evaluator. Resource samples and console logs are incremental. Evaluation generation remains the original implementation for comparability (including its existing attention-mask warning).

No raw dataset copying, automatic pruning of user files, duplicate checkpoint exports, or automatic full training. Hugging Face retains a shared model cache; datasets may create their own Arrow cache. Do not run multiple GPU notebooks simultaneously. Existing Colab results are not overwritten. Close other GPU jobs before running.

## Transfer prerequisites

Local prepared data currently lacks VIVOS/FLEURS/ViMedCSS; prepare all four sources in notebook 00 or copy the prepared files and registry from Drive. The runner refuses a partial-source benchmark. Import the complete V1 `final` directory from Colab before evaluating V1. To continue the exact Colab optimizer state, also transfer its full checkpoints into a separate run directory and reconcile config/version; that is not automatic in these notebooks. Retraining from pretrained base is opt-in and isolated from imported V1.

The current 30 reviewed clips are DEV only. Do not move them into gate TRAIN. Resolve the history-5-second error audit and decode EOS/truncation checks before choosing context policy. DEV has already been used for exploratory tuning and public TEST baselines have already been evaluated; report that usage accurately.

## User-controlled execution

Model inference/training cells now call the task directly when the user presses Run, with no silent RUN=False switch. Choose individual task cells; V1 contains both evaluation and training workflows, so Run All executes both. Only the user starts these jobs. Environment and file-audit cells do not perform model inference. The assistant must not run further training/evaluation unless the user explicitly changes this instruction. B1 may need an online first download because its cached snapshot is incomplete; do not force HF_HUB_OFFLINE=1 when running the notebook normally. No further installation or download was performed after the setup-only instruction.

Notebook subprocess output is streamed on both carriage returns and newlines, with a 10-second quiet-period heartbeat. Training logs microbatches, optimizer progress, loss/metrics, approximate ETA and sampled RAM/VRAM; checkpoint saves print their directory. All output is also saved to local/logs. After editing helper files while a notebook kernel is open, restart that kernel and rerun its setup cell to import the new helper code.

Local Trainer explicitly saves/restores FP16 GradScaler state. Its RNG loader handles PyTorch 2.6 compatibility only for checkpoints under the same local run; never place untrusted checkpoint files there. The small validation runs are plumbing tests, not ASR quality evidence.
