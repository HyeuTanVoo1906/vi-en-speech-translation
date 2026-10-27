# PROJECT_CONTEXT — DPL302m, Đề tài 1 (Vi speech → En text, cascade VAD → ASR → MT)

Mục đích: dán vào cuộc trò chuyện Claude mới. Lập ngày 2026-10-28 từ mã, `reports/*.json`, `README.md`, `docs/`, `git log` của repo này.
Mỗi số có nguồn trong ngoặc. "chưa xác minh" = không tìm thấy bằng chứng trong repo. Phần ASR (repo khác) chỉ biết qua `docs/ASR_TO_MT_HANDOVER_SPEC_VI.md`.

## 1. Mục tiêu và ràng buộc
- Đề tài 1 (`docs/Project.txt` §5.1): ASR tiếng Việt + dịch sang tiếng Anh. Input: file âm thanh hoặc micro real-time. Output: transcript Vi + bản dịch En (có thể kèm mốc thời gian). Cần offline + real-time, âm thanh dài (VAD/cửa sổ trượt), tự chuẩn hoá sampling rate.
- Metric gợi ý (Project.txt §5.1.6): ASR WER/CER; dịch BLEU (sacreBLEU), chrF++ (COMET/BERTScore là điểm cộng); cascade BLEU trên bản dịch từ audio; RTF, độ trễ real-time, kích thước mô hình.
- Rubric (Project.txt §7): dữ liệu 15, baseline 10, đề xuất + ablation 25, đánh giá 15, demo 10, báo cáo + slide 10, code 5, vấn đáp 10 (tổng 100). Cộng: FastAPI +3, Docker + CI/CD chạy thật +5, tự build dữ liệu quy mô đáng kể +2 (tối đa +10). Điểm cá nhân = điểm nhóm × hệ số đóng góp (Git, bảng phân công, vấn đáp).
- Quy định: nhóm 2–3 người, mỗi người có commit rõ ràng, cố định seed + ghi môi trường, không dùng test để train/chọn siêu tham số, trích dẫn nguồn, ghi rõ phần dùng AI hỗ trợ.
- Hạn nộp: chưa xác minh. `Project.txt` nhảy từ mục 7 sang mục 9, không có mục 8 và không có ngày.
- "Task 2": chưa xác minh, không có trong repo. Trong `Project.txt` "Đề tài 2" là Person Tracking, không liên quan nên không dùng để suy ra Task 2.

## 2. Phân vai và tích hợp
- Phần MT + cascade eval + demo: repo này (nhánh `main`, tác giả git duy nhất `HyeuTanVo`).
- Phần ASR: repo `Dawn2310/asr-vimedcss-phowhisper` (theo lời bạn; không có mã ASR trong repo này).
- Mô hình ASR cuối "V4B", đường dẫn adapter: chưa xác minh. Handover (ngày 21/10/2026) chỉ nói đến `V1_phowhisper_small_lora` (PhoWhisper-small + LoRA r=32, alpha=64; adapter `ASR/local/runs/V1_phowhisper_small_lora/local_v1/final`; WER 18.4786% / CER 14.3647% trên DEV tự gộp 4 nguồn 5168 câu; handover §4, §6). Không có chữ "V4B" trong repo này.
- Hợp đồng JSONL bạn nêu: `id, source, split, start, end, hyp_raw, hyp_norm, text, asr_model`. Khác với:
  - `predictions.jsonl` trong handover §4: `id, source, text, cs_terms, duration, hyp`.
  - JSONL ASR công khai mà repo này tạo (`cascade/run_public_asr.py`): `id, source, split, asr_model, hyp_raw, hyp_norm, text, text_norm, en_ref, duration` (không có `start`, `end`).
  - `cascade/eval_cascade.py` đọc `id, hyp_raw, hyp_norm, en_ref` từ file ASR. File của nhóm ASR không có `en_ref` (handover §3) nên phải join thêm từ `cascade/data/fleurs_*.jsonl`. Chưa có script join.
- Trạng thái tích hợp: `pipeline/inference.py` có tham số `asr_adapter` (PEFT, merge), mặc định `None` → demo hiện chạy `vinai/PhoWhisper-small` gốc (B0), kèm MT adapter `runs/mt/M4_s42/adapter`. Chưa có báo cáo nào chạy cascade với adapter ASR của nhóm. Chưa có `en_ref` cho dữ liệu y khoa ViMedCSS của nhóm ASR nên chưa đo BLEU trên miền đó.

## 3. Cấu trúc repo (2 cấp)
```
MT/        get_phomt.py, prepare_parallel.py, probe_*.py, smoke_translate.py
  mt/      model.py (Translator: vinai | nllb, nạp LoRA), noise.py (verbalize số, NoiseModel, swap_tone)
  scripts/ build_noise.py, build_num.py, build_rand.py, train_mt.py, bootstrap_compare.py
cascade/   build_eval_sets.py, build_asr_train.py, run_public_asr.py, eval_cascade.py
common/    textnorm.py (normalize_text, unify_tone)
pipeline/  inference.py (SpeechTranslator: VAD→ASR→MT), streaming.py (StreamSession), export.py (SRT)
app/       gradio_app.py
scripts/   asr_report.py, check_env.py, simulate_stream.py, translate_file.py
tests/     test_noise.py, test_textnorm.py
reports/   JSON kết quả (cascade_*, bootstrap_*, asr_public_wer, phomt_audit)
docs/      Project.txt, ASR_TO_MT_HANDOVER_SPEC_VI.md, PROJECT_CONTEXT.md
(gitignore) MT/data, cascade/data, runs
```
Lệnh (chạy từ gốc repo, `python -m ...` vì script import `MT`/`common`; chạy `python cascade/eval_cascade.py` lỗi `No module named 'MT'`, đã thử):
- Eval set FLEURS+FLoRes+: `python -m cascade.build_eval_sets`; FLEURS train: `python -m cascade.build_asr_train`.
- PhoMT: `python MT/get_phomt.py`, rồi `python -m MT.prepare_parallel`.
- ASR công khai: `python -m cascade.run_public_asr --model B0|B1 --set fleurs_dev [--limit N]`.
- Noise/cột: `python -m MT.scripts.build_noise` (cột `vi_asr`), `build_num` (`vi_num`), `build_rand` (`vi_rand`).
- Train: `python -m MT.scripts.train_mt --exp M4_s42 --src-col vi_asr --n 100000 --seed 42 [--resume]`.
- Eval: `python -m cascade.eval_cascade --set fleurs_dev --tag M4_s42 --adapter runs/mt/M4_s42/adapter [--mt nllb] [--verbalize] [--limit N]`.
- Bootstrap: `python -m MT.scripts.bootstrap_compare --a M0 --b M4_s42 M4_s43 [--resamples 500]` (mặc định 500; số lần đã dùng khi chạy thật không lưu trong JSON → chưa xác minh).
- Báo cáo ASR: `python -m scripts.asr_report`. Demo: `python -m app.gradio_app [--port 7860] [--mt-adapter ...] [--asr-adapter ...]`.
- Không có cờ `--smoke`; smoke-test hiện là `--limit N` (eval) hoặc `--n 640 --accum 2 --exp smoke` (train, `runs/mt/smoke` có 40 bước).

## 4. Dữ liệu
- Eval/cascade: FLEURS vi_vn (`google/fleurs`) ghép câu Anh từ FLoRes+ (`openlanguagedata/flores_plus`, split devtest) theo văn bản chuẩn hoá (`cascade/build_eval_sets.py`).
  - Dev: 361 clip, 149 câu duy nhất, 1.19 h. Test: 857 clip, 347 câu duy nhất, 3.00 h. Train: 2994 clip, 1482 câu, 9.08 h, không có `en_ref` (tính từ `cascade/data/fleurs_*.jsonl`). Nhiều giọng đọc cùng một câu → bootstrap resample theo nhóm `en_ref` (`bootstrap_compare.py`).
- MT train: PhoMT (`vinai/PhoMT`, detokenization). Train 2,977,999 cặp → giữ 2,946,114 → lấy mẫu 300,000 (seed 42); dev 18,719 → giữ 18,543 → mẫu 2,000 (`reports/phomt_audit.json`). Loại: ratio 17,604, length 13,629, trùng 652 (train). Audit không có khoá `blocked_*` → không có câu nào bị loại vì trùng FLEURS/FLoRes+ (suy từ vắng khoá). Mỗi lần train dùng `--n 100000` mẫu ngẫu nhiên từ 300k theo `random_state=seed`, nên đổi seed là đổi cả tập con dữ liệu lẫn khởi tạo (`train_mt.py::load`).
- License: chưa xác minh trong repo. README PhoMT có "Copyright (c) 2021 VinAI Research" kèm đoạn từ chối bảo hành, loại giấy phép cụ thể chưa xác minh. FLEURS, FLoRes+: chưa xác minh trong repo.
- Chống rò rỉ (`MT/prepare_parallel.py`): chặn khỏi MT train mọi câu Vi/En (đã normalize) của FLEURS dev + test và toàn bộ FLoRes+ dev + devtest. Phía ASR: handover §3 báo giao ID/nhóm/văn bản chuẩn hoá giữa FLEURS train và dev/test = 0. Không dùng dữ liệu y khoa ViMedCSS/VIVOS/VietMed để lọc MT train; handover đề xuất dùng `heldout_exclusion_manifest.csv` nhưng file này không có trong `docs/` và `prepare_parallel.py` không đọc nó. Handover §3: chưa xác minh PhoMT không trùng dữ liệu ASR.
- Quy ước cột/chuẩn hoá:
  - `normalize_text` (`common/textnorm.py`, khớp `normalize_text` của nhóm ASR theo docstring): NFC → lowercase → bỏ `[^\w\s]` → gộp khoảng trắng; số giữ nguyên chữ số.
  - `vi_norm` = `normalize_text(vi)`. `vi_num` = `normalize_text(verbalize_numbers(vi))` (số → chữ).
  - `vi_asr` = verbalize + normalize, 70% câu bị nhiễu theo `NoiseModel` học từ cặp (gold, hyp B0) trên FLEURS train: WER 7.83%, 1856 luật span, tỉ lệ chèn 0.14% (`MT/data/noise_model.json`). Mức nhiễu đo trên dev là 6.76% (comment `build_rand.py`).
  - `vi_rand` = verbalize + normalize, 80% câu bị nhiễu ngẫu nhiên (RATE 0.085: xoá/đổi dấu/thay từ/xoá ký tự), đối chứng không dùng đặc trưng ASR thật.
  - `unify_tone`: đổi kiểu dấu thanh về kiểu cũ của PhoMT (hoà/thuỷ). PhoMT: 66,053 kiểu cũ / 1 kiểu mới; FLEURS dev gold 34.5% kiểu mới (`phomt_audit.json`).
  - Biến thể kết quả: `raw` (`text_raw`/`hyp_raw`) và `norm` (`text_norm`/`hyp_norm`); nguồn đầu vào `gold`, `B0` (PhoWhisper-small zero-shot), `B1` (openai/whisper-small zero-shot).
- Dữ liệu tự build / tập Offline riêng của nhóm: chưa có trong repo (chưa xác minh "Offline" nghĩa chính xác là gì).

## 5. Kết quả thí nghiệm (chỉ FLEURS **dev**; chưa chạy MT trên test)
Metric: sacreBLEU `BLEU()` và `CHRF(word_order=2)` (chrF++), beam 5; nguồn `reports/cascade_fleurs_dev_<tag>_<mt>.json`. Mô hình MT: M0 = `vinai/vinai-translate-vi2en-v2` zero-shot; baseline thay thế = `facebook/nllb-200-distilled-600M`.
Định nghĩa biến thể (theo `src-col` trong `runs/mt/*/summary.json`): LoRA trên 100k câu PhoMT, 1 epoch, 3125 bước. M1=`vi` (thô), M2=`vi_norm`, M2n=`vi_num`, M3=`vi_rand`, M4=`vi_asr` (seed 42/43/44). M0tone = M0 + `unify_tone` lúc eval. M4_s42v = M4_s42 đánh giá với `--verbalize` (tag suy ra từ tên file bootstrap + commit a44fd9a).

BLEU (gold | B0 | B1; mỗi ô raw / norm):
| Tag | gold | B0 | B1 |
|---|---|---|---|
| M0 NLLB | 34.81 / 24.77 | 21.38 / 20.21 | 16.12 / 14.50 |
| M0 VinAI | 37.41 / 32.43 | 23.55 / 25.13 | 17.65 / 17.66 |
| M0tone | 37.30 / 32.34 | 23.62 / 25.24 | 17.61 / 17.70 |
| M1_s42 | 36.88 / 33.02 | 23.72 / 25.07 | 17.70 / 17.88 |
| M2_s42 | 37.15 / 33.74 | 24.41 / 24.81 | 17.26 / 17.34 |
| M2n_s42 | 37.87 / 32.71 | 25.55 / 25.69 | 17.67 / 17.48 |
| M3_s42 | 36.68 / 32.53 | 26.05 / 26.09 | 18.62 / 18.66 |
| M4_s42 | 37.90 / 33.49 | 25.51 / 26.55 | 19.13 / 19.00 |
| M4_s43 | 36.66 / 32.68 | 26.12 / 26.33 | 18.31 / 18.71 |
| M4_s44 | 37.32 / 32.96 | 26.22 / 26.21 | 18.50 / 17.95 |
| M4_s42v | 37.44 / 33.23 | 25.51 / 26.57 | 19.12 / 19.00 |

chrF++ (cùng thứ tự ô; chỉ các dòng chính): M0 VinAI 61.64 / 58.75 | 50.60 / 51.93 | 43.88 / 43.70. M2 61.47 / 59.79 | 51.26 / 51.71 | 43.78 / 44.14. M3 61.48 / 59.12 | 52.36 / 52.90 | 45.03 / 45.40. M4_s42 61.82 / 59.38 | 52.15 / 53.02 | 45.16 / 45.47. M4_s43 61.09 / 59.17 | 52.47 / 52.91 | 44.82 / 45.19. M4_s44 61.37 / 59.38 | 52.38 / 52.93 | 45.19 / 44.90. NLLB 58.31 / 51.29 | 47.92 / 45.74 | 41.33 / 38.98. Các dòng còn lại xem JSON.

WER ASR công khai (`reports/asr_public_wer.json`, ref số giữ chữ số | ref số đọc thành chữ): dev B0 11.33% | 8.92%, B1 24.65% | 26.65%; test B0 12.32% | 9.74%, B1 22.54% | 24.90%.

Bootstrap (cặp, resample theo nhóm câu; ∆BLEU, CI 95%; `reports/bootstrap_fleurs_dev_*.json`):
- Có ý nghĩa (CI không chứa 0) so với M0, đầu vào B0 raw: M2n +2.00 [+0.74,+3.19]; M3 +2.51 [+1.22,+3.74]; M4 s42/s43/s44 +1.96 [+0.14,+3.70] / +2.57 [+1.27,+3.95] / +2.67 [+1.40,+4.06].
- B0 norm: chỉ M4_s42 +1.42 [+0.27,+2.78]; s43 [-0.11,+2.45], s44 [-0.15,+2.29] chứa 0. B1 norm: M3 +1.00 [+0.15,+1.98], M4_s42 +1.34 [+0.19,+2.45]; M4_s43, s44 chứa 0. B1 raw: chỉ M4_s42 +1.48 [+0.22,+2.65].
- Đầu vào gold (văn bản sạch): không biến thể nào hơn M0 có ý nghĩa (raw: mọi CI chứa 0; norm: M2 +1.31 [-0.55,+3.30]).
- M2n so với M2: gold norm -1.03 [-1.93,-0.13] (M2n thấp hơn); B0 raw +1.14 [+0.34,+2.11].
- M3 so với M2n: gold raw -1.20 [-2.26,-0.13] (M3 thấp hơn); B1 norm +1.19 [+0.42,+1.93].
- **M4 so với M3: cả 18 ô (3 seed × 6 ô) đều có CI chứa 0** (sát biên: `B1_norm_M4_s44` [-1.63,+0.00]; `gold_raw_M4_s42` p=0.050). Không có bằng chứng M4 hơn M3.
- M4 so với M2n: M4_s42 hơn trên B1 raw [+0.64,+2.36] và B1 norm [+0.68,+2.42]; M4_s43 hơn trên B1 norm [+0.22,+2.18]; B0 không ô nào có CI > 0 (B0 norm s42 [-0.05,+1.73]).
- M4_s42 so với M4_s42v: |∆| ≤ 0.02 trên B0/B1, gold raw -0.46 [-0.97,+0.06] → `--verbalize` lúc eval gần như không đổi điểm.
- Biến động theo seed: M4 B0 norm 26.55/26.33/26.21, B1 norm 19.00/18.71/17.95; chênh lệch theo seed cùng cỡ với một số ∆ trên.
- M1 và M0tone chưa có bootstrap: M1 gold raw 36.88 và B0 raw 23.72 so với M0 37.41/23.55, chưa xác minh ý nghĩa thống kê.

Kết luận được hỗ trợ (dev, một MT backbone, ASR = B0/B1):
1. Fine-tune LoRA bằng nguồn Vi bị nhiễu (M3, M4) hoặc có số đọc thành chữ (M2n) tăng BLEU trên đầu vào ASR B0 raw so với M0 (+2.0…+2.7), CI không chứa 0.
2. Mức tăng không thấy trên đầu vào gold; M2n/M3 còn thấp hơn M2/M2n ở vài ô gold.
3. NLLB-600M kém M0 VinAI ở mọi ô (BLEU).
4. Dùng M4_s42 làm mô hình demo (`pipeline/inference.py::DEFAULT_MT_ADAPTER`, `app/gradio_app.py --mt-adapter` mặc định). Lý do chọn chưa xác minh; seed 42 cũng là seed có điểm B0 norm và B1 cao nhất trong 3 seed trên dev.
Kết luận KHÔNG được hỗ trợ: M4 hơn M3 (nhiễu kiểu ASR hơn nhiễu ngẫu nhiên); M4 hơn M0 trên văn bản sạch; mức tăng trên tập test; mức tăng với ASR của nhóm (V1/V4B); mức tăng chủ yếu nhờ nhiễu chứ không nhờ đọc số thành chữ (M2n đã gần M3/M4 trên B0). Lưu ý: noise model của M4 học từ lỗi B0 trên FLEURS train, và B0 cũng là một nguồn đánh giá nên B0 có khả năng "khớp nhiễu" (chưa kiểm chứng).
Siêu tham số train (`train_mt.py`, `runs/mt/*/summary.json`): LoRA r=16, alpha=32, dropout 0.05, target `q,k,v,out_proj,fc1,fc2`; AdamW lr 2e-4, cosine, warmup 3%, bs 8 × accum 4, max_len 128, fp16 autocast, clip 1.0; peak VRAM 5.17 GB; thời gian 1772–4721 s (M4_s43 4721 s, M4_s44 1772 s, lý do chênh chưa xác minh). `dev_loss` cuối: M1 0.9895, M2 1.0418, M2n 1.0525, M3 1.1962, M4 1.1472/1.1269/1.1366 (s42/43/44) — không so sánh được giữa các cột nguồn khác nhau.

## 6. Pipeline và demo
- `pipeline/inference.py`: `load_audio` (mono, resample 16 kHz) → Silero VAD (`min_silence 300 ms`, `max_speech 25 s`, `pad 100 ms`) → gộp đoạn (`max_gap 0.8 s`, tối đa 25 s) → PhoWhisper-small fp16 (beam 1, `max_new_tokens 225`, batch 8 đoạn) → `normalize_text(verbalize_numbers())` → VinAI vi2en + LoRA (beam 5). Trả `segments` (start, end, vi, en) và `stats` (vad_s, asr_s, mt_s, rtf).
- `pipeline/streaming.py`: `StreamSession` luồng nền; chốt câu sau 0.6 s im lặng hoặc đủ 15 s; bản dịch tạm mỗi 0.8 s. `scripts/simulate_stream.py` phát lại file như luồng; không có log kết quả streaming lưu trong repo (chưa xác minh chất lượng/độ trễ).
- `app/gradio_app.py` (Gradio 4.44.1, 4 tab): tải file; micro ghi xong mới dịch; micro trực tiếp; văn bản → En. Xuất SRT (tiếng Anh) + JSON. Chạy ở `127.0.0.1`, không có link public.
- pytest (chạy ngày 2026-10-28, `.venv-mt`): `7 passed in 0.02s` (`tests/test_noise.py`, `tests/test_textnorm.py`). Không có test cho pipeline, streaming, app.

## 7. Môi trường
- Python 3.10.11 (`.venv-mt`), GPU NVIDIA GeForce RTX 4050 Laptop 6141 MiB (nvidia-smi). Ghim (`requirements.txt`, `requirements-lock.txt`): torch 2.6.0+cu124, torchaudio 2.6.0+cu124, transformers 4.46.3, peft 0.13.2, accelerate 1.1.1, sentencepiece 0.2.0, datasets 3.2.0, numpy 1.26.4, sacrebleu 2.6.0, jiwer 4.0.0, gradio 4.44.1, silero-vad 5.1.2, fastapi 0.115.6 (chưa có mã API).
- Kích hoạt (PowerShell, từ gốc repo): `.\.venv-mt\Scripts\Activate.ps1`. Cài: `py -3.10 -m venv .venv-mt`, cài torch cu124 trước rồi `pip install -r requirements.txt` (README + dòng đầu `requirements.txt`). Kiểm tra: `python scripts/check_env.py`.
- Cảnh báo: handover ghi nhóm ASR dùng `.venv-asr` (Python 3.10.8, cùng torch/transformers/peft); chưa kiểm thử chạy chung ASR của nhóm + VinAI Translate trên một GPU (handover §6).

## 8. Git
- Nhánh `main`, 22 commit (14 ngày 2026-10-23, 8 ngày 2026-10-24), một tác giả `HyeuTanVo`. Remote: **chưa có** (`git remote -v` rỗng). Working tree sạch khi bắt đầu. Commit mới nhất `66f890d` "Streaming: background worker...".
- `.gitignore` loại: `.venv*/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `data/`, `runs/`, `local/`, `checkpoints/`, `*.safetensors`, `*.bin`, `*.pt`, `*.parquet`, `*.wav/mp3/flac`, `.env`. Hệ quả: adapter MT (`runs/mt/*/adapter`, ~34 MB) và `runs/` (1.1 GB), `MT/data` (1.5 GB), `cascade/data` (1.9 GB, gồm `.wav`) không có trên Git. Cần link/script tải để nộp (Project.txt §4 mục 5).

## 9. Đã làm / chưa làm
Đã làm (có bằng chứng): eval set FLEURS dev/test + En ref; ASR công khai B0/B1 (dev/test/train) + WER; baseline M0 VinAI và NLLB; PhoMT làm sạch + chặn rò rỉ; LoRA M1–M4 (3 seed M4) + đối chứng M2n/M3/M0tone; bootstrap theo nhóm câu; pipeline VAD→ASR→MT, streaming, Gradio 4 tab, SRT/JSON; 7 test; ghim phiên bản.
Chưa làm / chưa thấy trong repo:
- Dữ liệu tự build / tập Offline riêng (+2 điểm) và EDA (phân bố độ dài, giới tính, nhiễu...; chỉ có thống kê gộp trong `phomt_audit.json`).
- Bảng Paper vs Thực nghiệm; đồ thị loss (log TensorBoard + `train.log` có trong `runs/mt/*`, chưa xuất hình); failure cases / phân tích lỗi (chưa có file).
- Benchmark RTF/VRAM/kích thước có hệ thống: `stats.rtf` và RTF/VRAM của `run_public_asr.py` chỉ in ra, không lưu; kích thước mô hình MT chưa ghi (adapter 34 MB là số `du`).
- Cascade MT trên FLEURS test (chỉ ASR B0/B1 đã chạy trên test, `runs/asr_public/fleurs_test_*.jsonl`, `asr_public_wer.json`); tích hợp ASR của nhóm (V1/V4B) vào cascade và eval; COMET/BERTScore; so sánh end-to-end speech translation.
- FastAPI (+3), Docker, CI/CD, deploy (+5); báo cáo PDF 15–25 trang; slide 12–20; video demo 2–5 phút; bảng phân công; link tải dữ liệu/checkpoint; remote Git.
- `README.md` mới 3 dòng, thiếu hướng dẫn train/eval/suy luận mà Project.txt §4 yêu cầu.

## 10. Quy ước làm việc với bạn (theo yêu cầu của bạn)
- Trả lời bằng tiếng Việt. Bạn tự chạy mọi lệnh train/eval trong PowerShell; mỗi lần một bước và chạy thử bản nhỏ trước (smoke). Không hỏi HF token trong chat. TEST chỉ chạy đúng một lần ở cuối. Ghi gì cũng phải kèm nguồn xác minh; không suy diễn.

## 11. Điểm không chắc / mâu thuẫn cần xử lý
1. "V4B" (lời bạn) khác "V1" (handover 21/10); chưa có đường dẫn adapter V4B.
2. Hợp đồng JSONL bạn nêu (có `start`, `end`, `asr_model`) khác handover (`hyp`, `cs_terms`, `duration`) và khác định dạng `eval_cascade.py` cần (`en_ref`).
3. Handover coi là chưa xác minh việc PhoMT không trùng dữ liệu ASR; `prepare_parallel.py` chưa dùng `heldout_exclusion_manifest.csv`.
4. Gate trong handover §7 cho thấy sửa lỗi ASR có thể gây hại (KEEP 8.9035%, gate 9.1559% WER trên DEV nội bộ); chưa liên quan trực tiếp MT nhưng cần giữ nguyên tắc so với baseline.
5. Chạy test ASR (B0/B1) đã diễn ra; nếu quy tắc "TEST một lần" tính cả ASR thì đã vi phạm một phần. Cần bạn xác nhận phạm vi quy tắc.
6. `README.md` nói chung chung, không khớp mức độ đầy đủ của `docs/` và mã; `requirements.txt` có `fastapi` nhưng chưa có API.
7. Thời gian train M4 thay đổi lớn giữa seed (1772–4721 s); `dev_loss` không so sánh được giữa các cột nguồn.
8. Ý nghĩa "Offline" trong danh sách việc chưa làm và "Task 2": chưa xác minh.
