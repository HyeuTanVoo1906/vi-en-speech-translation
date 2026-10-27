# ASR-to-MT Handover Spec — 21/10/2026

Báo cáo dựa trên code, config, logs, checkpoint và metadata trong workspace. Yêu cầu ghi 8 mục nhưng chỉ liệt kê 7; mục 8 bổ sung artifact và giới hạn kiểm chứng. Đường dẫn tính từ workspace root.

## 1. Kiến trúc ASR mới và ablation

Backbone thực chạy vẫn là **vinai/PhoWhisper-small + LoRA**, chưa có Dual Encoder, CTC hybrid hay LID head. `ASR/asr/model.py::load_model` dùng WhisperForConditionalGeneration, WhisperProcessor và PEFT. `ASR/experiments/V1_phowhisper_small_lora/config.yaml`: r=32, alpha=64, dropout=.05; target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2.

Phần bổ sung đã triển khai là **history prompting và gate KEEP/REPLACE**, giữ frozen V1. `ASR/scripts/context_gate_pipeline.py::generate` giải mã history 20 giây trước clip, clip không prompt (base), clip có history prompt (context); giới hạn nội dung prompt 128 token. Chấm hai ứng viên bằng teacher forcing trên audio hiện tại, cùng prefix và không history. `fit_head/train` fit hai logistic head benefit/harm; `ASR/asr/context_gate.py::choose` quyết định theo ngưỡng. Đây là Track B diagnostic, chưa phải toàn bộ P1: chưa có N-best/span editor, closed-loop memory hay deadline streaming (`ASR/local/CONTEXT_GATE.md`).

`load_model`: language=vi, task=transcribe; forced_decoder_ids=None ở model.config và generation_config. evaluate/gate tiếp tục truyền vi/transcribe. Không override suppress_tokens trong code; không có snapshot giá trị hiệu lực để liệt kê. Không chèn thẻ EN theo span. Cấu hình hướng tới chép lời, **không bảo đảm model không dịch ngầm**. Nhãn và LoRA hỗn hợp là cơ chế học code-switching hiện có.

Ablation thực có config: B0=PhoWhisper-small zero-shot; B1=openai/whisper-small zero-shot; V1=PhoWhisper-small LoRA bốn nguồn (`ASR/experiments/*/config.yaml`). Chưa có config B2/B3a nên không tự định nghĩa lại thành kết quả đã triển khai. B0/B1/B2 trong `MT/MT_Error_Aware_Research_Blueprint_2026-10-14.md` dòng 529–531 là pretrained MT/clean-adapted MT/random-noise MT, khác namespace ASR. Gate có run riêng gate_v1.

**V1 đã hoàn tất**, không còn căn cứ ghi “đang train”: global_step=2724/max_steps=2724; epoch=2.9994838443 (~3); best checkpoint=2600, best DEV-subset WER=.1339108584. Nguồn: `ASR/local/runs/V1_phowhisper_small_lora/local_v1/checkpoints/checkpoint-2724/trainer_state.json`; `ASR/scripts/local_train.py::run_training`; log `ASR/local/logs/20261018_170854_868263.log` có train_runtime và 2724/2724. Có final adapter và full DEV predictions. Các run validation/smoke không phải lần huấn luyện chính này.

## 2. Output contract và chuẩn hóa nhãn

`ASR/asr/data.py::Featurizer.__call__` gọi `ASR/asr/metrics.py::normalize_text` khi normalize_text=true (manifest thực chạy bật true), rồi tokenizer(texts). Quy tắc: **NFC → lowercase → regex [^\w\s] thành khoảng trắng → gộp whitespace**. Giữ dấu tiếng Việt, chữ số, underscore; xóa dấu câu; AI/API/GPU thành ai/api/gpu. Không NFKC, không đổi số thành chữ. Ký hiệu phần trăm/dấu thập phân bị thay bằng khoảng trắng. `prepare_data.py::vivos` lower từ trước; `row` strip; fleurs dùng raw cols[2].

`Collator.__call__` pad labels bằng -100 và bỏ decoder-start nếu tất cả hàng có token đó; không sửa từ. Từ tiếng Anh/thuật ngữ giữ chính tả transcript sau normalization, không có hàm phiên âm hoặc thẻ EN cho từng span. Corpus vẫn có biến thể chính tả.

**Target không phải bảo đảm đầu ra 100%.** `ASR/scripts/evaluate.py::main` chỉ batch_decode(skip_special_tokens=True) rồi strip để ghi hyp. Normalization prediction chỉ diễn ra khi scoring. MT cần lưu raw hyp và normalized hyp riêng; raw prediction có thể khác casing/punctuation, sai từ hoặc thiếu nội dung.

Tám mẫu thực kèm đầy đủ Raw → Normalized Target → Prediction nằm ở phần cuối báo cáo và `samples.jsonl`. Raw ở đây là text trong manifest đã prepare, không phải transcript trước mọi xử lý. Nguồn: `ASR/local/runs/V1_phowhisper_small_lora/dev_local_v1/eval/dev/predictions.jsonl`; normalized target tính lại đúng normalize_text mà Featurizer sử dụng.

## 3. Dữ liệu, splits và leakage

Train/eval thực dùng VIVOS, FLEURS vi_vn, VietMed, ViMedCSS. Không Common Voice trong config. `ASR/scripts/prepare_data.py::SOURCES` loại VietMed-NER (nguy cơ trùng VietMed), TED2020 (text-only). Có MedEV_raw, VietMix_raw, MultiMed-ST_Vietnamese dưới dataset/MT nhưng không được loader ASR sử dụng.

Manifest: `ASR/data/<SOURCE>/{train,dev,test}.parquet`, thêm ViMedCSS/hard.parquet. `prepare_data.py::SCHEMA`: id:string, source:string, split:string, group_id:string, speaker_id:string, text:string, cs_terms:string (phân cách ;), duration:float32, audio:binary FLAC. Không có en_ref. `row/to_flac` resample 16kHz và mean channel thành mono. `asr/data.py::load_split` lọc 0<duration<=30s; không chunk câu dài.

Số hàng trước lọc duration, từ `ASR/data/registry.json`:

| Nguồn | Train | Dev | Test | Hard |
|---|---:|---:|---:|---:|
| VIVOS | 11476 | 184 | 760 | — |
| FLEURS | 2994 | 361 | 857 | — |
| VietMed | 2773 | 2912 | 3437 | — |
| ViMedCSS | 11832 | 1714 | 1614 | 658 |

`prepare_data.py::vivos` chia dev theo hash speaker; fleurs giữ official split/group sentence; vietmed giữ official/group recording; vimedcss giữ official/group video URL. Không có bước tự sửa overlap official splits.

Audit trực tiếp metadata parquet cho kết quả dưới đây. Mỗi số là giao tập giá trị duy nhất không rỗng giữa TRAIN và split tương ứng; text áp normalize_text.

| Train so với | ID trùng | Group trùng | Text chuẩn hóa trùng |
|---|---:|---:|---:|
| ViMedCSS DEV | 1 | 909 | 48 |
| ViMedCSS TEST | 2 | 898 | 44 |
| ViMedCSS HARD | 0 | 248 | 1 |
| VIVOS DEV | 0 | 0 | 2 |
| VIVOS TEST | 0 | 0 | 178 |
| FLEURS DEV/TEST | 0 | 0 | 0 |
| VietMed DEV/TEST | 0 | 0 | 0 |

Nguồn kết quả: `split_overlap_audit.json` cùng thư mục báo cáo, tính từ ASR/data parquet. Speaker giao tập bằng 0, nhưng FLEURS/ViMedCSS speaker_id rỗng nên không suy ra speaker-disjoint. Đây là metadata trước lọc duration, chưa phải audio/cross-corpus fingerprint audit. README nói group không tách qua split nhưng ViMedCSS thực tế không thỏa. Benchmark này chưa chứng minh generalization trên video độc lập.

Không có en_ref trong manifest ASR và không thấy PhoMT trong loader. **Chưa xác minh không trùng PhoMT**: thiếu corpus/audit đối chiếu; tên nguồn và schema không chứng minh câu chưa từng có trong PhoMT.

Đã xuất `heldout_exclusion_manifest.csv`, 12.497 hàng DEV/TEST/HARD nguyên manifest, gồm ID/source/split/group/speaker/text/duration/normalized_text. MT dùng ID có namespace làm utterance key; loại khỏi MT train mọi heldout ID, câu VI chuẩn hóa, source+group và source+speaker khi có. Corpus MT không có ID cần đối chiếu nội dung và provenance, thêm near-duplicate audit. File này không tự làm sạch train hiện hữu. Muốn test video độc lập phải chia lại ViMedCSS theo video và train lại ASR; chỉ lọc MT không sửa exposure trước đó của ASR.

Gate `context_gate_pipeline.py::prepare` loại video official DEV/TEST/HARD: còn 1676 clip/731 video; TRAIN=1133, CAL=281, DEV=262. Backbone V1 từng train official TRAIN nên internal DEV gate không độc lập với backbone (`ASR/local/CONTEXT_GATE.md`, `gate_v1/prepared.json`).

## 4. Lỗi ASR và prediction dump cho error-aware MT

Full DEV V1: WER=18.4786%, CER=14.3647%, substitutions=10344, deletions=7626, insertions=5675. Riêng ViMedCSS WER=16.0974%, term recall=71.2631% trên 1813 terms. Nguồn `ASR/local/runs/V1_phowhisper_small_lora/dev_local_v1/eval/dev/metrics.json`; `metrics.py::compute_metrics` tính substring recall, không CS-WER.

Mẫu thực: hormone→hocmon; Estradiol→ester dion; nang trứng→tế bào; bỏ “Thì ngoài ra”, bỏ cuối “có thể dẫn đến”; g→gram. Có thuật ngữ giữ đúng: estrogen, methionin, hayflick, basedow. Chưa có thống kê riêng tại biên Việt-Anh/mất phủ định để xếp hạng “lỗi hay gặp nhất”; không suy diễn file→phai/code→cốt khi chưa thấy prediction.

`ASR/scripts/evaluate.py::main` đã xuất predictions.jsonl: id, source, text (gold), cs_terms, duration, hyp. Thiếu group/speaker/en_ref; join manifest theo source+split+id, kiểm tra uniqueness. Dump DEV có sẵn tại đường dẫn trên; không đưa DEV/TEST vào MT train nếu vẫn dùng làm evaluation. Dump train từ backbone đã thấy câu đó có bias in-sample; noisy training nghiêm túc cần out-of-fold backbone hoặc dữ liệu riêng.

Lệnh từ workspace root, chưa chạy lại inference:

```powershell
.\.venv-asr\Scripts\python.exe ASR/scripts/local_run.py eval V1_phowhisper_small_lora --run-id mt_handover_dev --split dev --checkpoint ASR/local/runs/V1_phowhisper_small_lora/local_v1/final
```

Output: `ASR/local/runs/V1_phowhisper_small_lora/mt_handover_dev/eval/dev/{predictions.jsonl,metrics.json}`. Wrapper dùng resource profile. Wrapper eval chỉ hỗ trợ dev/test/hard. Generic evaluator hỗ trợ train:

```powershell
Set-Location ASR
..\.venv-asr\Scripts\python.exe scripts/evaluate.py V1_phowhisper_small_lora --split train --checkpoint local/runs/V1_phowhisper_small_lora/local_v1/final
```

Output: `ASR/experiments/V1_phowhisper_small_lora/eval/train/`. Generic evaluator không áp resource profile, batch mặc định 16; cần experiment config riêng phù hợp GPU, không sửa config đã benchmark. Test/hard cần --allow-test sau khi khóa cấu hình.

Gate đã có `ASR/local/context/gate_v1/pairs.jsonl`: base/context/history/reference/features/truncation/partition; `dev_decisions.jsonl` chứa quyết định. Nguồn tạo: `context_gate_pipeline.py::generate/evaluate`; không có en_ref và không coi internal partitions là heldout ASR độc lập.

## 5. Inference API tích hợp

Chữ ký thực:

```python
# ASR/asr/model.py
load_model(cfg: dict, for_training: bool = False, checkpoint: Path | None = None)
# trả (model, processor); checkpoint LoRA được merge_and_unload khi inference

# ASR/asr/data.py
decode_audio(flac_bytes: bytes) -> np.ndarray
to_device_features(processor, audios: list[bytes], device: str, dtype: torch.dtype) -> torch.Tensor
```

decode_audio đọc float32 và assert 16kHz, không tự resample. to_device_features nhận list FLAC bytes; đường dẫn/bytes được convert mono 16kHz tại `prepare_data.py::to_flac`. model.generate nhận log-mel tensor; processor.batch_decode trả List[str]; evaluator đóng gói JSON. Chưa có public transcribe(filepath) hoàn chỉnh hay timestamps/confidence/language_tags trong dump baseline.

Hàm decode(audio,prompt=None) lồng trong `context_gate_pipeline.py::generate` nhận waveform numpy, trả (text, mean_log_token_score, truncated). Mean log score không phải calibrated probability. `asr.context_gate.choose(model,row)` trả (KEEP/REPLACE,p_benefit,p_harm); caller chọn base/context.

Chưa có VAD/chunker trên tuyến evaluate. max_duration=30 là lọc, không chia chunks. Gate đọc start/end theo alignment manifest và history 20s trước start; chưa hỗ trợ audio dài tùy ý hoặc closed-loop streaming.

## 6. VRAM, RTF và môi trường

GPU ghi trong `ASR/local/runs/V1_phowhisper_small_lora/local_v1/runs.jsonl`: NVIDIA GeForce RTX 4050 Laptop GPU. `evaluate.py::main`: CUDA fp16, CPU fp32; không int8/bf16 inference trong tuyến này. Beam=1, max_new_tokens=225; local batch thực=1, khác base batch=16.

Telemetry `ASR/local/runs/V1_phowhisper_small_lora/dev_local_v1/resources.jsonl`: max sampled total GPU used=1.78520064 GB; allocator reserved/peak=0.633339904 GB (thập phân). Đây là số log quan sát, không đo riêng weights hay bảo đảm peak thực toàn tiến trình. `local_run.py::sample` định nghĩa telemetry. Profile 4.5GB là ngân sách tổng, không phải VRAM model. Chưa có benchmark đồng thời VinAI Translate để bảo đảm tránh OOM.

| Full DEV | WER | RTF decode-only |
|---|---:|---:|
| B0 PhoWhisper zero-shot | 24.7267% | .2081963 |
| B1 Whisper zero-shot | 58.7516% | .1808143 |
| V1 PhoWhisper LoRA | 18.4786% | .1813866 |

Nguồn: `ASR/local/runs/<exp>/dev_local_v1/eval/dev/metrics.json`. `evaluate.py::main` tính thời gian generate/tổng duration; 5168 câu, 9.83113 audio giờ. V1 suy ra khoảng 1.242 giây generate/câu từ tổng, không p50/p95 hay end-to-end. Không gồm load/features/MT; generic timer không synchronize CUDA rõ ràng. Gate có ba decode, acoustic scoring và pacing riêng, chưa có RTF so sánh tương đương.

Đọc trực tiếp `.venv-asr` bằng importlib.metadata, lưu `environment.json`: Python 3.10.8; torch=2.6.0+cu124; torchaudio=2.6.0+cu124; transformers=4.46.3; peft=0.13.2; accelerate=1.1.1; sentencepiece=0.2.0; ctranslate2/faster-whisper không cài. `ASR/requirements.txt` là requirements được ghim, không đại diện đầy đủ môi trường. runs.jsonl/gate_v1/generation.json xác nhận torch/transformers của run. Chưa kiểm thử runtime kết hợp VinAI Translate trong đợt audit.

## 7. Ba yêu cầu kỹ thuật cho MT

1. Giữ raw hyp và normalized hyp riêng; không tự phiên âm/dịch lại từ Anh đã đúng. Khi cần khớp train distribution, dùng đúng `metrics.py::normalize_text`. Casing và dấu câu không thể khôi phục chắc chắn từ normalized label. Kiểm tra riêng số/đơn vị (mẫu 450 g→450 gram), không coi strip punctuation là bộ xử lý số an toàn.
2. Học correction từ cặp thực, kèm ràng buộc giữ từ đúng: hormone→hocmon và Estradiol→ester dion là ví dụ noisy/gold; estrogen/hayflick có thể đã đúng. Không thay thế đồng âm toàn cục. Dùng cs_terms/gold để nghiên cứu/chấm điểm, không cấp annotation heldout cho inference nếu deployment không có nguồn tương ứng. `context_gate_pipeline.py::generate` tạo candidates/features trước khi truy cập reference là quy tắc cần giữ.
3. Giữ base làm đối chứng và đo harm trước khi dùng correction/context. `ASR/local/context/gate_v1/evaluation.json`: internal DEV KEEP WER=8.9035%, always-context=12.9277%, gate=9.1559%, sửa 2/262 và cả hai gây hại; oracle pair=7.4313%. CAL cải thiện nhưng DEV chưa xác nhận. Chưa có bằng chứng gate tốt hơn base làm đầu vào MT mặc định. Đo deletion/substitution tiếng Việt và mất phủ định riêng; term recall không đủ. `choose` có abstain khi context/history truncation nhưng không bảo đảm không gây hại.

## 8. Artifact và giới hạn bàn giao

- `ASR_TO_MT_HANDOVER_SPEC_VI.md`: báo cáo này.
- `samples.jsonl`: 8 mẫu raw/normalized target/prediction có ID.
- `heldout_exclusion_manifest.csv`: 12.497 hàng DEV/TEST/HARD giữ ngoài MT train; giữ cả hàng dài/empty để loại trừ bảo thủ.
- `split_overlap_audit.json`: giao ID/group/speaker/text chuẩn hóa giữa train và từng heldout split trong mỗi nguồn.
- `environment.json`: phiên bản package hiện có trong .venv-asr.

Chỉ tạo artifact báo cáo; không sửa config/checkpoint/dataset gốc, không chạy TEST/HARD inference mới. Chưa audit audio fingerprint, near duplicates cross-corpus, PhoMT, English reference hoặc ASR+MT đồng thời. Trước khi chốt kết quả nghiên cứu cần khắc phục split video ViMedCSS, khóa checkpoint/split/hash và đánh giá trên heldout độc lập.

### Samples: Raw -> Normalized Target -> ASR Prediction

**vimedcss:Med_CS-100-17**

- Raw: Thì ngoài ra, ở cho chúng ta thấy là cái hàm lượng dinh dưỡng so với Alpha lipid ở trên một lon có cùng trọng lượng là 450 g và sữa Colos IgGOLD cũng là 450 g.
- Normalized Target: thì ngoài ra ở cho chúng ta thấy là cái hàm lượng dinh dưỡng so với alpha lipid ở trên một lon có cùng trọng lượng là 450 g và sữa colos iggold cũng là 450 g
- ASR Prediction: ở cho chúng ta thấy là cái hàm lượng dinh dưỡng so với alpha lipid ở trên một non có cùng trọng lượng là 450 gram và sữa colos iggold cũng là 450 gram

**vimedcss:Med_CS-1000-9**

- Raw: khi ngăn methionin sẽ giúp cho cắt đoạn sự phát triển của các tế bào ung thư.
- Normalized Target: khi ngăn methionin sẽ giúp cho cắt đoạn sự phát triển của các tế bào ung thư
- ASR Prediction: khi ngăn methionin sẽ giúp cho các đoạn sự phát triển của các tế bào ung thư

**vimedcss:Med_CS-1006-14**

- Raw: Tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng, hỗ trợ kích thích sự phát triển của nang trứng.
- Normalized Target: tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng hỗ trợ kích thích sự phát triển của nang trứng
- ASR Prediction: tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng hỗ trợ kích thích sự phát triển của tế bào

**vimedcss:Med_CS-1006-35**

- Raw: Dễ dàng mệt mỏi, tâm trạng lâng lâng, thường xuyên cảm thấy chán nản và lo lắng, da khô, ngược lại, nếu hàm lượng estrogen ở nam giới quá cao có thể dẫn đến.
- Normalized Target: dễ dàng mệt mỏi tâm trạng lâng lâng thường xuyên cảm thấy chán nản và lo lắng da khô ngược lại nếu hàm lượng estrogen ở nam giới quá cao có thể dẫn đến
- ASR Prediction: dàng mệt mỏi tâm trạng lâng lâng thường xuyên cảm thấy chán nản và lo lắng da khô ngược lại nếu hàm lượng estrogen ở nam giới quá cao

**vimedcss:Med_CS-1006-9**

- Raw: Thông thường, hormone này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp. Estradiol E2, các chuyên gia đánh giá Estradiol là loại estrogen mạnh nhất.
- Normalized Target: thông thường hormone này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp estradiol e2 các chuyên gia đánh giá estradiol là loại estrogen mạnh nhất
- ASR Prediction: thông thường hocmon này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp ester dion e2 các chuyên gia đánh giá ester dion là loại estrogen mạnh nhất

**vimedcss:Med_CS-1008-4**

- Raw: Làm việc trong quá trình nuôi cấy tế bào, Hayflick nhận thấy có điều gì đó khác thường. Một nhóm tế bào già hơn ngừng phân chia và ông không hiểu tại sao điều đó lại xảy ra.
- Normalized Target: làm việc trong quá trình nuôi cấy tế bào hayflick nhận thấy có điều gì đó khác thường một nhóm tế bào già hơn ngừng phân chia và ông không hiểu tại sao điều đó lại xảy ra
- ASR Prediction: làm việc trong quá trình nuôi cấy tế bào hayflick nhận thấy có điều gì đó khác thường một nhóm tế bào già hơn ngừng phân chia và ông không hiểu tại sao điều đó

**vimedcss:Med_CS-1011-23**

- Raw: Hiện tại, hội chứng siêu Nam chưa có phương pháp điều trị dứt điểm, nhưng vẫn có những hướng điều trị khả quan như thay thế testosterone, điều trị vô sinh, phẫu thuật để giảm các triệu chứng biểu hiện bệnh.
- Normalized Target: hiện tại hội chứng siêu nam chưa có phương pháp điều trị dứt điểm nhưng vẫn có những hướng điều trị khả quan như thay thế testosterone điều trị vô sinh phẫu thuật để giảm các triệu chứng biểu hiện bệnh
- ASR Prediction: hiện tại hội chứng sưu nam chưa có phương pháp điều trị dứt điểm nhưng vẫn có những hướng điều trị khả quan như thay thế testosterone điều trị vô sinh phẫu thuật để giảm các triệu chứng

**vimedcss:Med_CS-1016-2**

- Raw: Đối với bệnh cường giáp đó thì chúng ta có hai cái nhóm bệnh chính, thứ nhất là bệnh Basedow, thứ hai là bú giáp nhân độc hay là đơn nhân hoặc là đa nhân độc.
- Normalized Target: đối với bệnh cường giáp đó thì chúng ta có hai cái nhóm bệnh chính thứ nhất là bệnh basedow thứ hai là bú giáp nhân độc hay là đơn nhân hoặc là đa nhân độc
- ASR Prediction: đối với bệnh cường giáp đó thì chúng ta có hai cái nhóm bệnh chính thứ nhất là bệnh basedow thứ hai là bú giáp nhân độc hay là đơn nhân hoặc là đang nhân độc

