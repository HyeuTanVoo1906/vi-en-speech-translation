# Context gate V1 — baseline có thể huấn luyện

Mở `ASR/notebooks/local/07_context_gate.ipynb` hoặc chạy CLI từ thư mục dự án.

## Pipeline

1. `prepare`: lấy timestamp đã kiểm tra waveform, nối transcript TRAIN theo ID, kiểm tra file audio và loại video xuất hiện trong DEV/TEST/HARD chính thức. Chia video theo SHA256 seed 13: 70% TRAIN, 15% CAL, 15% DEV nội bộ. Không dùng transcript của các tập chính thức để lọc.
2. `generate`: giữ nguyên checkpoint V1; nhận dạng audio history 20 giây ngay trước clip, nhận dạng clip không prompt (KEEP), rồi với prompt history tự sinh (ứng viên REPLACE). Chấm cả hai ứng viên bằng teacher forcing trên cùng audio hiện tại, cùng prefix ngôn ngữ/task và không có history prompt. Sinh feature trước khi dùng reference để tạo nhãn benefit/harm theo số lỗi từ.
3. `train`: fit hai logistic regression trên TRAIN, một dự đoán cải thiện và một dự đoán gây hại. StandardScaler chỉ fit TRAIN. Chọn ngưỡng benefit/harm trên CAL để giảm tổng lỗi từ với tỷ lệ câu bị hại tối đa 1% tổng câu CAL. Hòa thì ưu tiên ít harm và ít sửa. Nếu không tốt hơn KEEP trên CAL, xuất chính sách KEEP_ONLY. Nếu TRAIN không có ứng viên cải thiện, dừng với thông báo rõ ràng.
4. `evaluate`: một lần đánh giá DEV nội bộ với gate đã cố định; báo WER KEEP, always-context, oracle hai ứng viên, gate, coverage, số câu giúp/hại và Brier score. Xuất quyết định từng câu.

Gate là baseline **Track B native history prompting**, không phải toàn bộ kiến trúc P1 trong blueprint. Feature overlap history là thống kê từ vựng, không phải semantic encoder. Có hai ứng viên toàn câu, chưa có N-best/span editor, noisy-history controls, closed-loop memory hay deadline streaming. Xác suất logistic chưa có calibrator riêng; CAL hiện chọn ngưỡng vận hành. Harm budget là số đo thực nghiệm, không phải bảo đảm thống kê.

## Dữ liệu đã chuẩn bị

Manifest 3.386 clip đủ history có 1.710 clip thuộc video xuất hiện trong TEST/HARD chính thức. Chúng được bỏ khỏi dữ liệu gate, không sửa/xóa dữ liệu gốc. Còn 1.676 clip / 731 video:

| Tập nội bộ | Clip | Video |
|---|---:|---:|
| TRAIN | 1.133 | 494 |
| CAL | 281 | 120 |
| DEV | 262 | 117 |

V1 đã được fine-tune trên dữ liệu TRAIN gốc, nên các tập nội bộ này **không phải dữ liệu chưa từng được backbone thấy**. Kết quả chỉ phục vụ phát triển gate; cần official DEV có alignment hoặc out-of-fold backbone và đánh giá closed-loop trước khi kết luận chất lượng triển khai. Việc lọc lần này không khắc phục hồi tố overlap của huấn luyện V1 trước đó.

Người dùng đã xác nhận nghe kiểm tra. Pipeline không tự sửa trường `not_listened` trong báo cáo alignment cũ và chỉ dùng manifest waveform_verified có sẵn; 21 clip review chưa được thêm nếu chưa có quyết định dạng file.

## Chạy đầy đủ

```powershell
.\.venv-asr\Scripts\python.exe -u ASR/scripts/context_gate_pipeline.py run --offline
```

`--offline` dùng base model đã có cache. Checkpoint mặc định:
`ASR/local/runs/V1_phowhisper_small_lora/local_v1/final`.

Kết quả ở `ASR/local/context/gate_v1`:

- `segments.jsonl`, `prepared.json`: dữ liệu và chia tập.
- `generation.json`: hash input, code, adapter, phiên bản và cấu hình.
- `pairs.jsonl`: ứng viên, history tự sinh, feature và nhãn; ghi ngay sau mỗi clip.
- `gate.json`: hai mô hình tuyến tính, scaler, ngưỡng, hash dữ liệu; không dùng pickle.
- `calibration_curve.json`: các điểm gain/harm/coverage trên CAL.
- `evaluation.json`, `dev_decisions.jsonl`: đánh giá và output gate.

Chạy lại cùng lệnh tiếp tục các clip còn thiếu. Input/code/checkpoint thay đổi thì phải dùng `--run-dir` mới; không trộn nhãn giữa các cấu hình. Nếu tiến trình bị kill giữa lúc ghi một dòng JSON, trình đọc sẽ báo lỗi thay vì lặng lẽ bỏ dữ liệu; giữ bản sao và kiểm tra dòng cuối trước khi tiếp tục. Sau khi `gate.json` đã đóng băng, không thêm cặp dữ liệu vào run đó.

Giới hạn GPU và pacing lấy từ `local/profile.json` (4,5 GB tổng VRAM mục tiêu, duty cycle 0,7, batch 1). Dùng chung `local/gpu.lock` với các notebook cũ. Đây là kiểm tra theo bước và giới hạn allocator, không phải quota GPU cứng. Không chạy đồng thời B0/B1/V1 và sinh dữ liệu gate.

## Chạy từng bước

```powershell
.\.venv-asr\Scripts\python.exe ASR/scripts/context_gate_pipeline.py prepare
.\.venv-asr\Scripts\python.exe -u ASR/scripts/context_gate_pipeline.py generate --offline
.\.venv-asr\Scripts\python.exe ASR/scripts/context_gate_pipeline.py train
.\.venv-asr\Scripts\python.exe ASR/scripts/context_gate_pipeline.py evaluate
```

Pilot riêng: thêm `--run-dir ASR/local/context/gate_pilot` ở mọi bước và `--limit 30` ở generate (30 clip mỗi partition). Train pilot cần `--allow-partial`; không coi kết quả pilot là chất lượng cuối. Muốn train đầy đủ, dùng run khác. History 5 giây: thêm `--history-seconds 5` ở prepare/run với run-dir riêng.

Kiểm thử CPU:

```powershell
.\.venv-asr\Scripts\python.exe -m unittest discover -s ASR/scripts -p test_context_gate.py
```

`asr.context_gate.choose(model, row)` là hàm suy luận độc lập: chỉ cần hai hypothesis, feature, cờ truncation; không cần reference. EOS/max-token được ghi riêng; context hoặc history bị cắt thì gate giữ nguyên KEEP.
