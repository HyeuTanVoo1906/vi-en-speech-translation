# ASR — môi trường huấn luyện (Google Colab)

Thiết kế nghiên cứu: [ASR_Context_Streaming_Research_Blueprint_2026-10-13.md](ASR_Context_Streaming_Research_Blueprint_2026-10-13.md)

## Cấu trúc

```text
ASR/
├── colab.ipynb              # chạy trên Colab: mount Drive → baseline → train → eval
├── requirements.txt
├── asr/                     # thư viện dùng chung
│   ├── config.py            # _base.yaml + config thí nghiệm, ghi runs.jsonl
│   ├── data.py              # đọc parquet, trộn nguồn, tạo log-mel + labels
│   ├── model.py             # Whisper / PhoWhisper, LoRA
│   └── metrics.py           # chuẩn hóa text, WER/CER, CS-term recall
├── scripts/
│   ├── prepare_data.py      # raw → data/<SOURCE>/<split>.parquet (chạy 1 lần)
│   ├── train.py             # train 1 thí nghiệm, tự resume
│   └── evaluate.py          # giải mã + chấm 1 split
├── experiments/
│   ├── _base.yaml           # cấu hình mặc định chung
│   └── <ID>/                # MỖI THÍ NGHIỆM MỘT THƯ MỤC
│       ├── config.yaml      #   chỉ ghi phần khác _base.yaml
│       ├── runs.jsonl       #   mỗi lần train/eval: config, phiên bản thư viện, GPU, hash dữ liệu
│       ├── checkpoints/     #   checkpoint trong lúc train (giữ best + mới nhất)
│       ├── final/           #   model tốt nhất theo DEV WER (LoRA: chỉ adapter)
│       └── eval/<split>/    #   predictions.jsonl + metrics.json
└── data/                    # sinh bởi prepare_data.py (không commit)
```

## Quy ước đặt tên thí nghiệm

| Tiền tố | Ý nghĩa | Ví dụ |
|---|---|---|
| `B*` | Baseline, không train (`train: null`) | `B0_phowhisper_small_zeroshot`, `B1_whisper_small_zeroshot` |
| `V*` | Biến thể có train | `V1_phowhisper_small_lora` |
| `*G` | Model đã có, giải mã với guard: bỏ clip im lặng, giới hạn token theo độ dài audio (ngưỡng lấy từ TRAIN) | `V1G_phowhisper_small_lora_guard` |
| `*B` | Model đã có, giải mã beam search (`eval.num_beams: 5`) thay vì greedy | `V3B_phowhisper_small_lora_vsplit_beam5` |
| `*V` / `V3`, `V4` | Dùng ViMedCSS chia lại theo video (`data/ViMedCSS_vsplit`) — không so ViMedCSS với V1/V2 | `B0V_...`, `V3_phowhisper_small_lora_vsplit` |

Thêm thí nghiệm mới: copy một thư mục, đổi tên (`V2_...`), sửa `config.yaml`. Không sửa config của thí nghiệm đã có kết quả — tạo ID mới. Một mapping có `_replace: true` thay hẳn mapping của `_base.yaml` thay vì gộp (dùng cho danh sách nguồn mới).

**ViMedCSS chia lại theo video**: split chính thức có 81,7% clip TRAIN thuộc video cũng nằm trong DEV/TEST/HARD. `python scripts/build_v3_data.py` gộp các split, chia lại 80/10/10 theo video (seed 13), bỏ clip rỗng/im lặng, ghi `data/ViMedCSS_vsplit/` và `data/v3_train_exclude.json` (`data.train_exclude`); `data/ViMedCSS` gốc giữ nguyên.

## Quy trình

**1. Chuẩn bị dữ liệu (một lần, nên chạy trên máy local)**

```bash
cd ASR
pip install -r requirements.txt          # hoặc chỉ: pyarrow soundfile soxr numpy
python scripts/prepare_data.py --raw-dir ../dataset/ASR
```

Mọi nguồn được đưa về một schema, audio 16 kHz mono FLAC. Kết quả: `data/<SOURCE>/{train,dev,test}.parquet`, `ViMedCSS/hard.parquet` và `data/registry.json` (số dòng, số giờ, số group, sha256). Chạy lại sẽ bỏ qua các file đã có (thêm `--overwrite` để tạo lại).

| Nguồn | Split | Group (không tách qua split) |
|---|---|---|
| VIVOS | train/test chính thức; dev = ~5% speaker của train (hash cố định) | speaker |
| FLEURS vi_vn | train/dev/test chính thức | sentence id |
| VietMed | train/dev/test chính thức | bản ghi gốc |
| ViMedCSS | train/validation→dev/test/hard chính thức | video nguồn |

Không dùng: **VietMed-NER** (audio có khả năng trùng VietMed → leakage), **TED2020** (chỉ có text), **MultiMed-ST tiếng Việt** (`dataset/MT/MultiMed-ST_Vietnamese`): là VietMed chia lại kèm bản dịch — train/eval của nó trộn VietMed train và DEV (2.319 câu VietMed DEV nằm trong MultiMed-ST train), `corrected.test` = VietMed TEST. Không dùng làm dữ liệu ASR, và phía MT không train trên nó nếu còn đánh giá trên VietMed DEV/TEST.

**2. Đưa lên Drive**: upload cả thư mục `ASR/` (gồm `data/`) vào `MyDrive/DPL/ASR`.

**3. Mở `colab.ipynb` trên Colab** (Runtime → T4 GPU) và chạy lần lượt các cell.

Lệnh tương đương:

```bash
python scripts/evaluate.py B0_phowhisper_small_zeroshot --split dev --data-dir /content/data
python scripts/train.py    V1_phowhisper_small_lora              --data-dir /content/data
python scripts/evaluate.py V1_phowhisper_small_lora   --split dev --data-dir /content/data
```

## Tái lập và an toàn kết quả

- **Resume**: checkpoint ghi thẳng lên Drive; Colab ngắt kết nối thì chạy lại `train.py`, nó tiếp tục từ checkpoint mới nhất (gồm optimizer, scheduler, trạng thái RNG).
- **Seed** cố định (`seed` trong config) cho model, thứ tự dữ liệu và việc lấy mẫu theo giờ.
- **`runs.jsonl`** lưu config đã merge, phiên bản thư viện, GPU và `registry.json` của dữ liệu ở mỗi lần chạy.
- **TEST bị khóa**: `evaluate.py --split test|hard` bắt buộc `--allow-test`. Chỉ dùng sau khi đã chốt model và config trên DEV.
- **Prior exposure**: PhoWhisper đã được fine-tune trên VIVOS, nên số liệu VIVOS của các model PhoWhisper bị lạc quan. Luôn báo cáo WER theo từng nguồn (`metrics.json → by_source`).
- `cs_term_recall` (chỉ có ở ViMedCSS) là chỉ số xấp xỉ, kiểm tra thuật ngữ bằng khớp chuỗi con. Nó không phải CS-WER đầy đủ.
