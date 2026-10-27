# Notebook local

**Context gate mới:** `07_context_gate.ipynb` chuẩn bị dữ liệu theo video → sinh cặp KEEP/context bằng V1 → train gate logistic → chọn ngưỡng CAL → đánh giá DEV nội bộ. Xem `ASR/local/CONTEXT_GATE.md`. Notebook này dùng manifest alignment đã xác minh; không cần chạy lại B0/B1/V1. Nếu pipeline đang chạy nền, dùng cell theo dõi thay vì khởi chạy thêm.

Chạy lần lượt, mỗi lần chỉ một tác vụ GPU:

1. `00_setup.ipynb`: kiểm tra môi trường, RAM/VRAM, dữ liệu và hướng dẫn nhập checkpoint Colab.
2. `01_B0.ipynb`: đánh giá PhoWhisper-small zero-shot.
3. `02_B1.ipynb`: đánh giá Whisper-small zero-shot.
4. `03_V1.ipynb`: tự kiểm tra checkpoint. Chưa có checkpoint thì chọn workflow train: train thử → train đầy đủ → đánh giá thử → đánh giá DEV. Nếu đã có final local/Colab thì chọn workflow evaluate và bỏ qua train. Không chọn trọng số validation_* hoặc train_smoke.
5. `04_context_data.ipynb`: lập danh sách TRAIN tách video với DEV cho phần context/gate.
6. `05_compare_runs.ipynb`: tổng hợp WER/CER và mức dùng RAM/VRAM.
7. `06_download_train_audio.ipynb`: sau notebook 04, tải audio gốc TRAIN từng lô tối đa 30 video. Cell kiểm tra dung lượng chạy trước; cell tải do bạn chủ động bấm ▶. Giới hạn thư mục audio 10 GB, giữ ổ còn 20 GB trống; bỏ qua video trên 60 phút trong lô đầu. Không dùng GPU.
11. `11_TEST.ipynb`: TEST cuối. V4B đã chốt trước trong `ASR/reports/final_freeze_20261027/freeze.json` (`scripts/freeze_final.py`); mỗi ô giải mã TEST + HARD một lần cho V4B, B0V, V3, V4, (V3B), có kết quả thì không chạy lại. Tắt kernel notebook cũ và Chrome trước khi chạy (bộ nhớ commit). Sinh lại bằng `ASR/local/build_test_notebook.py`.
10. `10_V4.ipynb`: V4 = V3 + SpecAugment (chỉ đổi một biến) → train ~16 giờ, DUTY 1,0 → đánh giá V4 → beam search 5 cho V3 (V3B) và V4 (V4B) → bảng so B0V/V3/V3B/V4/V4B. Sinh lại bằng `ASR/local/build_v4_notebook.py`.
9. `09_V3.ipynb`: dữ liệu V3 (ViMedCSS chia lại theo video, loại clip im lặng/nhãn quá dày) → train V3 từ PhoWhisper gốc, lr 3e-4, DUTY 1,0 → đánh giá V3 và B0V trên DEV mới → (tùy chọn) V1/V2/B0/B1 với guard. Sinh lại bằng `ASR/local/build_v3_notebook.py`.
8. `08_V2.ipynb`: train tiếp V1 (khởi tạo từ `local_v1/final`, lr 1e-4, 2 epoch, eval batch 8) → đánh giá DEV → bảng so V1/V2. Chạy lại cell train với cùng run-id `local_v2` để tiếp tục từ checkpoint. Sinh lại bằng `ASR/local/build_v2_notebook.py`.

Notebook 06 lưu FLAC mono 16 kHz tại `ASR/local/imports/original_audio/train`. Công cụ yt-dlp/FFmpeg được cài vào .venv-asr khi bạn chạy cell cài đặt; Deno portable lưu tại `ASR/local/tools/deno`. File tải tạm nằm trong thư mục audio và được tính vào ngân sách. Giới hạn dung lượng được giám sát mỗi giây với dự phòng, không phải quota cứng của hệ điều hành. File tạm được giữ khi lỗi và chỉ xóa sau khi FLAC hoàn chỉnh đã lưu thành công. Dataset và checkpoint không bị xóa.

Sau lô đầu, xem `train_audio_last_batch.csv` và `train_audio_available.csv` trong `ASR/local/context`. `pending` nghĩa là chưa kiểm tra khớp tiếng với clip dataset; không lấy trực tiếp danh sách này để train gate. Chạy lại cell tải sẽ tiếp tục lô mới, không tải lại FLAC đã hợp lệ. Video lỗi được bỏ qua trừ khi bạn bật RETRY_FAILED. Lịch sử mọi lượt nằm trong `train_audio_downloads.jsonl` và log chi tiết trong `download_logs/`.

Sau lỗi Chrome `Failed to decrypt with DPAPI`, notebook 06 đã chuyển sang file Netscape cục bộ `ASR/local/secrets/youtube.cookies.txt`, không đọc trực tiếp cơ sở dữ liệu Chrome. Bạn tự xuất cookie của trang YouTube bằng công cụ được FAQ yt-dlp liên kết, lưu đúng đường dẫn rồi chạy cấu hình → kiểm tra → tải một video. Không gửi/commit cookie. Thư mục secrets đã nằm trong .gitignore; assistant không đọc file cookie. Không cần lặp lại việc kết thúc Chrome hoặc tắt bảo vệ Windows. Cookie không bảo đảm khắc phục được chặn HTTP 429.

Sau khi thử thành công, notebook 06 đã đặt `DOWNLOAD_ALL=True`, thử toàn bộ video còn thiếu trong manifest, nghỉ 8 giây giữa các video, và dùng lại cookie file hiện có. Hạn mức thư mục 10 GB, ổ trống 20 GB và giới hạn video 60 phút vẫn áp dụng; vì vậy toàn bộ danh sách có thể chưa tải hết nếu chạm ngưỡng hoặc bị YouTube chặn. Chạy lại tiếp tục các file còn thiếu, không tải lại FLAC hợp lệ. Chỉ lưu audio, không lưu hình MP4. Bạn tự chạy; assistant không khởi chạy tải.

Mở notebook trong VS Code, chọn Python kernel có sẵn; các tác vụ model sử dụng `.venv-asr` qua tiến trình con để giải phóng bộ nhớ khi kết thúc. Chạy cell khai báo và cấu hình trước, sau đó bấm ▶ đúng cell tác vụ bạn muốn. B0/B1 chỉ đánh giá; V1 có cả phần đánh giá và phần huấn luyện, chọn phần phù hợp.

Đã bỏ các cờ RUN=False ở cell đánh giá/huấn luyện để tránh chạy xong 0 giây mà không làm gì. Bấm ▶ cell tác vụ là bắt đầu ngay, không cần đổi True/False. Không bấm Run All trong V1 trừ khi bạn muốn chạy cả đánh giá, train thử và train đầy đủ. Khi đổi profile hoặc checkpoint, đổi `--run-id` ở cell muốn chạy để không lẫn kết quả cũ. Cell kiểm tra môi trường chỉ in thông tin, không tải model để giải mã hay huấn luyện. Bước chuẩn bị dữ liệu trong notebook 00 vẫn giữ RUN_PREPARE để không chuyển đổi lại dữ liệu ngoài ý muốn.

Log xuất hiện ngay: START/LOG, nhập thư viện, tải model, đọc dữ liệu; nếu tiến trình chưa có dòng mới trong 10 giây sẽ hiện RUNNING. Tiến trình đánh giá có số mẫu đã xử lý. Huấn luyện có bước optimizer, microbatch, phần trăm, epoch, ETA ước lượng, RAM/VRAM, loss theo lịch logging và thông báo checkpoint. ETA giai đoạn đầu có thể chưa chính xác do chưa tính đủ các lần đánh giá/lưu.

Huấn luyện local lưu mỗi 100 bước optimizer hoặc sau 10 phút tại ranh giới bước gần nhất; giữ best/latest tối đa 2 checkpoint theo cơ chế Trainer. Bấm Interrupt sẽ dừng cả tiến trình con; tiếp tục từ checkpoint hoàn chỉnh gần nhất bằng cùng run-id. Kernel bị kill đột ngột có thể không chạy được phần dọn tiến trình; kiểm tra GPU trước khi mở lượt khác.

Notebook 04 mới chuẩn bị dữ liệu cho phần sau; chưa có tập dữ liệu gate hay thuật toán gate đã huấn luyện. Không lấy 30 clip DEV đã thử để làm TRAIN.

RAM hệ thống và VRAM GPU là hai loại bộ nhớ khác nhau. Máy hiện có khoảng 16 GiB RAM hệ thống và 6 GiB VRAM. Profile mặc định dành ngân sách 4,5 GB thập phân cho tổng VRAM ước tính; RAM hệ thống được đo riêng, chưa giới hạn. Pacing 70% tạo thời gian nghỉ giữa tác vụ, không khóa GPU-Util ở 70%.

Chi tiết checkpoint, tiếp tục huấn luyện, cache và giới hạn kiểm soát tài nguyên: `ASR/local/README.md`.
