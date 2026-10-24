import argparse, json, os, tempfile
import gradio as gr
from common.textnorm import normalize_text
from MT.mt.noise import verbalize_numbers
from pipeline.export import to_srt
from pipeline.inference import SpeechTranslator, load_audio
from pipeline.streaming import StreamSession

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--mt-adapter", default="runs/mt/M4_s42/adapter")
ap.add_argument("--asr-adapter", default=None)
args = ap.parse_args()
ST = SpeechTranslator(asr_adapter=args.asr_adapter, mt_adapter=args.mt_adapter)
HEADERS = ["Bắt đầu (s)", "Kết thúc (s)", "Tiếng Việt (ASR)", "English"]
WAVE = gr.WaveformOptions(show_recording_waveform=False)


def run_audio(path):
    if not path:
        return [], "", "", None, "Chưa có âm thanh."
    try:
        r = ST.translate_file(path)
        if not r["segments"]:
            pk = float(load_audio(path).abs().max())
            return [], "", "", None, f"Không phát hiện giọng nói. Âm lượng đỉnh {pk:.4f} (gần 0 nghĩa là micro không thu được tiếng, kiểm tra thiết bị micro)."
    except Exception as e:
        return [], "", "", None, f"Lỗi: {e}"
    d = tempfile.mkdtemp()
    srt, js = os.path.join(d, "english.srt"), os.path.join(d, "result.json")
    open(srt, "w", encoding="utf-8").write(to_srt(r["segments"], "en"))
    json.dump(r, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rows = [[s["start"], s["end"], s["vi"], s["en"]] for s in r["segments"]]
    st = r["stats"]
    info = (f"Độ dài {st['audio_s']}s | VAD {st['vad_s']}s | ASR {st['asr_s']}s | MT {st['mt_s']}s | "
            f"RTF {st['rtf']} | {st['segments']} đoạn")
    return rows, r["vi"], r["en"], [srt, js], info


def run_text(text, as_asr):
    if not text or not text.strip():
        return ""
    src = normalize_text(verbalize_numbers(text)) if as_asr else text
    return ST.mt.translate([src], batch_size=1, desc="mt")[0]


def _view(sess):
    rows = [[g["start"], g["end"], g["vi"], g["en"]] for g in sess.segments]
    vi = " ".join(g["vi"] for g in sess.segments)
    en = " ".join(g["en"] for g in sess.segments)
    status = ("Đang nói..." if sess.speaking else "Đang nghe...") + f" ({len(sess.segments)} đoạn đã dịch)"
    return rows, vi, en, status


def stream_fn(audio, sess):
    if sess is None:
        sess = StreamSession(ST)
    try:
        if audio is not None:
            sr, chunk = audio
            sess.push(sr, chunk)
    except Exception as e:
        rows, vi, en, _ = _view(sess)
        return sess, rows, vi, en, f"Lỗi: {e}"
    return (sess,) + _view(sess)


def stream_stop(sess):
    if sess is None:
        return None, [], "", "", "Chưa ghi âm."
    try:
        sess.flush()
    except Exception as e:
        rows, vi, en, _ = _view(sess)
        return sess, rows, vi, en, f"Lỗi: {e}"
    rows, vi, en, _ = _view(sess)
    return sess, rows, vi, en, f"Đã dừng. {len(sess.segments)} đoạn."


def stream_reset():
    return None, [], "", "", "Đang nghe..."


with gr.Blocks(title="Vietnamese speech to English") as demo:
    gr.Markdown("# Dịch giọng nói tiếng Việt sang tiếng Anh\nCascade: Silero VAD, PhoWhisper-small (ASR), VinAI Translate vi2en + LoRA (MT, mô hình M4).")
    with gr.Tabs():
        with gr.Tab("Tải file âm thanh"):
            a1 = gr.Audio(sources=["upload"], type="filepath", label="File âm thanh (wav, flac, mp3, ogg)")
            b1 = gr.Button("Dịch", variant="primary")
        with gr.Tab("Micro (ghi xong mới dịch)"):
            a2 = gr.Audio(sources=["microphone"], type="filepath", label="Ghi âm", waveform_options=WAVE)
            b2 = gr.Button("Dịch", variant="primary")
        with gr.Tab("Micro trực tiếp (real-time)"):
            gr.Markdown("Nhấn ghi âm và nói. Mỗi khi bạn dừng khoảng 1 giây, câu vừa nói sẽ được nhận dạng và dịch ngay, trong lúc micro vẫn mở.")
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Nói vào micro", waveform_options=WAVE)
            rt_state = gr.State(None)
            rt_status = gr.Markdown("Chưa ghi âm.")
            rt_table = gr.Dataframe(headers=HEADERS, wrap=True, label="Các câu đã dịch")
            rt_vi = gr.Textbox(label="Tiếng Việt (ASR)", lines=3)
            rt_en = gr.Textbox(label="English", lines=3)
            rt_outs = [rt_state, rt_table, rt_vi, rt_en, rt_status]
            mic.stream(stream_fn, [mic, rt_state], rt_outs, stream_every=0.5, time_limit=None)
            mic.start_recording(stream_reset, None, rt_outs)
            mic.stop_recording(stream_stop, rt_state, rt_outs)
        with gr.Tab("Văn bản sang English"):
            t_in = gr.Textbox(lines=4, label="Văn bản tiếng Việt")
            t_asr = gr.Checkbox(value=False, label="Xử lý như đầu ra ASR (viết thường, bỏ dấu câu, đọc số thành chữ)")
            t_btn = gr.Button("Dịch", variant="primary")
            t_out = gr.Textbox(lines=4, label="English")
            t_btn.click(run_text, [t_in, t_asr], t_out)
    gr.Markdown("### Kết quả của hai tab tải file và micro ghi xong mới dịch")
    info = gr.Markdown()
    table = gr.Dataframe(headers=HEADERS, wrap=True, label="Theo từng đoạn")
    vi_box = gr.Textbox(label="Toàn bộ tiếng Việt (ASR)", lines=3)
    en_box = gr.Textbox(label="Toàn bộ English", lines=3)
    files = gr.File(label="Tải về (SRT tiếng Anh và JSON)", file_count="multiple")
    outs = [table, vi_box, en_box, files, info]
    b1.click(run_audio, a1, outs)
    b2.click(run_audio, a2, outs)

demo.queue().launch(server_name="127.0.0.1", server_port=args.port)
