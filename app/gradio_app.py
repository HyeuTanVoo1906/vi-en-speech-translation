import argparse, json, os, tempfile
import gradio as gr
from common.textnorm import normalize_text
from MT.mt.noise import verbalize_numbers
from pipeline.export import to_srt
from pipeline.inference import SpeechTranslator

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--mt-adapter", default="runs/mt/M4_s42/adapter")
ap.add_argument("--asr-adapter", default=None)
args = ap.parse_args()
ST = SpeechTranslator(asr_adapter=args.asr_adapter, mt_adapter=args.mt_adapter)


def run_audio(path):
    if not path:
        return [], "", "", None, "Chưa có âm thanh."
    try:
        r = ST.translate_file(path)
    except Exception as e:
        return [], "", "", None, f"Lỗi: {e}"
    if not r["segments"]:
        return [], "", "", None, "Không phát hiện giọng nói trong âm thanh."
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


with gr.Blocks(title="Vietnamese speech to English") as demo:
    gr.Markdown("# Dịch giọng nói tiếng Việt sang tiếng Anh\nCascade: Silero VAD, PhoWhisper-small (ASR), VinAI Translate vi2en + LoRA (MT, mô hình M4).")
    with gr.Tabs():
        with gr.Tab("Tải file âm thanh"):
            a1 = gr.Audio(sources=["upload"], type="filepath", label="File âm thanh (wav, flac, mp3, ogg)")
            b1 = gr.Button("Dịch", variant="primary")
        with gr.Tab("Micro"):
            a2 = gr.Audio(sources=["microphone"], type="filepath", label="Ghi âm", waveform_options=gr.WaveformOptions(show_recording_waveform=False))
            b2 = gr.Button("Dịch", variant="primary")
        with gr.Tab("Văn bản sang English"):
            t_in = gr.Textbox(lines=4, label="Văn bản tiếng Việt")
            t_asr = gr.Checkbox(value=False, label="Xử lý như đầu ra ASR (viết thường, bỏ dấu câu, đọc số thành chữ)")
            t_btn = gr.Button("Dịch", variant="primary")
            t_out = gr.Textbox(lines=4, label="English")
            t_btn.click(run_text, [t_in, t_asr], t_out)
    info = gr.Markdown()
    table = gr.Dataframe(headers=["Bắt đầu (s)", "Kết thúc (s)", "Tiếng Việt (ASR)", "English"], wrap=True, label="Theo từng đoạn")
    vi_box = gr.Textbox(label="Toàn bộ tiếng Việt (ASR)", lines=3)
    en_box = gr.Textbox(label="Toàn bộ English", lines=3)
    files = gr.File(label="Tải về (SRT tiếng Anh và JSON)", file_count="multiple")
    outs = [table, vi_box, en_box, files, info]
    b1.click(run_audio, a1, outs)
    b2.click(run_audio, a2, outs)

demo.queue().launch(server_name="127.0.0.1", server_port=args.port)
