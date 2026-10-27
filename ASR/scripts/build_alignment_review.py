"""Build an offline listening package and final audit tables from measured alignment."""
import argparse
import collections
import datetime
import hashlib
import html
import io
import json
import statistics
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from validate_train_alignment import ROOT, SR, load_range, read_csv, write_csv


def audit_raw_problems(rows, out):
    targets = {r["segment_id"]: r for r in rows
               if r["reason"] in ("silent_or_near_silent_dataset_clip", "read_or_analysis_error",
                                  "raw_dataset_empty", "raw_dataset_digital_silence")}
    findings = []
    for path in sorted((ROOT.parent/"dataset/ASR/ViMedCSS/data").glob("train-*.parquet")):
        if not targets:
            break
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            ids = parquet.read_row_group(group, columns=["segment_id"])["segment_id"].to_pylist()
            if not any(sid in targets for sid in ids):
                continue
            for item in parquet.read_row_group(group, columns=["segment_id", "audio"]).to_pylist():
                sid = item["segment_id"]
                if sid not in targets:
                    continue
                record = dict(segment_id=sid, raw_parquet=path.name,
                              raw_audio_bytes=len(item["audio"]["bytes"]))
                try:
                    audio, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32")
                    record.update(samples=len(audio), sample_rate=sr,
                                  duration_s=len(audio)/sr,
                                  peak=float(np.max(np.abs(audio))) if len(audio) else 0)
                    record["raw_status"] = ("empty" if not len(audio) else
                                            "digital_silence" if not np.any(audio) else "has_signal")
                except Exception as exc:
                    record.update(raw_status="decode_error", error=str(exc))
                row = targets.pop(sid)
                row["raw_dataset_status"] = record["raw_status"]
                if record["raw_status"] in ("empty", "digital_silence"):
                    row.setdefault("automated_alignment_reason", row["reason"])
                    row.update(status="exclude", reason="raw_dataset_"+record["raw_status"])
                findings.append(record)
    (out/"raw_problem_clips.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    write_csv(out/"raw_problem_clips.csv", findings)
    return findings


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    out = args.output
    rows = read_csv(out/"alignment_all.csv")
    raw_findings = audit_raw_problems(rows, out)
    # Alignment validity and the amount of preceding history are separate properties.
    # Apply this policy to measurements from runs started before the policy was split.
    numeric = ["nominal_start_s", "nominal_end_s", "matched_start_s", "matched_end_s",
               "offset_s", "duration_delta_s", "correlation", "peak_margin",
               "first_third_correlation", "middle_third_correlation", "last_third_correlation"]
    for r in rows:
        for key in numeric:
            if r.get(key, "") != "":
                r[key] = float(r[key])
        if r["reason"] == "matched_but_less_than_20s_history":
            unchanged = abs(r["offset_s"]) <= .08 and abs(r["duration_delta_s"]) <= .05
            r["status"] = "pass" if unchanged else "adjust"
            r["reason"] = "strong_consistent_waveform_match" if unchanged else "strong_match_timestamp_correction"
        r["history_20s_ready"] = (r.get("matched_start_s") or -1) >= 20
        r["listening_status"] = "not_listened"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(out/"alignment_all.csv", rows, fields)
    for status in ("pass", "adjust", "review", "exclude"):
        write_csv(out/f"alignment_{status}.csv", [r for r in rows if r["status"] == status], fields)
    safe = []
    for r in rows:
        if r["status"] not in ("pass", "adjust"):
            continue
        safe.append(dict(segment_id=r["segment_id"], video_id=r["video_id"], audio_path=r["audio_path"],
                         start_s=r["matched_start_s"], end_s=r["matched_end_s"],
                         history_start_s=max(0, r["matched_start_s"]-20),
                         available_history_s=min(20, r["matched_start_s"]),
                         history_20s_ready=r["history_20s_ready"],
                         alignment_status="waveform_verified",
                         timestamp_action=r["status"], correlation=r["correlation"],
                         listening_status="not_listened"))
    write_csv(out/"train_audio_waveform_verified.csv", safe)
    write_csv(out/"train_audio_waveform_verified_20s.csv",
              [r for r in safe if r["history_20s_ready"]], list(safe[0]) if safe else [])
    # Every uncertain/corrected clip, plus a deterministic sample of accepted clips.
    passed = sorted([r for r in rows if r["status"] == "pass"],
                    key=lambda r: hashlib.sha256(r["segment_id"].encode()).hexdigest())[:20]
    selected = [r for r in rows if r["status"] != "pass"] + passed
    selected.sort(key=lambda r: ({"review":0, "exclude":1, "adjust":2, "pass":3}[r["status"]],
                                  r["video_id"], r["segment_id"]))
    wanted = {r["segment_id"]: r for r in selected}
    media = out/"listening"
    media.mkdir(exist_ok=True)
    parquet = pq.ParquetFile(ROOT/"data/ViMedCSS/train.parquet")
    exported = set()
    for group in range(parquet.num_row_groups):
        ids = parquet.read_row_group(group, columns=["id"])["id"].to_pylist()
        if not any(i.removeprefix("vimedcss:") in wanted for i in ids):
            continue
        for item in parquet.read_row_group(group, columns=["id", "audio"]).to_pylist():
            sid = item["id"].removeprefix("vimedcss:")
            if sid not in wanted:
                continue
            r = wanted[sid]
            try:
                clip, sr = sf.read(io.BytesIO(item["audio"]), dtype="float32")
            except Exception as exc:
                r["review_hint"] = "Không đọc được audio dataset: " + str(exc)
                r["dataset_audio_exported"] = False
                r["matched_audio_exported"] = False
                sf.write(media/f"{sid}_nominal.wav",
                         load_range(Path(r["audio_path"]), r["nominal_start_s"], r["nominal_end_s"]), SR)
                continue
            r["dataset_audio_exported"] = True
            rms = [float(np.sqrt(np.mean(piece**2))) for piece in np.array_split(clip, 3)]
            r["dataset_third_rms"] = json.dumps([round(v, 8) for v in rms])
            r["review_hint"] = ("Một phần clip im lặng/gần im lặng; cần nghe xác nhận."
                                if min(rms) < 1e-5 else "Đối chiếu nội dung và các biên đầu/cuối.")
            sf.write(media/f"{sid}_dataset.wav", clip, sr)
            path = Path(r["audio_path"])
            nominal = load_range(path, r["nominal_start_s"], r["nominal_end_s"])
            sf.write(media/f"{sid}_nominal.wav", nominal, SR)
            if r.get("matched_start_s", "") != "":
                matched = load_range(path, r["matched_start_s"], r["matched_end_s"])
                sf.write(media/f"{sid}_matched.wav", matched, SR)
                # Dataset left, matched source right: useful for detecting drift by listening.
                n = max(len(clip), len(matched))
                pair = np.zeros((n, 2), dtype="float32")
                pair[:len(clip), 0] = clip
                pair[:len(matched), 1] = matched
                sf.write(media/f"{sid}_stereo.wav", pair, SR)
                r["matched_audio_exported"] = True
            else:
                r["matched_audio_exported"] = False
            exported.add(sid)
    write_csv(out/"listening_index.csv", selected)
    payload = json.dumps(selected, ensure_ascii=False).replace("<", "\\u003c")
    document = """<!doctype html><html lang="vi"><meta charset="utf-8">
<title>Kiểm tra alignment audio TRAIN</title>
<style>
body{font:16px system-ui;margin:30px auto;max-width:1150px;padding:0 20px;color:#172033;background:#f3f6fa}
header,.card{background:white;border:1px solid #d8e0eb;border-radius:12px;padding:22px;margin:16px 0}
h1{font-size:27px}h2{font-size:20px}small{color:#526176}audio{width:100%;max-width:480px}
.players{display:grid;grid-template-columns:1fr 1fr;gap:14px}.player{padding:12px;background:#f3f6fa;border-radius:8px}
button,select,input{padding:10px;margin:4px;border:1px solid #acb9ca;border-radius:6px;background:white}
.pill{font-weight:700;color:#19538b}.warning{border-left:4px solid #bf6b18;padding:10px}
@media(max-width:700px){.players{grid-template-columns:1fr}}
</style><header><h1>Đối chiếu audio TRAIN</h1>
<p>Các clip cần xem lại, cần chỉnh mốc và 20 clip đạt được chọn cố định để kiểm tra mẫu.</p>
<p class="warning">Kết quả phân loại do đối chiếu tín hiệu. Chưa có clip nào được nghe xác nhận.
Transcript chỉ để người nghe tham khảo; thuật toán không dùng transcript để tìm mốc.</p>
<p>Nghe A (dataset), B (mốc cũ), C (mốc tìm được). D phát dataset bên trái và bản khớp bên phải.
Với mốc lệch nhiều, kiểm tra cả ngữ cảnh trước clip trước khi sử dụng làm history.</p>
<select id="filter"><option value="all">Tất cả</option><option value="review">Cần nghe kiểm tra</option>
<option value="adjust">Cần chỉnh mốc</option><option value="exclude">Cần loại</option>
<option value="pass">Mẫu đạt</option></select><input id="search" placeholder="Tìm segment/video ID">
<button id="export">Xuất kết quả nghe đã ghi</button><p id="count"></p></header><main id="cards"></main>
<script>
const rows=PAYLOAD;
const labels={review:'Cần kiểm tra',adjust:'Cần chỉnh',exclude:'Cần loại',pass:'Đạt bằng tín hiệu'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const key='alignment-review-20261020';let decisions={};try{decisions=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function save(id,value){decisions[id]=value;try{localStorage.setItem(key,JSON.stringify(decisions))}catch(e){}}
function player(id,suffix,title){return `<div class="player">${title}<br><audio controls preload="none" src="listening/${encodeURIComponent(id)}_${suffix}.wav"></audio></div>`}
function render(){
 const f=document.querySelector('#filter').value,q=document.querySelector('#search').value.toLowerCase();
 const items=rows.filter(r=>(f==='all'||r.status===f)&&(r.segment_id+' '+r.video_id).toLowerCase().includes(q));
 document.querySelector('#count').textContent=`${items.length} clip`;
 document.querySelector('#cards').innerHTML=items.map(r=>`<section class="card"><h2>${esc(r.segment_id)} · <span class="pill">${labels[r.status]}</span></h2>
 <small>Video: ${esc(r.video_id)} · Lý do: ${esc(r.reason)} · ${esc(r.review_hint)}</small>
 <p>Mốc cũ: ${r.nominal_start_s}–${r.nominal_end_s}s · Mốc tìm được: ${r.matched_start_s}–${r.matched_end_s}s
 · Độ lệch: <b>${r.offset_s}s</b> · Tương quan: ${r.correlation}</p>
 <p>${esc(r.transcript)}</p><div class="players">${r.dataset_audio_exported?player(r.segment_id,'dataset','A — Clip dataset'):'<p>Audio dataset không đọc được.</p>'}
 ${player(r.segment_id,'nominal','B — Audio gốc theo mốc cũ')}
 ${r.matched_audio_exported?player(r.segment_id,'matched','C — Audio gốc theo mốc tìm được'):''}
 ${r.matched_audio_exported?player(r.segment_id,'stereo','D — Trái: dataset / phải: bản khớp'):''}</div>
 <p>Kết quả nghe: <select data-id="${esc(r.segment_id)}"><option value="">Chưa nghe</option>
 <option value="confirmed_match">Đã nghe — khớp</option><option value="needs_adjustment">Cần chỉnh thêm</option>
 <option value="reject">Không khớp — loại</option><option value="uncertain">Chưa chắc</option></select></p></section>`).join('');
 document.querySelectorAll('[data-id]').forEach(el=>{el.value=decisions[el.dataset.id]||'';el.onchange=()=>save(el.dataset.id,el.value)});
}
document.querySelector('#filter').onchange=render;document.querySelector('#search').oninput=render;
document.querySelector('#export').onclick=()=>{
 const text='segment_id,listening_decision\\n'+Object.entries(decisions).map(([id,v])=>id+','+v).join('\\n');
 const u=URL.createObjectURL(new Blob(['\\ufeff'+text],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=u;a.download='listening_decisions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
};render();
</script></html>""".replace("PAYLOAD", payload)
    (out/"listen_review.html").write_text(document, encoding="utf-8")
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    counts = dict(collections.Counter(r["status"] for r in rows))
    summary.update(counts={s:counts.get(s,0) for s in ("pass","adjust","review","exclude")},
                   listening_package_clips=len(exported), listening_performed=False,
                   listening_package_rows=len(selected),
                   raw_problem_counts=dict(collections.Counter(r["raw_status"] for r in raw_findings)),
                   verified_clips=len(safe), verified_clips_with_full_20s_history=sum(r["history_20s_ready"] for r in safe),
                   history_policy="Alignment status is independent of preceding history; full-20s manifest filters separately",
                   finalized_at=datetime.datetime.now().isoformat())
    summary["input_manifests_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"local/context/train_candidates.csv", ROOT/"local/context/train_audio_available.csv"]}
    source_parquet = ROOT/"data/ViMedCSS/train.parquet"
    summary["dataset_parquet"] = dict(path=str(source_parquet), size_bytes=source_parquet.stat().st_size,
                                      modified_ns=source_parquet.stat().st_mtime_ns)
    summary["code_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/"scripts/validate_train_alignment.py", Path(__file__)]}
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    adjusted = [r for r in rows if r["status"] == "adjust"]
    by_group = collections.defaultdict(list)
    for r in adjusted:
        by_group[(r["video_id"], r["segment_id"].rsplit("-", 1)[0])].append(r["offset_s"])
    groups = [dict(video_id=vid, segment_group=group, clips=len(offsets),
                   median_offset_s=round(statistics.median(offsets), 6),
                   min_offset_s=min(offsets), max_offset_s=max(offsets))
              for (vid, group), offsets in sorted(by_group.items())]
    write_csv(out/"timestamp_correction_groups.csv", groups)
    largest = sorted(adjusted, key=lambda r: abs(r["offset_s"]), reverse=True)[:10]
    examples = "\n".join(
        f"| {r['segment_id']} | {r['nominal_start_s']:.3f} | {r['matched_start_s']:.3f} | {r['offset_s']:+.3f} | {r['correlation']:.4f} |"
        for r in largest)
    text = f"""# Kiểm tra alignment audio TRAIN — 20/10/2026

Đã đối chiếu đủ **{len(rows):,} clip** trong `train_audio_available.csv` với clip dataset chuẩn bị sẵn trong `ASR/data/ViMedCSS/train.parquet`.

| Phân loại | Số clip | Ý nghĩa |
|---|---:|---|
| Đạt ở mốc hiện tại | {counts.get('pass',0):,} | Khớp waveform mạnh; độ lệch không quá 80 ms và chênh thời lượng không quá 50 ms |
| Cần chỉnh mốc | {counts.get('adjust',0):,} | Tìm được vị trí khớp mạnh; dùng mốc đề xuất trong bảng |
| Cần nghe kiểm tra | {counts.get('review',0):,} | Chưa đủ bằng chứng để chấp nhận hoặc loại |
| Cần loại khỏi tập alignment hiện tại | {counts.get('exclude',0):,} | Clip dataset gốc rỗng/im lặng hoàn toàn, thiếu clip hoặc nguồn không đủ chiều dài |

## File kết quả

- [Toàn bộ kết quả](alignment_all.csv)
- [Đạt](alignment_pass.csv), [cần chỉnh](alignment_adjust.csv), [cần nghe kiểm tra](alignment_review.csv), [cần loại](alignment_exclude.csv)
- [Mốc đã đối chiếu waveform](train_audio_waveform_verified.csv): {len(safe):,} clip; dùng mốc đo được cho cả clip đạt và clip cần chỉnh.
- [Mốc đã đối chiếu và đủ 20 giây history](train_audio_waveform_verified_20s.csv): {summary['verified_clips_with_full_20s_history']:,} clip. Thiếu vài chục mili giây history không có nghĩa audio sai; được tách thành thuộc tính riêng.
- [Nhóm độ lệch](timestamp_correction_groups.csv)
- [Mở trang nghe đối chiếu](listen_review.html): {len(selected):,} mục; {len(exported):,} clip đọc được audio dataset. Gồm các clip cần chỉnh/cần kiểm tra/cần loại, cộng 20 clip đạt được chọn cố định. Với clip dataset rỗng chỉ có audio nguồn theo mốc cũ để tham khảo.
- [Đối chiếu lỗi với dataset gốc](raw_problem_clips.csv): {dict(collections.Counter(r['raw_status'] for r in raw_findings))}. `digital_silence` là toàn bộ mẫu âm thanh bằng 0; `empty` là không có mẫu âm thanh.

## Cách kiểm tra

1. Đọc clip FLAC mono 16 kHz trong parquet và audio gốc tải về; không chạy ASR hay huấn luyện.
2. Tìm tương quan chéo chuẩn hóa waveform ở 4 kHz trong vùng ±3 giây quanh mốc metadata.
3. Nếu điểm khớp dưới 0,85, tìm trên toàn bộ video; loại cửa sổ nguồn im lặng khỏi phép chuẩn hóa để tránh đỉnh khớp giả do sai số số học.
4. Kiểm tra độc lập ba phần đầu/giữa/cuối ở 8 kHz, trong vùng ±100 ms quanh mốc tìm được.
5. Chỉ chấp nhận khi điểm toàn clip ≥0,80, cả ba phần ≥0,65, chênh độ lệch giữa ba phần ≤40 ms và đỉnh chính hơn vị trí thay thế ít nhất 0,15. Vị trí thay thế cách đỉnh chính ít nhất 250 ms trong vùng tìm kiếm.

Các ngưỡng là tiêu chí sàng lọc bảo thủ, không phải xác suất khớp. Độ phân giải tìm kiếm toàn clip là 0,25 ms; số thập phân trong CSV không thể hiện độ chính xác tuyệt đối của timestamp.

## Những mốc cần chỉnh nhiều nhất

| Clip | Bắt đầu cũ (s) | Bắt đầu đề xuất (s) | Độ lệch (s) | Tương quan |
|---|---:|---:|---:|---:|
{examples}

Không áp dụng một độ lệch chung cho toàn video: các nhóm clip có thể dùng hệ mốc khác nhau. Mốc đề xuất được đo riêng từng clip.

## Phần còn cần xác nhận

**Chưa thực hiện nghe trực tiếp.** Tất cả `listening_status` vẫn là `not_listened`. Trang nghe cung cấp dataset, audio theo mốc cũ, audio theo mốc mới và bản stereo A/B; người nghe có thể ghi quyết định và xuất CSV.

Đối chiếu waveform xác nhận clip dataset có mặt ở vị trí tìm được, không chứng minh transcript của dataset hoàn toàn chính xác. Với clip cần chỉnh lớn, cần xem cả ngữ cảnh phía trước khi dùng cho thí nghiệm context.

Các clip `review` không có trong manifest đã xác minh. Không biến trường hợp chưa chắc thành dữ liệu đạt hoặc loại vĩnh viễn. Phạm vi lần này chỉ gồm {len(rows):,} clip đã có trong manifest tải thành công; video tải lỗi và clip đã loại do vượt thời lượng trước lần kiểm tra này nằm ngoài phạm vi.

## Kiểm chứng và nguồn

- Bộ kiểm tra tổng hợp xác nhận độ lệch biết trước, clip không liên quan, hỏng phần cuối, nguồn im lặng và trường hợp thiếu rất ít history.
- `negative_controls.json` ghi 12 phép thử ghép clip thật với video sai: không trường hợp nào đạt ngưỡng chấp nhận. Đây là kiểm tra lỗi cơ bản, không phải ước lượng đầy đủ tỷ lệ khớp nhầm.
- `summary.json` ghi tham số, dấu vân tay hai manifest đầu vào, kích thước/mtime parquet và mã kiểm tra.
- Dữ liệu nguồn và các manifest ban đầu được giữ nguyên; kết quả nằm riêng trong thư mục này.

Chạy lại từ thư mục dự án:

```powershell
.\\.venv-asr\\Scripts\\python.exe ASR/scripts/validate_train_alignment.py --output ASR/local/context/alignment_audit_new
.\\.venv-asr\\Scripts\\python.exe ASR/scripts/build_alignment_review.py ASR/local/context/alignment_audit_new
```
"""
    (out/"REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
