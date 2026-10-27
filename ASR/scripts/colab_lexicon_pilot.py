"""Paste this whole file into a Colab cell AFTER the beam N-best pilot.

Diagnostic only: TRAIN vocabulary + spelling similarity, not an audio reranker.
Required notebook variables: RAW_DATA, PROJECT, row, dataset_prediction, beam_texts.
Gold is accessed only after the complete candidate pool is frozen.
"""
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import csv
import hashlib
import json
import unicodedata

import pandas as pd
from IPython.display import display
from asr.metrics import normalize_text, compute_metrics


def spelling_key(text):
    text = unicodedata.normalize('NFD', normalize_text(text))
    return ''.join(c for c in text if c.isalnum() and not unicodedata.combining(c))


def build_candidates(train_csv, greedy, beams, top_k=5):
    # No reference argument: retrieval cannot inspect DEV labels.
    counts = Counter()
    with open(train_csv, encoding='utf-8-sig', newline='') as handle:
        for item in csv.DictReader(handle):
            terms = {normalize_text(t) for t in item['cs_terms_list'].split(';')}
            counts.update(t for t in terms if t)
    lexicon = [(term, spelling_key(term)) for term in sorted(counts)]
    base = normalize_text(greedy).split()
    spans = set()
    for beam in beams:
        other = normalize_text(beam).split()
        for tag, i, j, _, _ in SequenceMatcher(None, base, other, autojunk=False).get_opcodes():
            if tag != 'equal' and i < j:
                # Bounded one-to-three-word windows touching a disagreement.
                for left in range(max(0, i - 1), j):
                    for right in range(max(left + 1, i + 1), min(len(base), left + 3) + 1):
                        spans.add((left, right))
    candidates = []
    seen = set()

    def add(text, origin, span='', term='', similarity=None):
        text = normalize_text(text)
        if text not in seen:
            seen.add(text)
            candidates.append(dict(prediction=text, origin=origin, span=span,
                                   term=term, similarity=similarity,
                                   train_rows=counts.get(term, 0)))

    add(greedy, 'KEEP')
    for i, beam in enumerate(beams, 1):
        add(beam, f'beam_{i}')
    for left, right in sorted(spans):
        span = ' '.join(base[left:right])
        key = spelling_key(span)
        ranked = sorted(
            ((SequenceMatcher(None, key, term_key, autojunk=False).ratio(), term)
             for term, term_key in lexicon if term_key and term != span),
            key=lambda x: (-x[0], x[1]),
        )[:top_k]
        for similarity, term in ranked:
            add(' '.join(base[:left] + term.split() + base[right:]),
                'train_lexicon', span, term, similarity)
    return candidates, counts, sorted(spans)


train_csv = Path(RAW_DATA) / 'ViMedCSS/ViMedCSS-Metadata/train_set.csv'
pool, counts, spans = build_candidates(train_csv, dataset_prediction, beam_texts)
report_dir = Path(PROJECT) / 'reports'
report_dir.mkdir(parents=True, exist_ok=True)
stem = report_dir / f"lexicon_pilot_{row['segment_id']}"

# Persist the reference-blind pool BEFORE consulting the reference for diagnostics.
pd.DataFrame(pool).to_csv(str(stem) + '_candidates.csv', index=False)
manifest = dict(
    lexicon_source=str(train_csv), train_sha256=hashlib.sha256(train_csv.read_bytes()).hexdigest(),
    unique_terms=len(counts), top_k_per_span=5, max_span_words=3,
    spans=spans, audio_input='dataset clip', selection='none; diagnostic only',
)
Path(str(stem) + '_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

reference = row['segment_text']
for candidate in pool:
    candidate['WER (%)'] = 100 * compute_metrics([reference], [candidate['prediction']])['wer']
results = pd.DataFrame(pool)
results.to_csv(str(stem) + '_scored.csv', index=False)
keep_wer = results.iloc[0]['WER (%)']
oracle_wer = results['WER (%)'].min()
print(f'Thuật ngữ TRAIN: {len(counts)} | Ứng viên duy nhất: {len(results)}')
print(f'KEEP WER: {keep_wer:.2f}% | Oracle pool WER: {oracle_wer:.2f}%')
print('Oracle dùng nhãn để đo tiềm năng, KHÔNG phải kết quả tự động chọn.')
with pd.option_context('display.max_colwidth', None):
    display(results.sort_values('WER (%)', kind='stable').head(15))
print('Đã lưu:', str(stem) + '_scored.csv')
