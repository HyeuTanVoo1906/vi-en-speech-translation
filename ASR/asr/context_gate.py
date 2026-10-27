"""Reference-blind features and a portable, two-head linear context gate."""
import math
import re
import unicodedata
from difflib import SequenceMatcher

FEATURES = ['audio_delta', 'base_audio', 'context_audio', 'history_confidence',
            'edit_ratio', 'length_delta', 'history_overlap_delta', 'duration',
            'history_words', 'base_truncated', 'context_truncated', 'history_truncated']


def words(text):
    return re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFC', text).lower()).split()


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def features(base, context, history, base_audio, context_audio, history_confidence,
             duration, base_truncated=False, context_truncated=False, history_truncated=False):
    """No reference, term list, ID, split, or gold-derived feature is accepted."""
    b, c, h = words(base), words(context), set(words(history))
    overlap = lambda seq: sum(w in h for w in seq) / max(1, len(seq))
    result = [context_audio-base_audio, base_audio, context_audio, history_confidence,
              1-SequenceMatcher(None, b, c, autojunk=False).ratio(),
              (len(c)-len(b))/max(1, len(b)), overlap(c)-overlap(b), duration/30,
              len(h)/100, float(base_truncated), float(context_truncated), float(history_truncated)]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Nonfinite feature')
    return result


def label(reference, base, context):
    ref = words(reference)
    if not ref:
        raise ValueError('Empty reference cannot label gate data')
    b, c = edit_distance(ref, words(base)), edit_distance(ref, words(context))
    return dict(base_errors=b, context_errors=c, ref_words=len(ref), benefit=int(c < b), harm=int(c > b))


def probability(head, x):
    if 'constant' in head:
        return head['constant']
    z = head['intercept'] + sum(w*(v-m)/s for w, v, m, s in
                               zip(head['coef'], x, head['mean'], head['scale']))
    return 1/(1+math.exp(-max(-40, min(40, z))))


def choose(model, row):
    p = probability(model['benefit'], row['features'])
    harm = probability(model['harm'], row['features'])
    replace = (words(row['base']) != words(row['context']) and
               not row['context_truncated'] and not row['history_truncated'] and
               p >= model['benefit_threshold'] and harm <= model['harm_threshold'])
    return ('REPLACE' if replace else 'KEEP'), p, harm
