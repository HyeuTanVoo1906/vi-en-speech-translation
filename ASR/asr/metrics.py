"""Text normalization and ASR metrics (WER/CER, code-switch term recall)."""
import re
import unicodedata

import jiwer

_NON_WORD = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    """NFC, lowercase, punctuation -> space, collapse whitespace. Same rule for train targets and scoring."""
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(_NON_WORD.sub(" ", text).split())


def compute_metrics(refs: list[str], hyps: list[str], cs_terms: list[str] | None = None) -> dict:
    refs = [normalize_text(r) for r in refs]
    hyps = [normalize_text(h) for h in hyps]
    keep = [i for i, r in enumerate(refs) if r]  # jiwer cannot score empty references
    r, h = [refs[i] for i in keep], [hyps[i] for i in keep]
    words = jiwer.process_words(r, h)
    out = {
        "n_utts": len(keep),
        "n_empty_refs_skipped": len(refs) - len(keep),
        "wer": words.wer,
        "cer": jiwer.cer(r, h),
        "ref_words": words.hits + words.substitutions + words.deletions,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
    }
    if cs_terms and any(cs_terms):
        # Approximate: a term counts as recognized if its normalized form occurs in the hypothesis.
        total = found = 0
        for terms, hyp in zip(cs_terms, hyps):
            for term in filter(None, (normalize_text(t) for t in terms.split(";"))):
                total += 1
                found += term in hyp
        out.update(cs_terms_total=total, cs_term_recall=found / total if total else None)
    return out
