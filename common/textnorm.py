import re
import unicodedata

def normalize_text(s: str) -> str:
    """Same rule as the ASR team's metrics.normalize_text: NFC, lowercase, drop punctuation, collapse spaces."""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
