import re
import unicodedata

def normalize_text(s: str) -> str:
    """Same rule as the ASR team's metrics.normalize_text: NFC, lowercase, drop punctuation, collapse spaces."""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

_TONE_PAIRS = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_MAP = {}
for _n, _o in _TONE_PAIRS.items():
    _TONE_MAP[_n] = _o
    _TONE_MAP[_n.capitalize()] = _o.capitalize()
    _TONE_MAP[_n.upper()] = _o.upper()
_TONE_RE = re.compile("|".join(map(re.escape, _TONE_MAP)))

def unify_tone(s: str) -> str:
    """Dua dau thanh ve kieu cu cua PhoMT (hoa -> hoa, thuy -> thuy). 'quy' va 'quy' khong doi."""
    s = unicodedata.normalize("NFC", s)
    return _TONE_RE.sub(lambda m: _TONE_MAP[m.group(0)], s)
