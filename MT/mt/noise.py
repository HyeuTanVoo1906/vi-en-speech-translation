import json, random, re, unicodedata
from collections import Counter, defaultdict
from common.textnorm import normalize_text

ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = ["", "ngàn", "triệu", "tỷ"]
TONES = ["\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]
_THOUS = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_DEC = re.compile(r"\b(\d+)[.,](\d{1,2})\b")
_NUM = re.compile(r"\d+")


def _group(n, full):
    h, t, u = n // 100, n // 10 % 10, n % 10
    w = []
    if h:
        w += [ONES[h], "trăm"]
    elif full and (t or u):
        w += ["không", "trăm"]
    if t > 1:
        w += [ONES[t], "mươi"]
        if u == 1:
            w.append("mốt")
        elif u == 5:
            w.append("lăm")
        elif u:
            w.append(ONES[u])
    elif t == 1:
        w.append("mười")
        if u == 5:
            w.append("lăm")
        elif u:
            w.append(ONES[u])
    elif u:
        if h or full:
            w.append("lẻ")
        w.append(ONES[u])
    return w


def number_to_words(s, rng=None):
    if len(s) > 9 or (len(s) > 1 and s[0] == "0"):
        return " ".join(ONES[int(c)] for c in s)
    n = int(s)
    if n == 0:
        return "không"
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    out = []
    for i in range(len(groups) - 1, -1, -1):
        g = groups[i]
        if not g:
            continue
        out += _group(g, i < len(groups) - 1)
        if i:
            sc = SCALES[i]
            if i == 1 and rng is not None and rng.random() < 0.3:
                sc = "nghìn"
            out.append(sc)
    return " ".join(out)


def verbalize_numbers(s, rng=None):
    s = _THOUS.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("%", " phần trăm ")
    s = _DEC.sub(lambda m: f" {number_to_words(m.group(1), rng)} phẩy {number_to_words(m.group(2), rng)} ", s)
    return _NUM.sub(lambda m: " " + number_to_words(m.group(0), rng) + " ", s)


def align_regions(ref, hyp):
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "x", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("x", i - 1, None))
            i -= 1
        else:
            ops.append(("x", None, j - 1))
            j -= 1
    ops.reverse()
    regions, cr, ch = [], [], []
    for kind, a, b in ops + [("=", None, None)]:
        if kind == "x":
            if a is not None:
                cr.append(ref[a])
            if b is not None:
                ch.append(hyp[b])
        elif cr or ch:
            regions.append((tuple(cr), tuple(ch)))
            cr, ch = [], []
    return regions, d[n][m]


def swap_tone(w, rng):
    d = unicodedata.normalize("NFD", w)
    pos = [i for i, c in enumerate(d) if c in TONES]
    if not pos:
        return w
    i = pos[0]
    choices = [t for t in TONES if t != d[i]] + [""]
    new = rng.choice(choices)
    rest = d[:i] + d[i + 1:]
    j = i
    while j < len(rest) and rest[j] in "\u0302\u0306\u031b":
        j += 1
    return unicodedata.normalize("NFC", rest[:j] + new + rest[j:])


class NoiseModel:
    def __init__(self, lex, occ, ins, wer, p_ins):
        self.lex, self.occ, self.ins, self.wer, self.p_ins = lex, occ, ins, wer, p_ins
        self.tot = {r: sum(c.values()) for r, c in lex.items()}
        self.ins_words = list(ins.keys())
        self.ins_w = list(ins.values())

    @classmethod
    def build(cls, pairs):
        lex, ins, occ = defaultdict(Counter), Counter(), Counter()
        tot = dist = 0
        for ref, hyp in pairs:
            tot += len(ref)
            for n in range(1, 5):
                for k in range(len(ref) - n + 1):
                    occ[tuple(ref[k:k + n])] += 1
            regs, dd = align_regions(ref, hyp)
            dist += dd
            for r, h in regs:
                if not r:
                    ins.update(h)
                elif len(r) <= 4:
                    lex[r][h] += 1
        occ = {r: occ[r] for r in lex}
        return cls(dict(lex), occ, ins, dist / tot, sum(ins.values()) / tot)

    def corrupt(self, words, rng, scale=1.0):
        out, i, n = [], 0, len(words)
        g_tone, g_del = 0.4 * self.wer * scale, 0.1 * self.wer * scale
        while i < n:
            done = False
            for L in (4, 3, 2, 1):
                r = tuple(words[i:i + L])
                if len(r) == L and r in self.lex and rng.random() < scale * self.tot[r] / (self.occ[r] + 3):
                    c = self.lex[r]
                    out += list(rng.choices(list(c.keys()), weights=list(c.values()))[0])
                    i += L
                    done = True
                    break
            if done:
                continue
            x = rng.random()
            if x < g_del:
                pass
            elif x < g_del + g_tone:
                out.append(swap_tone(words[i], rng))
            else:
                out.append(words[i])
            if self.ins_words and rng.random() < scale * self.p_ins:
                out.append(rng.choices(self.ins_words, weights=self.ins_w)[0])
            i += 1
        return out

    def save(self, path):
        obj = {"wer": self.wer, "p_ins": self.p_ins, "ins": dict(self.ins),
               "lex": {" ".join(r): {" ".join(h): c for h, c in cnt.items()} for r, cnt in self.lex.items()},
               "occ": {" ".join(r): c for r, c in self.occ.items()}}
        json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        o = json.load(open(path, encoding="utf-8"))
        lex = {tuple(k.split()): Counter({tuple(h.split()): c for h, c in v.items()}) for k, v in o["lex"].items()}
        occ = {tuple(k.split()): c for k, c in o["occ"].items()}
        return cls(lex, occ, Counter(o["ins"]), o["wer"], o["p_ins"])


def make_noisy(raw, model, rng, p_noisy=0.7, scale=1.0):
    clean = normalize_text(verbalize_numbers(raw, rng))
    if rng.random() < p_noisy:
        noisy = " ".join(model.corrupt(clean.split(), rng, scale))
        if noisy.strip():
            return noisy
    return clean
