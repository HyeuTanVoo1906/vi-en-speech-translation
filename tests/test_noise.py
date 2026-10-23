import random
from common.textnorm import normalize_text
from MT.mt.noise import number_to_words, verbalize_numbers

def v(s):
    return normalize_text(verbalize_numbers(s))

def test_numbers():
    assert number_to_words("2015") == "hai ngàn không trăm mười lăm"
    assert number_to_words("21") == "hai mươi mốt"
    assert number_to_words("105") == "một trăm lẻ năm"
    assert number_to_words("1000") == "một ngàn"
    assert number_to_words("007") == "không không bảy"

def test_verbalize():
    assert v("Năm 2015") == "năm hai ngàn không trăm mười lăm"
    assert v("9.000 học viên") == "chín ngàn học viên"
    assert v("giảm 3,5%") == "giảm ba phẩy năm phần trăm"

def test_swap_tone_no_stray_marks():
    import random
    from MT.mt.noise import swap_tone
    rng = random.Random(0)
    for w in ["động", "ếch", "hoạt", "người", "thuở", "được", "đường"]:
        for _ in range(100):
            out = swap_tone(w, rng)
            assert not any(0x300 <= ord(c) <= 0x36F for c in out), (w, out)
