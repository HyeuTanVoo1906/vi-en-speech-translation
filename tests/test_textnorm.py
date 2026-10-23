from common.textnorm import normalize_text, unify_tone

def test_normalize_colos():
    raw = "Thì ngoài ra, ở cho chúng ta thấy là cái hàm lượng dinh dưỡng so với Alpha lipid ở trên một lon có cùng trọng lượng là 450 g và sữa Colos IgGOLD cũng là 450 g."
    want = "thì ngoài ra ở cho chúng ta thấy là cái hàm lượng dinh dưỡng so với alpha lipid ở trên một lon có cùng trọng lượng là 450 g và sữa colos iggold cũng là 450 g"
    assert normalize_text(raw) == want

def test_normalize_estradiol():
    raw = "Thông thường, hormone này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp. Estradiol E2, các chuyên gia đánh giá Estradiol là loại estrogen mạnh nhất."
    want = "thông thường hormone này sẽ phân bố trong hầu hết các mô nhưng chủ yếu là mỡ và cơ bắp estradiol e2 các chuyên gia đánh giá estradiol là loại estrogen mạnh nhất"
    assert normalize_text(raw) == want

def test_normalize_estrogen():
    raw = "Tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng, hỗ trợ kích thích sự phát triển của nang trứng."
    want = "tác dụng của estrogen có thể ảnh hưởng đến những cơ quan như buồng trứng hỗ trợ kích thích sự phát triển của nang trứng"
    assert normalize_text(raw) == want

def test_unify_tone():
    assert unify_tone("Văn hóa và thủy điện, khỏe, tòa án, quý, quỳ") == "Văn hoá và thuỷ điện, khoẻ, toà án, quý, quỳ"
    assert unify_tone("hoá thuỷ") == "hoá thuỷ"
