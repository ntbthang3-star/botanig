"""Chay thu Random Forest cua nhom tren bo test.csv."""
from mo_hinh_web import du_doan

# Vi du: 4 chi so dat (N, P, K, pH).
values = [90, 42, 43, 6.5, 20.9, 82, 202.9]
active = [0, 1, 2, 3]
result = du_doan(values, active)

print("Thuat toan: MyRandomForest (50 cay, Gini, soft voting)")
print("Dac trung:", ", ".join(result["features"]))
print(f"Accuracy tren test.csv: {result['accuracy']:.2%}")
print("Top 5 cay du doan:")
names = [
    "Lúa nước", "Ngô", "Đậu gà", "Đậu đỏ", "Đậu triều", "Đậu mốc", "Đậu xanh",
    "Đậu đen", "Đậu lăng", "Lựu", "Chuối", "Xoài", "Nho", "Dưa hấu",
    "Dưa lưới", "Táo", "Cam", "Đu đủ", "Dừa", "Bông", "Đay", "Cà phê",
]
ranked = sorted(enumerate(result["probabilities"]), key=lambda x: x[1], reverse=True)[:5]
for rank, (index, probability) in enumerate(ranked, start=1):
    print(f"{rank}. {names[index]}: {probability:.2%}")
print("Luu y: soft-voting score khong phai xac suat thanh cong ngoai dong ruong.")
