"""Huấn luyện và dự đoán bằng thuật toán Random Forest tự cài đặt của nhóm."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from thuat_toan import MyRandomForest

ROOT = Path(__file__).resolve().parent
# Thứ tự chỉ số trong giao diện HTML: N, P, K, pH, nhiệt độ, độ ẩm, lượng mưa.
UI_FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
# Bộ dữ liệu CSV có thứ tự cột khác giao diện; dùng tên cột để chọn đúng đặc trưng.
CROP_UI_NAMES = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

def _normalize(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())

@lru_cache(maxsize=32)
def _fit_for_features(feature_tuple):
    """Mỗi tổ hợp đặc trưng được chọn có một RF riêng, học từ train.csv."""
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    features = list(feature_tuple)
    X_train = train[features].astype(float)
    y_train = train["Crop"].astype(str)
    X_test = test[features].astype(float)
    y_test = test["Crop"].astype(str)

    model = MyRandomForest(
        n_estimators=50,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        vote="soft",
        random_state=42,
    )
    model.fit(X_train, y_train)
    accuracy = float(np.mean(model.predict(X_test) == y_test.to_numpy()))
    return model, accuracy

def du_doan(values, active_indices):
    """Trả xác suất theo đúng thứ tự 22 cây trên giao diện Botaniq."""
    active = sorted({int(i) for i in active_indices})
    if len(active) < 2:
        raise ValueError("Vui lòng chọn ít nhất 2 chỉ số để dự đoán.")
    if any(i < 0 or i >= len(UI_FEATURES) for i in active):
        raise ValueError("Danh sách chỉ số đầu vào không hợp lệ.")
    values = [float(x) for x in values]
    if len(values) != 7:
        raise ValueError("Cần gửi đủ 7 giá trị từ giao diện.")

    features = tuple(UI_FEATURES[i] for i in active)
    model, accuracy = _fit_for_features(features)
    sample = pd.DataFrame([[values[i] for i in active]], columns=list(features))
    raw = model.predict_proba(sample)[0]
    classes = [_normalize(x) for x in model.classes_]
    by_class = {name: float(raw[j]) for j, name in enumerate(classes)}

    probabilities = [by_class.get(name, 0.0) for name in CROP_UI_NAMES]
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    return {
        "probabilities": probabilities,
        "accuracy": accuracy,
        "features": list(features),
        "n_estimators": model.n_estimators,
        "criterion": model.criterion,
    }
