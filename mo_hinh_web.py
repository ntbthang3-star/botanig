"""Dự đoán cây trồng bằng Random Forest tự cài đặt của nhóm (thuat_toan.py).

Web dùng ĐÚNG các mô hình đã chọn ở notebook:
  - RF4 (4 chỉ số đất N, P, K, pH): tham số tìm ở notebook 02, bỏ phiếu mềm.
  - RF7 (đủ 7 chỉ số): tham số tìm ở notebook 04b mục 7, bỏ phiếu cứng.
Hai mô hình này được nạp từ crop_models.pkl (xuất ở notebook 04b, mục 18).
Nếu không nạp được file đó thì huấn luyện lại từ train.csv với đúng tham số (cùng seed nên cho cùng kết quả).
Với bộ chỉ số khác (người dùng bỏ bớt chỉ số), huấn luyện một rừng cùng tham số RF7 (max_features
được cắt không vượt quá số chỉ số), giống mục 16a của notebook 04b, rồi lưu cache trong bộ nhớ.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from thuat_toan import MyRandomForest

ROOT = Path(__file__).resolve().parent

# Thứ tự chỉ số trong giao diện HTML.
UI_FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]
# Thứ tự cột dùng khi huấn luyện ở notebook và trong crop_models.pkl (khác thứ tự giao diện).
MODEL_ORDER = ["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]
SOIL = ("N", "P", "K", "pH")

# Thứ tự 22 cây trên giao diện Botaniq (tên nhãn trong dữ liệu).
CROP_UI_NAMES = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

# Tham số đã chọn ở notebook (dùng khi phải huấn luyện lại).
PARAMS_RF4 = dict(n_estimators=100, criterion="entropy", max_depth=8, min_samples_split=2,
                  min_samples_leaf=3, max_features=1, vote="soft", random_state=42)
PARAMS_RF7 = dict(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features=3, vote="hard", random_state=42)

N_THU_ON_DINH, MUC_THU_ON_DINH = 20, 0.10


def _normalize(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _data():
    return pd.read_csv(ROOT / "train.csv"), pd.read_csv(ROOT / "test.csv")


@lru_cache(maxsize=1)
def _saved_models():
    import joblib

    path = ROOT / "crop_models.pkl"
    data = joblib.load(path)

    print("Đã đọc file:", path)
    print("Các khóa:", list(data["models"].keys()))

    return dict(data["models"])

def _params(features):
    if tuple(features) == SOIL:
        return dict(PARAMS_RF4)
    p = dict(PARAMS_RF7)
    p["max_features"] = min(p["max_features"], len(features))
    return p


@lru_cache(maxsize=32)
def _model_for(features):
    """features: tuple tên chỉ số theo MODEL_ORDER. Trả về (mô hình, Accuracy trên test.csv, nguồn)."""
    train, test = _data()
    cols = list(features)
    models = _saved_models()
    model = models.get(features)
    nguon = "crop_models.pkl"
    if model is None:
        model = MyRandomForest(**_params(features)).fit(train[cols].astype(float), train["Crop"].astype(str))
        nguon = "huấn luyện từ train.csv"
    accuracy = float(np.mean(model.predict(test[cols].astype(float)) == test["Crop"].astype(str).to_numpy()))
    return model, accuracy, nguon


@lru_cache(maxsize=1)
def _train_range():
    train, _ = _data()
    return {c: (float(train[c].min()), float(train[c].max())) for c in UI_FEATURES}


def du_doan(values, active_indices):
    """values: 7 giá trị theo thứ tự giao diện; active_indices: chỉ số (theo giao diện) được dùng.
    Trả về xác suất theo đúng thứ tự 22 cây trên giao diện, kèm Accuracy test của mô hình đã dùng."""
    active = sorted({int(i) for i in active_indices})
    if len(active) < 2:
        raise ValueError("Vui lòng chọn ít nhất 2 chỉ số để dự đoán.")
    if any(i < 0 or i >= len(UI_FEATURES) for i in active):
        raise ValueError("Danh sách chỉ số đầu vào không hợp lệ.")
    values = [float(x) for x in values]
    if len(values) != 7:
        raise ValueError("Cần gửi đủ 7 giá trị từ giao diện.")

    chosen = {UI_FEATURES[i]: values[i] for i in active}
    features = tuple(f for f in MODEL_ORDER if f in chosen)      # đúng thứ tự cột của mô hình
    model, accuracy, nguon = _model_for(features)

    sample = pd.DataFrame([[chosen[f] for f in features]], columns=list(features))
    raw = model.predict_proba(sample)[0]
    classes = [_normalize(x) for x in model.classes_]
    by_class = {name: float(raw[j]) for j, name in enumerate(classes)}
    probabilities = [by_class.get(name, 0.0) for name in CROP_UI_NAMES]
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]

    # Độ ổn định: làm lệch ngẫu nhiên ±10% (20 lần), đếm số lần cây đứng đầu không đổi.
    rng = np.random.default_rng(7)
    noisy = pd.DataFrame(np.tile(sample.to_numpy(), (N_THU_ON_DINH, 1)) *
                         rng.uniform(1 - MUC_THU_ON_DINH, 1 + MUC_THU_ON_DINH, (N_THU_ON_DINH, len(features))),
                         columns=list(features))
    if "humidity" in noisy:
        noisy["humidity"] = noisy["humidity"].clip(upper=100)
    top = model.classes_[int(np.argmax(raw))]
    stability = [bool(p == top) for p in model.predict(noisy)]

    rng_train = _train_range()
    return {
        "probabilities": probabilities,
        "accuracy": accuracy,
        "features": list(features),
        "n_estimators": model.n_estimators,
        "criterion": model.criterion,
        "vote": model.vote,
        "model_source": nguon,
        "stability": stability,
        "out_of_range": [f for f in features if not rng_train[f][0] <= chosen[f] <= rng_train[f][1]],
        "train_range": {f: list(rng_train[f]) for f in UI_FEATURES},
    }
