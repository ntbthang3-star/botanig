"""Tiện ích đọc dữ liệu CSV gốc của nhóm."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]


def doc_train():
    return pd.read_csv(ROOT / "train.csv")


def doc_test():
    return pd.read_csv(ROOT / "test.csv")


def danh_sach_cay():
    df = doc_train()
    return sorted(df["Crop"].dropna().astype(str).unique().tolist())
