"""Hiển thị nguyên giao diện HTML/CSS/JavaScript gốc bên trong Streamlit."""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "Botaniq_v20.html"


def hien_thi_website():
    if not HTML_FILE.exists():
        st.error(f"Không tìm thấy {HTML_FILE.name}. Hãy đặt file này cùng thư mục app.py.")
        return

    # Ẩn phần khung mặc định của Streamlit để trang HTML chiếm toàn bộ chiều ngang.
    st.markdown(
        """
        <style>
          #MainMenu, footer, header {visibility: hidden; height: 0;}
          [data-testid="stAppViewContainer"] { background: #222a10; }
          [data-testid="stMain"] { padding: 0 !important; }
          .block-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
          iframe { border: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    html = HTML_FILE.read_text(encoding="utf-8")
    # Chạy toàn bộ HTML gốc trong iframe: CSS, JavaScript, ảnh base64, đăng nhập,
    # điều hướng và các tương tác phía trình duyệt vẫn do mã nguồn gốc điều khiển.
    components.html(html, height=1250, scrolling=True)
