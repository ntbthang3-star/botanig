"""Botaniq: Streamlit + giao diện HTML + Random Forest tự cài đặt của nhóm."""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from mo_hinh_web import du_doan

st.set_page_config(
    page_title="Botaniq · Hệ thống hỗ trợ chọn cây trồng",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ROOT = Path(__file__).resolve().parent
botaniq_component = components.declare_component(
    "botaniq_html_frontend",
    path=str(ROOT / "botaniq_component" / "frontend"),
)
st.markdown("""<style>
#MainMenu, footer, header {visibility:hidden; height:0;}
[data-testid="stAppViewContainer"] {background:#222a10;}
[data-testid="stMain"] {padding:0 !important;}
.block-container {max-width:100% !important; padding:0 !important; margin:0 !important;}
iframe {border:0 !important;}
</style>""", unsafe_allow_html=True)

if "botaniq_result" not in st.session_state:
    st.session_state.botaniq_result = None
if "botaniq_processed_signature" not in st.session_state:
    st.session_state.botaniq_processed_signature = None
payload = botaniq_component(
    result=st.session_state.botaniq_result,
    height=1250,
    key="botaniq_main_component",
)
if isinstance(payload, dict) and payload.get("signature"):
    signature = payload["signature"]
    if signature != st.session_state.botaniq_processed_signature:
        try:
            result = du_doan(payload.get("values", []), payload.get("active", []))
            result["signature"] = signature
            st.session_state.botaniq_result = result
            st.session_state.botaniq_processed_signature = signature
        except Exception as exc:
            st.session_state.botaniq_result = {
                "signature": signature,
                "error": str(exc),
            }
            st.session_state.botaniq_processed_signature = signature
        st.rerun()
