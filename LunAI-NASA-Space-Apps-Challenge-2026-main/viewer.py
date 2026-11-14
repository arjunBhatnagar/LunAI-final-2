"""The LunAI Streamlit page: a full-screen 3D viewer. Start the app with ``streamlit run app.py``."""
from __future__ import annotations

import streamlit as st

from lunai import services
from lunai.visuals import build_viewer_html

st.set_page_config(page_title="LunAI | Mission Map", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        height: 100vh !important; width: 100vw !important; margin: 0 !important; padding: 0 !important;
        overflow: hidden !important; background: #03070d;
      }
      header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
      section[data-testid="stSidebar"] { display: none !important; }
      .block-container { max-width: none !important; height: 100vh; padding: 0 !important; margin: 0 !important; }
      iframe { position: fixed !important; inset: 0 !important; z-index: 10000 !important;
               width: 100vw !important; height: 100vh !important; border: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading results…", ttl=3600)
def viewer_page(epoch_iso: str, models: tuple) -> str:
    del models  # only part of the cache key, so changing a model setting rebuilds the page
    return build_viewer_html(services.build_viewer_config(epoch_iso))


try:
    # The classifier and the available mission-path CSVs are part of the cache key, so changes show on refresh.
    page = viewer_page(services.current_epoch(), (services.classifier().name, tuple(services.mission_path_ids())))
except Exception as exc:  # Show model/data problems in the page instead of a blank screen.
    st.error(f"LunAI could not start: {exc}")
    st.stop()

# The page is generated locally from trusted templates and data, so same-origin embedding is fine
# (the viewer uses it to call the /api routes defined in app.py).
if hasattr(st, "iframe"):
    st.iframe(page, height=900)
else:  # Streamlit releases before st.iframe
    from streamlit.components.v1 import html

    html(page, height=900, scrolling=False)
