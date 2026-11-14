"""LunAI server: the Streamlit page (viewer.py) plus a small HTTP API for the 3D viewer.

Run with:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from lunai.api import routes

app = st.App(str(Path(__file__).with_name("viewer.py")), routes=routes)

if __name__ == "__main__":
    app.run()
