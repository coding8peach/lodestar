"""Lodestar UI entry point: `uv run lodestar ui` runs `streamlit run` on this file."""

import streamlit as st
from dotenv import load_dotenv

from lodestar.paths import PROJECT_ROOT
from lodestar.quiet import silence
from lodestar.ui.pages import decisions_page, queue_page, review_page, usage_page

load_dotenv(PROJECT_ROOT / ".env")
silence()
st.set_page_config(page_title="Lodestar", layout="wide")
# Tighter than Streamlit's defaults: less space above the content, smaller page titles.
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
  h1 { font-size: 1.9rem !important; padding-top: 0 !important; padding-bottom: 0.2rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; line-height: 1.3; }
  [data-testid="stMetricLabel"] p { font-size: 0.85rem; }
  [data-testid="stMetric"] { padding: 0; }
</style>
""", unsafe_allow_html=True)

from lodestar.app.service import demo_mode  # noqa: E402

if demo_mode():
    st.info("**Demo:** a fictional candidate and fictional job postings, analyzed by the real agents. "
            "Actions that call an LLM, fetch job boards or change data are disabled. "
            "Source: [github.com/coding8peach/lodestar](https://github.com/coding8peach/lodestar)")

st.navigation([
    st.Page(review_page, title="Review", url_path="review", default=True),
    st.Page(queue_page, title="Queue", url_path="queue"),
    st.Page(decisions_page, title="Decisions", url_path="decisions"),
    st.Page(usage_page, title="Usage", url_path="usage"),
]).run()
