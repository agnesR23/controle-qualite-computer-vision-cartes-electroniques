import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from dotenv import load_dotenv

load_dotenv()


def get_api_url() -> str:
    """Return local or deployed inference API URL."""
    try:
        return st.secrets["API_URL"]
    except (KeyError, StreamlitSecretNotFoundError):
        return os.getenv("API_URL", "http://127.0.0.1:8000")