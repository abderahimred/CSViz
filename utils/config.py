import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

def get_openai_api_key():
    """
    Get the OpenAI API key from environment variables or the session state.
    
    Order of precedence:
    1. Environment variable (OPENAI_API_KEY)
    2. Session state (for user-provided keys)
    
    Returns:
        str: The OpenAI API key or None if not found
    """
    # First check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Then check session state (user might have provided their own key)
    if not api_key and "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        api_key = st.session_state.openai_api_key
        
    return api_key 