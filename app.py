import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# A simple check to see if the key is loaded
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Taxfix AI Assistant", layout="wide")

st.title("Taxfix Multi-Agent Assistant 🤖")

st.info("Project environment is set up and ready to go!")

if api_key:
    st.success("OpenAI API key loaded successfully.")
else:
    st.error("OpenAI API key not found. Please check your .env file.")