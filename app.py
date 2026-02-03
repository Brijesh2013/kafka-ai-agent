import os
import streamlit as st
from agent.graph import run_agent

# Show which Chroma mode is active so users can troubleshoot connectivity quickly
CHROMA_MODE = os.getenv("CHROMA_MODE", "http")
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")

st.set_page_config(page_title="Kafka AI Study Agent", layout="wide")

st.title("📘 Kafka Streaming AI Agent")
st.write("Ask anything about Apache Kafka")
st.info(f"Chroma mode: {CHROMA_MODE} — host: {CHROMA_HOST}:{CHROMA_PORT}. Set CHROMA_MODE=local to avoid requiring a Chroma HTTP server.")

query = st.text_input("Enter your Kafka question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(query)
                st.markdown(answer)
            except Exception as e:
                # Log a friendly message and show a helpful UI error without leaking internals
                import logging
                logging.getLogger(__name__).exception("Agent runtime error: %s", e)
                st.error("The agent encountered an error. Check the server logs and ensure required environment variables (OPENAI_API_KEY, CHROMA_MODE) are set correctly.")
    else:
        st.warning("Please enter a question")
