import streamlit as st
from agent.graph import run_agent

st.set_page_config(page_title="Kafka AI Study Agent", layout="wide")

st.title("📘 Kafka Streaming AI Agent")
st.write("Ask anything about Apache Kafka")

query = st.text_input("Enter your Kafka question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            answer = run_agent(query)
        st.markdown(answer)
    else:
        st.warning("Please enter a question")
