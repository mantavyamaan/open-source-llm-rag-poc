import os
import streamlit as st
import db
import json
from rag import ask_base_model, ask_rag_model, stream_base_model, stream_rag_model

st.set_page_config(
    page_title="Indian Constitution Helper LLM",
    page_icon="⚖️",
    layout="wide"
)

# Detect Mode
use_cloud = os.getenv("USE_CLOUD_SETUP", "false").lower() == "true"

st.title("Indian Constitution Helper LLM")
st.subheader("Indian Constitution Helper LLM (Qwen + RAG)")

if use_cloud:
    st.success("☁️ **Running in Enterprise Cloud Mode** (Amazon S3 + Pinecone + OpenAI)")
else:
    st.info("💻 **Running in Local Mode** (Local Folder + ChromaDB + Ollama)")

st.write(
    """
This POC compares a base open-source LLM with a RAG-optimized version.
The optimized version retrieves relevant constitutional documents before answering.
"""
)

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📄 Document Management")
    st.write("Upload constitutional documents here. The AI will automatically ingest them.")
    
    if use_cloud:
        st.info("💡 **Enterprise Cloud Setup**\nDocuments are now securely managed via Amazon S3. Configure your `.env` file and run `python ingest.py` to stream updates into Pinecone.")
        st.markdown("---")
        st.subheader("📚 Currently Stored Policies")
        st.write("Fetching policies directly from Amazon S3 is disabled in the UI for performance. Please check your S3 bucket directly.")
    else:
        st.info("💡 **Local Data Mode**\nYou can upload files here, or place them directly into the `data/` folder. Then run `python ingest.py` in your terminal.")
        
        uploaded_file = st.file_uploader("Upload Constitutional Document", type=["txt"])
        if uploaded_file is not None:
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved `{uploaded_file.name}` to data/ folder! Now run `python ingest.py` in your terminal.")
            
        st.markdown("---")
        st.subheader("📚 Currently Stored Policies")
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith('.txt')]
            if files:
                for f in files:
                    st.write(f"- {f}")
            else:
                st.write("No documents in data/ yet.")
        else:
            st.write("Directory data/ does not exist.")

    st.markdown("---")
    st.subheader("📜 Chat Logs")
    
    history = db.get_all_history()
    if history:
        with st.expander("View Past Conversations"):
            for entry in history:
                st.markdown(f"**Q: {entry['question']}**")
                st.markdown(f"*A: {entry['rag_answer']}*")
                st.caption(f"Sources: {entry['sources']} | {entry['timestamp']}")
                st.divider()
    else:
        st.write("No history yet.")

    st.markdown("---")
    st.subheader("📊 Evaluation Matrix")
    try:
        with open("eval/results.json", "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        
        matrix = eval_data.get("matrix", {})
        if matrix:
            st.markdown(
                f"""
| Metric | Base | RAG |
|---|---|---|
| **Accuracy** | {matrix.get('Factual_Accuracy', {}).get('Base', '0%')} | {matrix.get('Factual_Accuracy', {}).get('RAG', '0%')} |
| **Hallucinations** | {matrix.get('Hallucination_Rate', {}).get('Base', '0%')} | {matrix.get('Hallucination_Rate', {}).get('RAG', '0%')} |
| **Sourced** | {matrix.get('Source_Grounding', {}).get('Base', '0%')} | {matrix.get('Source_Grounding', {}).get('RAG', '0%')} |
                """
            )
        else:
            st.caption("No matrix data found.")
    except Exception:
        st.caption("Run evaluate.py to generate matrix.")

# --- Main App ---
question = st.text_input(
    "Ask a question about the Indian Constitution:",
    placeholder="Example: What are the fundamental rights of a citizen?"
)

if st.button("Ask") and question:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Base Open-source LLM")
        base_answer = st.write_stream(stream_base_model(question))

    with col2:
        st.markdown("### RAG-Optimized LLM")
        spinner_text = "Retrieving context from Pinecone..." if use_cloud else "Retrieving context from Local ChromaDB..."
        with st.spinner(spinner_text):
            try:
                rag_result = stream_rag_model(question)
                rag_answer = st.write_stream(rag_result["answer_stream"])

                st.markdown("### Sources")
                for source in rag_result["sources"]:
                    st.write(f"- {source}")
            except ValueError as e:
                st.error(f"Configuration Error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

    with st.expander("Retrieved Context"):
        st.text(rag_result["retrieved_context"])
        
    db.save_interaction(
        question=question,
        base_answer=base_answer,
        rag_answer=rag_answer,
        sources=rag_result["sources"]
    )
