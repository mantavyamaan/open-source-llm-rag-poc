import os
import streamlit as st
import db
import json
from rag import ask_base_model, ask_rag_model, stream_base_model, stream_rag_model
from ingest import ingest_single_file_local, ingest_single_file_cloud

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
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if use_cloud:
        st.info("💡 **Enterprise Cloud Setup**\nDocuments are now securely managed via Amazon S3. Configure your `.env` file.")
        
        uploaded_file = st.file_uploader("Upload Constitutional Document", type=["txt", "pdf", "docx", "pptx"])
        if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
            import boto3
            import tempfile
            st.info("Uploading and indexing to Cloud...")
            AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
            AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
            AWS_REGION = os.getenv("AWS_REGION")
            S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
            
            try:
                s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_filepath = tmp_file.name
                
                s3_client.upload_file(tmp_filepath, S3_BUCKET_NAME, uploaded_file.name)
                
                with st.spinner(f"Ingesting {uploaded_file.name} to Pinecone..."):
                    success = ingest_single_file_cloud(tmp_filepath, uploaded_file.name)
                
                os.remove(tmp_filepath)
                if success:
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✅ Automated Ingestion Complete for `{uploaded_file.name}`!")
            except Exception as e:
                st.error(f"Cloud ingestion failed: {e}")

        st.markdown("---")
        st.subheader("📚 Currently Stored Documents")
        st.write("Fetching documents directly from Amazon S3 is disabled in the UI for performance. Please check your S3 bucket directly.")
    else:
        
        uploaded_file = st.file_uploader("Upload Constitutional Document", type=["txt", "pdf", "docx", "pptx"])
        if uploaded_file is not None and uploaded_file.name not in st.session_state.processed_files:
            if not os.path.exists("data"):
                os.makedirs("data")
            filepath = os.path.join("data", uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner(f"Ingesting {uploaded_file.name} into local knowledge base..."):
                success = ingest_single_file_local(filepath, uploaded_file.name)
                if success:
                    st.session_state.processed_files.add(uploaded_file.name)
                    st.success(f"✅ Automated Ingestion Complete for `{uploaded_file.name}`!")
                else:
                    st.error(f"Failed to ingest {uploaded_file.name}")
            
        st.markdown("---")
        st.subheader("📚 Currently Stored Documents")
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.lower().endswith(('.txt', '.pdf', '.docx', '.pptx'))]
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
            import pandas as pd
            df = pd.DataFrame({
                "Metric": ["Accuracy", "Hallucinations", "Sourced"],
                "Base": [
                    matrix.get("Factual_Accuracy", {}).get("Base", "0%"),
                    matrix.get("Hallucination_Rate", {}).get("Base", "0%"),
                    matrix.get("Source_Grounding", {}).get("Base", "0%")
                ],
                "RAG": [
                    matrix.get("Factual_Accuracy", {}).get("RAG", "0%"),
                    matrix.get("Hallucination_Rate", {}).get("RAG", "0%"),
                    matrix.get("Source_Grounding", {}).get("RAG", "0%")
                ]
            })
            st.table(df.set_index("Metric"))
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
