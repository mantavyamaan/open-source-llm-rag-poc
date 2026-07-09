# Indian Constitution Helper LLM (RAG POC)

## Overview
This Proof of Concept (POC) demonstrates how a general-purpose, open-source Large Language Model (LLM) can be optimized for a specific use case—an **Indian Constitution Helper LLM**. 

Instead of relying on computationally expensive fine-tuning, this project utilizes **Retrieval-Augmented Generation (RAG)** to ground the LLM in constitutional and legal documents. This approach significantly improves factual accuracy, reduces hallucinations, allows for explicit source citations, and keeps data private by running the model locally.

## Features
* **Dual-Mode Architecture:** Run the entire system locally on your laptop, or instantly flip a switch to process massive 100 GB+ datasets using Enterprise Cloud infrastructure (Amazon S3 + Pinecone + OpenAI Embeddings).
* **Local Open-Source AI:** Uses `Qwen2.5:7b` via Ollama for fast, secure, and private on-device text generation.
* **Real-time UI Streaming:** Answers are streamed word-by-word into the UI (just like ChatGPT) for zero perceived latency.
* **Chat Logging:** All questions, generated answers, and source citations are silently logged to an SQLite database (`chat_history.db`) for auditing.
* **Source Citation:** The optimized model explicitly cites the exact constitutional document used to generate its answer.
* **Hallucination Mitigation:** Strict prompt engineering ensures the model refuses to answer if the information is not present in the provided legal documents.
* **Automated Evaluation Matrix:** Includes a benchmarking script that tests for Factual Accuracy, Hallucination Prevention, and Source Grounding. The results are displayed live in a beautiful matrix in the Streamlit UI.
* **Interactive UI:** Built with Streamlit for a clean, user-friendly chat interface with a sidebar for Document Management and Live Analytics.

## Tech Stack
* **Language:** Python 3.10+
* **LLM Engine:** Ollama (`qwen2.5:7b`)
* **Orchestration:** LangChain
* **Frontend:** Streamlit
* **Local Mode Stack:** ChromaDB, `nomic-embed-text`
* **Enterprise Cloud Stack:** Amazon S3, Pinecone, OpenAI (`text-embedding-3-small`)

---

## 🚀 Setup & Installation Guide

Follow these steps to get the project running on your local machine.

### 1. Prerequisites
Before you begin, you must have [Ollama](https://ollama.com/) installed on your machine.
Once Ollama is installed, open your terminal and pull the required models:
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. Clone the Repository
```bash
git clone https://github.com/mantavyamaan/open-source-llm-rag-poc.git
cd open-source-llm-rag-poc
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install the required Python packages using the provided requirements file:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Architecture Selection (Local vs. Cloud)
This project supports two modes. You configure them via a `.env` file.
Copy the `.env.example` file to a new file named `.env`:

### Option A: Local Mode (Default)
Runs 100% locally on your laptop using ChromaDB and Ollama.
* Leave `USE_CLOUD_SETUP=false` in your `.env` file.
* Put your `.txt` files in the local `data/` folder.

### Option B: Enterprise Cloud Mode (For 100 GB+ Datasets)
Bypasses local hardware limits by streaming from Amazon S3, embedding with OpenAI, and storing in Pinecone.
* Set `USE_CLOUD_SETUP=true` in your `.env` file.
* Fill in your AWS, Pinecone, and OpenAI API keys in the `.env` file.
* Put your `.txt` files in your Amazon S3 Bucket.

---

## 🏃‍♂️ Running the Application

### 1. Ingest the Data
Open your terminal and run the ingestion pipeline. It will automatically detect whether you are in Local or Cloud mode and process your data safely in batches.
```bash
python ingest.py
```

### 2. Launch the Web Interface
Start the Streamlit application to interact with the Constitution Helper:
```bash
streamlit run app.py
```
*This will open a browser window (usually at `http://localhost:8501`).*

### 3. View Logs
All chats are securely logged in the local `chat_history.db` file.

---

## 📊 Running Evaluations (Optional)
To benchmark the accuracy of the Base Model versus the RAG Model against a predefined set of questions, run the evaluation script:
```bash
python evaluate.py
```
*This will output the accuracy percentages directly in the terminal and save detailed results to `eval/results.json`.*