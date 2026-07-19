import os
import pickle
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import Dual Mode modules
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

# Hybrid and Reranking imports
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

DB_DIR = "vector_db"
BM25_STORE = os.path.join(DB_DIR, "bm25_docs.pkl")

llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.1
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert DocuMind AI Assistant.

Answer the user's question using only the provided context.

Rules:
1. If the exact answer is present in the context, answer clearly.
2. If the exact answer is not present, state clearly that you cannot find the exact information. However, if the context contains closely related topics, you MUST summarize this related information to be helpful.
3. If neither the exact answer nor related information is in the context, say: "I could not find this information in the provided documents."
3. GUARDRAIL: If the user asks a question that is completely unrelated to the provided documents, politely inform them that you are a specialized DocuMind AI Assistant and cannot answer off-topic questions.
5. Do not use outside knowledge. Do not guess.
6. Mention the source document names used.
7. Keep the answer concise and user-friendly.

Context:
{context}

Question:
{question}

Answer:
"""
)

def ask_base_model(question: str) -> str:
    prompt = f"""
You are a helpful assistant.

Answer the following question:

Question:
{question}
"""
    return llm.invoke(prompt)

def stream_base_model(question: str):
    """Returns a generator that yields the base model's answer in chunks."""
    prompt = f"""
You are a helpful assistant.

Answer the following question:

Question:
{question}
"""
    return llm.stream(prompt)


def get_base_vector_retriever():
    """Returns just the vector store retriever."""
    use_cloud = os.getenv("USE_CLOUD_SETUP", "false").lower() == "true"
    
    if use_cloud:
        # Initialize Pinecone
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
            raise ValueError("Missing Cloud API keys. Please configure your .env file or set USE_CLOUD_SETUP=false.")
            
        pc = Pinecone(api_key=PINECONE_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        vector_store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    else:
        # Initialize Local ChromaDB
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
    # We fetch more docs initially for the reranker (e.g. k=15)
    return vector_store.as_retriever(search_kwargs={"k": 15})

def get_bm25_retriever():
    """Loads chunks from pickle and returns a BM25 keyword retriever."""
    if os.path.exists(BM25_STORE):
        try:
            with open(BM25_STORE, "rb") as f:
                docs = pickle.load(f)
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 15 # Fetch top 15 keyword matches
                return bm25_retriever
        except Exception as e:
            print(f"Failed to load BM25 store: {e}")
            
    # If no BM25 store exists, return None
    return None

def get_advanced_retriever():
    """Builds the hybrid search + reranking pipeline."""
    vector_retriever = get_base_vector_retriever()
    bm25_retriever = get_bm25_retriever()
    
    # 1. Hybrid Search (Ensemble)
    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4] # 60% Semantic, 40% Keyword
        )
        base_retriever = ensemble_retriever
    else:
        base_retriever = vector_retriever
        
    # 2. Reranker (Cross Encoder)
    # Using a fast and highly accurate local BAAI model
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    # Take the top 4 most relevant chunks from the ensemble's pool
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)
    
    # Combine into a single retrieval pipeline
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever


def ask_rag_model(question: str) -> dict:
    retriever = get_advanced_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    final_prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    answer = llm.invoke(final_prompt)
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_context": context
    }

def stream_rag_model(question: str) -> dict:
    """Returns a dictionary containing a generator for the streaming answer, plus sources and context."""
    retriever = get_advanced_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [f"Document Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs]
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer_stream = chain.stream({"context": context, "question": question})
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    return {
        "answer_stream": answer_stream,
        "sources": sources,
        "retrieved_context": context
    }


if __name__ == "__main__":
    question = "What are the key capabilities outlined in the project overview?"

    print("Base Model Answer:")
    for chunk in stream_base_model(question):
        print(chunk, end="", flush=True)
    print("\n")

    print("RAG Optimized Answer (Hybrid + Reranked):")
    try:
        result = stream_rag_model(question)
        for chunk in result["answer_stream"]:
            print(chunk, end="", flush=True)
        print("\nSources:", result["sources"])
    except Exception as e:
        print(f"Error: {e}")
