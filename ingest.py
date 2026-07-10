import os
import boto3
import tempfile
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

DB_DIR = "vector_db"
DATA_DIR = "data"

def load_document(filepath):
    """Routes the file to the correct document loader based on extension."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(filepath)
        elif ext == '.pptx':
            loader = UnstructuredPowerPointLoader(filepath)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(filepath)
        elif ext == '.txt':
            loader = TextLoader(filepath, encoding='utf-8')
        else:
            print(f"Unsupported file extension {ext} for {filepath}")
            return []
        
        return loader.load()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def ingest_single_file_local(filepath, filename):
    """Ingests a single file into the local Chroma vector database."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    
    docs = load_document(filepath)
    if not docs:
        return False
        
    for doc in docs:
        doc.metadata["source"] = filename
        
    split_chunks = splitter.split_documents(docs)
    if split_chunks:
        BATCH_SIZE = 100
        for i in range(0, len(split_chunks), BATCH_SIZE):
            batch = split_chunks[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
    return True

def ingest_single_file_cloud(filepath, filename):
    """Ingests a single file into the cloud Pinecone vector database."""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY]):
        print("Missing cloud API keys.")
        return False

    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)

    docs = load_document(filepath)
    if not docs:
        return False
        
    for doc in docs:
        doc.metadata["source"] = filename
        
    split_chunks = splitter.split_documents(docs)
    if split_chunks:
        BATCH_SIZE = 100
        for i in range(0, len(split_chunks), BATCH_SIZE):
            batch = split_chunks[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
    return True

def create_local_vector_database():
    print("Running in LOCAL MODE...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}/ directory. Please place your large files there.")
        return

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.txt', '.pdf', '.docx', '.pptx'))]
    if not files:
        print(f"No supported documents found in {DATA_DIR}/ directory.")
        return

    total_chunks_added = 0

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename} locally...")
        
        docs = load_document(filepath)
        if not docs:
            continue
            
        for doc in docs:
            doc.metadata["source"] = filename
            
        split_chunks = splitter.split_documents(docs)
        
        if split_chunks:
            BATCH_SIZE = 100
            for i in range(0, len(split_chunks), BATCH_SIZE):
                batch = split_chunks[i:i + BATCH_SIZE]
                vector_store.add_documents(batch)
                
            total_chunks_added += len(split_chunks)
            print(f"  -> Added {len(split_chunks)} chunks")

    print(f"\nLocal Vector database ingestion completed! Total chunks indexed: {total_chunks_added}")


def create_cloud_vector_database():
    print("Running in ENTERPRISE CLOUD MODE...")
    # Load Environment Variables
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, OPENAI_API_KEY]):
        print("Error: Missing required cloud API keys. Please configure your .env file.")
        return

    print("Connecting to Pinecone and Amazon S3...")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)

    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].lower().endswith(('.txt', '.pdf', '.docx', '.pptx'))]
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return

    if not files:
        print(f"No supported documents found in S3 Bucket: {S3_BUCKET_NAME}")
        return

    total_chunks_added = 0

    for filename in files:
        print(f"Streaming {filename} from Amazon S3...")
        
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        body = s3_object['Body'].read()
        
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(body)
            tmp_filepath = tmp_file.name
            
        docs = load_document(tmp_filepath)
        os.remove(tmp_filepath)
        
        if not docs:
            continue
            
        for doc in docs:
            doc.metadata["source"] = filename
            
        split_chunks = splitter.split_documents(docs)
        
        if split_chunks:
            BATCH_SIZE = 100
            for i in range(0, len(split_chunks), BATCH_SIZE):
                batch = split_chunks[i:i + BATCH_SIZE]
                vector_store.add_documents(batch)
                
            total_chunks_added += len(split_chunks)
            print(f"  -> Uploaded {len(split_chunks)} chunks to Pinecone")

    print(f"\nCloud Vector database ingestion completed! Total chunks indexed: {total_chunks_added}")


if __name__ == "__main__":
    use_cloud = os.getenv("USE_CLOUD_SETUP", "false").lower() == "true"
    if use_cloud:
        create_cloud_vector_database()
    else:
        create_local_vector_database()
