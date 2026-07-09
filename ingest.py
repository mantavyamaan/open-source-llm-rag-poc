import os
import boto3
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

DB_DIR = "vector_db"
DATA_DIR = "data"

def process_file_in_chunks_local(filepath, chunk_size=5000000):
    """Generator to read a massive file in safe chunks to prevent OOM errors."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            yield text

def create_local_vector_database():
    print("Running in LOCAL MODE...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}/ directory. Please place your large files there.")
        return

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    if not files:
        print(f"No .txt files found in {DATA_DIR}/ directory.")
        return

    total_chunks_added = 0

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Processing {filename} locally...")
        
        batch_number = 1
        for text_chunk in process_file_in_chunks_local(filepath):
            doc = Document(page_content=text_chunk, metadata={"source": filename})
            split_chunks = splitter.split_documents([doc])
            
            if split_chunks:
                BATCH_SIZE = 100
                for i in range(0, len(split_chunks), BATCH_SIZE):
                    batch = split_chunks[i:i + BATCH_SIZE]
                    vector_store.add_documents(batch)
                    
                total_chunks_added += len(split_chunks)
                print(f"  -> Added batch {batch_number} ({len(split_chunks)} chunks)")
            
            batch_number += 1

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
        files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.txt')]
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return

    if not files:
        print(f"No .txt files found in S3 Bucket: {S3_BUCKET_NAME}")
        return

    total_chunks_added = 0

    for filename in files:
        print(f"Streaming {filename} from Amazon S3...")
        
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=filename)
        body = s3_object['Body']
        
        batch_number = 1
        for text_chunk_bytes in body.iter_chunks(chunk_size=5000000):
            text_chunk = text_chunk_bytes.decode('utf-8', errors='ignore')
            doc = Document(page_content=text_chunk, metadata={"source": filename})
            split_chunks = splitter.split_documents([doc])
            
            if split_chunks:
                BATCH_SIZE = 100
                for i in range(0, len(split_chunks), BATCH_SIZE):
                    batch = split_chunks[i:i + BATCH_SIZE]
                    vector_store.add_documents(batch)
                    
                total_chunks_added += len(split_chunks)
                print(f"  -> Uploaded batch {batch_number} to Pinecone ({len(split_chunks)} chunks)")
            
            batch_number += 1

    print(f"\nCloud Vector database ingestion completed! Total chunks indexed: {total_chunks_added}")


if __name__ == "__main__":
    use_cloud = os.getenv("USE_CLOUD_SETUP", "false").lower() == "true"
    if use_cloud:
        create_cloud_vector_database()
    else:
        create_local_vector_database()
