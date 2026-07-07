import os
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import db


DB_DIR = "vector_db"


def load_documents():
    db_docs = db.get_all_documents()
    documents = []

    for item in db_docs:
        doc = Document(
            page_content=item["content"],
            metadata={"source": item["filename"]}
        )
        documents.append(doc)

    return documents


def create_vector_database():
    # Clear existing vector database to rebuild it dynamically
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    documents = load_documents()
    
    if not documents:
        print("No documents found in the database. Vector DB not created.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"Loaded {len(documents)} documents from SQLite.")
    print(f"Created {len(chunks)} chunks.")
    print("Vector database rebuilt successfully.")

    return vector_store


if __name__ == "__main__":
    create_vector_database()
