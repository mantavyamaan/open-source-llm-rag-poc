import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


DATA_DIR = "data"
DB_DIR = "vector_db"


def load_documents():
    documents = []

    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, file_name)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_name

            documents.extend(docs)

    return documents


def create_vector_database():
    documents = load_documents()

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

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")
    print("Vector database created successfully.")

    return vector_store


if __name__ == "__main__":
    create_vector_database()
