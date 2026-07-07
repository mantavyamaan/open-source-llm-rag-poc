from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


DB_DIR = "vector_db"

llm = OllamaLLM(
    model="qwen2.5:7b",
    temperature=0.1
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an internal HR policy assistant.

Answer the user's question using only the provided context.

Rules:
1. If the answer is present in the context, answer clearly.
2. If the answer is not present in the context, say:
   "I could not find this information in the provided documents."
3. Do not use outside knowledge.
4. Do not guess.
5. Mention the source document names used.
6. Keep the answer concise.

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


def ask_rag_model(question: str) -> dict:
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

    sources = list(
        set([doc.metadata.get("source", "unknown") for doc in docs])
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_context": context
    }


if __name__ == "__main__":
    question = "How many casual leaves are allowed per year?"

    print("Base Model Answer:")
    print(ask_base_model(question))

    print("\nRAG Optimized Answer:")
    result = ask_rag_model(question)
    print(result["answer"])
    print("Sources:", result["sources"])
