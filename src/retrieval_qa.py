import json
from pathlib import Path
from langchain_core.vectorstores import Chroma
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

corpus_path = Path("./data/corpus.jsonl")
docs = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

texts = [d["text"] for d in docs]
metadatas = [
    {"program": d["program"], "source": d["source"], "block_id": d["block_id"]}
    for d in docs
]


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(
    texts, embeddings, metadatas=metadatas, persist_directory="./chroma_db"
)

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.1, "max_length": 512}
)

qa = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True
)

if __name__ == "__main__":
    while True:
        query = input("Введите вопрос (или 'exit'): ")
        if query.strip().lower() == "exit":
            break
        result = qa({"query": query})
        print("Ответ:\n", result["result"])
        print("\nИсточник:")
        for doc in result["source_documents"]:
            print(
                f"[{doc.metadata['program']}/{doc.metadata['source']}] {doc.page_content[:200]}..."
            )
