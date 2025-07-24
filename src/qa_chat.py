import json
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

corpus_path = Path("./data/corpus.jsonl")
docs = []
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        docs.append(doc)

documents = [
    Document(
        page_content=item["text"],
        metadata={
            "program": item["program"],
            "source": item["source"],
            "block_id": item["block_id"],
        },
    )
    for item in docs
]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(docs_split, embedding_model)

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    temperature=0.1,
    max_length=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
)

if __name__ == "__main__":
    while True:
        query = input("Введите вопрос (или 'exit'): ")
        if query.strip().lower() == "exit":
            break
        result = qa_chain(query)
        print("Ответ:\n", result["result"])
        print("\nИсточники:")
        for doc in result["source_documents"]:
            meta = doc.metadata
            print(
                f"[{meta.get('program','')}/{meta.get('source','')}] {doc.page_content[:200]}..."
            )
