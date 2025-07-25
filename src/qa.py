import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

documents = []
loader = PyPDFLoader("./data/ai_plan.pdf")
documents.extend(loader.load())
loader = PyPDFLoader("./data/ai_product_plan.pdf")
documents.extend(loader.load())

loader = TextLoader("./data/ai_page.txt", encoding="utf-8")
documents.extend(loader.load())
loader = TextLoader("./data/ai_product_page.txt", encoding="utf-8")
documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents, embedding=OpenAIEmbeddings(), persist_directory="./data"
)

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.1, model_name="gpt-4.1-mini"),
    retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    verbose=False,
)

chat_history = []
print(
    "---------------------------------------------------------------------------------"
)
while True:
    query = input("Question: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print("Exiting")
        sys.exit()
    if query == "":
        continue
    result = pdf_qa.invoke({"question": query, "chat_history": chat_history})
    print("Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))
