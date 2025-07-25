import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()
openapi_key = os.getenv("OPENAI_API_KEY")
if not openapi_key:
    print("Set OPEN_API_KEY in your environment.")
    sys.exit(1)

data_dir = Path("./data")
documents = []
for pdf_file in data_dir.glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())
for txt_file in data_dir.glob("*.txt"):
    loader = TextLoader(str(txt_file), encoding="utf-8")
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

user_histories = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Задайте вопрос по магистратурам ИТМО.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text
    if user_id not in user_histories:
        user_histories[user_id] = []
    result = pdf_qa.invoke({"question": query, "chat_history": user_histories[user_id]})
    user_histories[user_id].append((query, result["answer"]))
    await update.message.reply_text(result["answer"])


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in your environment.")
        sys.exit(1)
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
