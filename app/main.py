from fastapi import FastAPI
from pydantic import BaseModel
import os

# Imports compatibles con LangChain 0.1.0
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# =========================
# CONFIGURACIÓN
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY no configurada")

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Si tu archivo está en la raíz/data/Conocimiento.txt
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "Conocimiento.txt")

# =========================
# MOTOR RAG
# =========================
# 1. Cargar y Dividir
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No existe: {DATA_PATH}")

loader = TextLoader(DATA_PATH, encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Vector DB y QA Chain
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# Usamos persist_directory para Chroma
vectorstore = Chroma.from_documents(
    chunks, 
    embeddings, 
    persist_directory="./chroma_db"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# =========================
# API
# =========================
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "RAG Online 🚀"}

@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        # En esta versión 0.1.0 usamos .invoke() o .run()
        result = qa_chain.invoke({"query": request.question})
        return {"answer": result["result"]}
    except Exception as e:
        return {"error": str(e)}
