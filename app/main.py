from fastapi import FastAPI
from pydantic import BaseModel
import os

# =========================
# IMPORTS COMPATIBLES (Versión 0.1.0)
# =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# =========================
# 1. CONFIGURACIÓN DE RUTAS Y API
# =========================
# Validar API KEY
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY no configurada en Render")

# Configurar rutas para encontrar el archivo .txt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Sube un nivel y busca en la carpeta data
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "Conocimiento.txt")

# =========================
# 2. INICIALIZACIÓN DEL MOTOR RAG
# =========================
# Cargar documento
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo en: {DATA_PATH}")

loader = TextLoader(file_path=DATA_PATH, encoding="utf-8")
documents = loader.load()

# Dividir texto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings y Base de Datos (en memoria para esta versión)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Crear la cadena de respuesta
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# =========================
# 3. API FASTAPI
# =========================
app = FastAPI(title="Chatbot RAG")

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "API RAG funcionando con versiones estables 🚀"}

@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        # En esta versión de LangChain usamos .invoke o .run
        response = qa_chain.invoke({"query": request.question})
        return {
            "pregunta": request.question,
            "respuesta": response["result"]
        }
    except Exception as e:
        return {"error": str(e)}
