from fastapi import FastAPI
from pydantic import BaseModel
import os

# =========================
# IMPORTS LANGCHAIN MODERNOS
# =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# Importación corregida para versiones modernas de LangChain
from langchain.chains import RetrievalQA

# =========================
# 1. VALIDAR API KEY
# =========================
# Es mejor obtenerla aquí para pasarla explícitamente a los componentes
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ ERROR: La variable OPENAI_API_KEY no está configurada en Render")

# =========================
# 2. APP FASTAPI
# =========================
app = FastAPI(
    title="API NLP RAG",
    description="API de preguntas y respuestas usando LangChain + ChromaDB",
    version="1.0.0"
)

# =========================
# 3. MODELO DE REQUEST
# =========================
class QuestionRequest(BaseModel):
    question: str

# =========================
# 4. CARGA DE DOCUMENTOS
# =========================
# Usamos una ruta absoluta basada en la ubicación de este archivo para evitar errores
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "Conocimiento.txt")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo en: {DATA_PATH}")

loader = TextLoader(
    file_path=DATA_PATH,
    encoding="utf-8"
)
documents = loader.load()

# =========================
# 5. DIVISIÓN DE TEXTO
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# =========================
# 6. EMBEDDINGS Y VECTOR DB
# =========================
# Pasamos la API Key explícitamente para evitar fallos de contexto
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# =========================
# 7. MODELO LLM
# =========================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

# =========================
# 8. CADENA RAG
# =========================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# =========================
# 9. ENDPOINTS
# =========================
@app.get("/")
def health_check():
    return {"status": "API RAG activa 🚀", "python_version": "3.12.7"}

@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        # Usamos invoke que es el estándar actual
        response = qa_chain.invoke({"query": request.question})
        return {
            "question": request.question,
            "answer": response["result"]
        }
    except Exception as e:
        return {"error": str(e)}
