from fastapi import FastAPI
from pydantic import BaseModel
import os

# Imports compatibles con LangChain 0.1.0 y Python 3.12
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI(title="Chatbot RAG - Fase 4 UNAD")

class QuestionRequest(BaseModel):
    question: str

# 1. Configuración de API Key desde las variables de entorno de Render
api_key = os.getenv("OPENAI_API_KEY")

# 2. Configuración de Rutas (Ajustada a tu estructura de GitHub)
# BASE_DIR es la carpeta 'app'. PROJECT_ROOT es la raíz donde está 'Conocimiento.txt'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "Conocimiento.txt")

# Variable global para la cadena de QA
qa_chain = None

# 3. Inicialización del Motor de Inteligencia (RAG)
if os.path.exists(DATA_PATH):
    try:
        # Carga del documento (Guía de actividades Fase 4)
        loader = TextLoader(DATA_PATH, encoding="utf-8")
        documents = loader.load()
        
        # División de texto (Fragmentación para mejor procesamiento)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Creación de Embeddings y Base de Datos Vectorial (ChromaDB)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Configuración de la cadena de recuperación (RetrievalQA)
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        print("✅ Base de conocimientos cargada con éxito desde la raíz")
        
    except Exception as e:
        print(f"❌ Error al inicializar RAG: {e}")
else:
    print(f"⚠️ ADVERTENCIA: No se encontró el archivo en {DATA_PATH}. El chatbot no tendrá contexto.")

# 4. Endpoints de la API
@app.get("/")
def health_check():
    """Verifica si la API y el archivo de conocimiento están listos"""
    estado_rag = "Activo ✅" if qa_chain else "Archivo no encontrado ❌"
    return {
        "proyecto": "NLP Fase 4 - UNAD",
        "status": "Online",
        "base_conocimiento": estado_rag,
        "ruta_buscada": DATA_PATH
    }

@app.post("/chat")
def chat(request: QuestionRequest):
    """Procesa preguntas usando la base de conocimientos"""
    if not qa_chain:
        return {"respuesta": "Lo siento, mi base de conocimientos no está disponible en este momento."}
    
    try:
        # Ejecución de la consulta
        response = qa_chain.invoke({"query": request.question})
        return {
            "pregunta": request.question,
            "respuesta": response["result"]
        }
    except Exception as e:
        return {"error": str(e)}
