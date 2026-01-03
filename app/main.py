from fastapi import FastAPI
from pydantic import BaseModel
import os

# Imports de versiones estables
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "Conocimiento.txt")

# 1. Cargar y dividir
loader = TextLoader(DATA_PATH, encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 2. Base de datos y Cadena
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

@app.get("/")
def home():
    return {"status": "Corriendo en versión estable 🚀"}

@app.post("/chat")
def chat(request: QuestionRequest):
    # En estas versiones se usa .run() o .__call__()
    response = qa_chain.run(request.question)
    return {"answer": response}
