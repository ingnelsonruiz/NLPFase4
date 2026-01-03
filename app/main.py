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
# IMPORTACIÓN CORRECTA:
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import RetrievalQA


# =========================
# 1. VALIDAR API KEY
# =========================
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("❌ OPENAI_API_KEY no está configurada")

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
DATA_PATH = "data/Conocimiento.txt"

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
embeddings = OpenAIEmbeddings()

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
    temperature=0
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
    return {"status": "API RAG activa 🚀"}

@app.post("/chat")
def chat(request: QuestionRequest):
    response = qa_chain.invoke({"query": request.question})
    return {
        "question": request.question,
        "answer": response["result"]
    }
