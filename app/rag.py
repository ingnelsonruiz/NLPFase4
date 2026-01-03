from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
# CAMBIO 1: Importar desde el paquete de splitters especializado
from langchain_text_splitters import RecursiveCharacterTextSplitter
# CAMBIO 2: Importación directa y limpia de RetrievalQA
from langchain.chains import RetrievalQA
import os

def build_rag_chain():
    # Validar API KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY no encontrada en las variables de entorno")

    # Modelo
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key
    )

    # Cargar documento (Asegúrate que la carpeta data existe)
    loader = TextLoader(
        file_path="data/Conocimiento.txt",
        encoding="utf-8"
    )
    documents = loader.load()

    # Dividir texto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Vector DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        return_source_documents=False
    )

    return qa_chain
