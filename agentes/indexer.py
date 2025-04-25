import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = PineconeClient(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Nombre único para el índice compartido
index_name = "cv-documents"

# Crear el índice una vez si no existe
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=spec)
    time.sleep(1)

cv_config = {
    "clara": "Curriculum vitae - Bureu Clara.pdf",
    "tomas": "CV_TomasBureu.pdf",
    "pedro": "Curriculum vitae - Pedro.pdf"
}

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

for namespace, file_path in cv_config.items():
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    documents = splitter.split_documents(pages)

    vectorstore = Pinecone.from_documents(
        documents=documents,
        embedding=embedder,
        index_name=index_name,
        namespace=namespace  # Namespace por persona
    )

    print(f"✅ CV de {namespace.capitalize()} cargado en namespace '{namespace}' dentro del índice '{index_name}'")

print("Todos los documentos fueron cargados exitosamente en namespaces separados.")
