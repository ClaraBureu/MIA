import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Leer documento
file_path = os.path.join(r'nlp_2\Curriculum vitae - Bureu Clara.pdf')
loader = PyPDFLoader(file_path)
pages = loader.load()

# Dividir en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = splitter.split_documents(pages)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index_name = "clara-bureu"

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    time.sleep(1)

pc.create_index(index_name, dimension=384, metric="cosine", spec=spec)

# Embeddings y upsert
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embedder,
    namespace="espacio"
)

print("Index creado y cargado con éxito.")

# %%
