# %%
import os
import time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore

# Load environment variables from a .env file
load_dotenv()

# Store environment variables in separate variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# %%
# Read a single document
def read_doc(file_path):
    file_loader = PyPDFLoader(file_path)
    document = file_loader.load()
    print(f"Loaded document with {len(document)} pages")
    return document


file_path = r'CV - Alex Barria.pdf'
total = read_doc(file_path)

# %%
# Split the document into chunks


def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc


documents = chunk_data(docs=total, chunk_size=300, chunk_overlap=50)
type(documents)

# %%
# Connect to Pinecone DB and manage index

pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'alex-barria'

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print("Index {} deleted".format(index_name))

# Check if index already exists (it shouldn't if this is the first time)
if index_name not in pc.list_indexes().names():
    print("Index created with the name: {}".format(index_name))
    pc.create_index(
        index_name,
        dimension=3072,  # dimensionality of text-embedding models/embedding-001
        metric='cosine',
        spec=spec
    )
else:
    print("Index with the name {} already exists".format(index_name))

# %%
# Load embedding model

embed_model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)

# %%
# Create and upsert embeddings into Pinecone

namespace = "espacio"

docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embed_model,
    namespace=namespace
)

print("Upserted values to {} index".format(index_name))

time.sleep(1)

# %%
