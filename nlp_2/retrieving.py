import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Cargar variables de entorno
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Inicializar modelos
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore_cv = PineconeVectorStore(
    index_name="clara-bureu",
    embedding=embed_model,
    namespace="espacio"
)
retriever = vectorstore_cv.as_retriever()

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
# Streamlit UI
st.set_page_config(page_title="RAG Clara Bureu", page_icon="📄")
st.title("📄 Analizador de CV")

query = st.text_input("Escribí tu pregunta:")

if query:
    result = qa.invoke(query)
    st.subheader("🔍 Respuesta:")
    st.write(result['result'])

    with st.expander("📝 Fuente"):
        for doc in result['source_documents']:
            st.markdown(f"**Página:** {doc.metadata.get('page', 'N/A')}")
            st.write(doc.page_content)

# %%
