from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Annotated, List, Dict
import operator
import re

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.tools import Tool

# Cargar variables de entorno
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-8b-8192")

# Crear vectorstores para cada CV con su propio namespace
vectorstore_clara = PineconeVectorStore(
    index_name="cv-documents",
    embedding=embed_model,
    namespace="clara"
)

vectorstore_tomas = PineconeVectorStore(
    index_name="cv-documents",
    embedding=embed_model,
    namespace="tomas"
)

vectorstore_pedro = PineconeVectorStore(
    index_name="cv-documents",
    embedding=embed_model,
    namespace="pedro"
)

# Crear retrievers para cada CV
def create_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

# Crear chains de QA para cada CV
qa_clara = create_qa_chain(vectorstore_clara)
qa_tomas = create_qa_chain(vectorstore_tomas)
qa_pedro = create_qa_chain(vectorstore_pedro)

# Funciones para cada agente
def query_clara_cv(query: str) -> str:
    print(f"[DEBUG] Querying Clara CV with namespace 'clara_cv': {query}")
    result = qa_clara.invoke(query)
    print("[DEBUG] Retrieval result:", result)
    return result['result']

def query_tomas_cv(query: str) -> str:
    result = qa_tomas.invoke(query)
    print("tool_tomas", result)
    return result['result']

def query_pedro_cv(query: str) -> str:
    result = qa_pedro.invoke(query)
    print("tool_pedro", result)
    return result['result']

clara_tool = Tool(
    name="query_clara_cv",
    func=query_clara_cv,
    description="Usa esta herramienta para consultar información del CV de Clara Bureu. La entrada debe ser una pregunta sobre su experiencia, habilidades o formación."
)

tomas_tool = Tool(
    name="query_tomas_cv",
    func=query_tomas_cv,
    description="Usa esta herramienta para consultar información del CV de Tomás Bureu. La entrada debe ser una pregunta sobre su experiencia, habilidades o formación."
)

pedro_tool = Tool(
    name="query_pedro_cv",
    func=query_pedro_cv,
    description="Usa esta herramienta para consultar información del CV de Pedro. La entrada debe ser una pregunta sobre su experiencia, habilidades o formación."
)

cv_tools = {
    "clara": clara_tool,
    "tomas": tomas_tool,
    "pedro": pedro_tool
}


class AgentState(TypedDict):
    query: str
    messages: Annotated[List[AnyMessage], operator.add]
    selected_agents: List[str]
    agent_outputs: Dict[str, str]

def detect_agents(state: AgentState) -> dict:
    query = state["query"]
    agents = [k for k in cv_tools if re.search(rf'\b{k}\b', query, re.IGNORECASE)]
    if not agents:
        agents = ["clara"]   # default: clara
    return {
        "query": query,
        "selected_agents": agents,
        "messages": state["messages"]
    }

def run_tools(state: AgentState) -> dict:
    query = state["query"]
    results = {}
    for agent in state["selected_agents"]:
        results[agent] = cv_tools[agent].invoke(query)
        print(results[agent])
    return {
        "query": query,
        "selected_agents": state["selected_agents"],
        "agent_outputs": results,
        "messages": state["messages"]
    }

# Nodo 3: Generar respuesta final con contexto de todos los agentes
def synthesize(state: AgentState) -> dict:
    query = state["query"]
    context = "\n\n".join(
        f"[{agent.upper()}]\n{output}" for agent, output in state["agent_outputs"].items()
    )
    prompt = [
        SystemMessage(content="Sos un asistente que analiza CVs. Vas a recibir respuestas de uno o múltiples agentes que consultaron sus propias fuentes. No debes generar información nueva ni agregar detalles adicionales. Debes dar una única respuesta que combine las salidas de los agentes de manera coherente y clara."),
        HumanMessage(content=f"Contexto:\n{context}\n\nPregunta:\n{query}")
    ]
    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [response],
        "finish": True  # Indicamos que el proceso ha terminado
    }

# Crear el grafo
graph = StateGraph(AgentState)
graph.add_node("detect", RunnableLambda(detect_agents))
graph.add_node("consult", RunnableLambda(run_tools))
graph.add_node("synthesize", RunnableLambda(synthesize))

graph.set_entry_point("detect")
graph.add_edge("detect", "consult")
graph.add_edge("consult", "synthesize")
graph.set_finish_point("synthesize")

multiagent_cv_graph = graph.compile()
