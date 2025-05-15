# --- IMPORTS Y CONFIGURACION ---
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Dict

from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
import sqlite3, os, datetime, random

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- CONEXIÓN A DB ---
conn = None

def get_connection():
    global conn
    if conn is None:
        conn = sqlite3.connect("data.db", check_same_thread=False)
    return conn

def create_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            first_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE,
            phone TEXT
        )""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PurchaseHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date_of_purchase TEXT,
            item_id INTEGER,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )""")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            price REAL NOT NULL
        );""")

    conn.commit()

def initialize_database():
    create_database()

    users = [
        (1, "Alice", "Smith", "alice@test.com", "123-456-7890"),
        (2, "Bob", "Johnson", "bob@test.com", "234-567-8901"),
    ]
    for u in users:
        add_user(*u)

    purchases = [
        (1, "2024-01-01", 101, 99.99),
        (2, "2023-12-25", 100, 39.99),
    ]
    for p in purchases:
        add_purchase(*p)

    products = [
        (7, "Hat", 19.99),
        (8, "Shoes", 39.99),
    ]
    for p in products:
        add_product(*p)

def add_user(user_id, first_name, last_name, email, phone):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE user_id = ?", (user_id,))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO Users (user_id, first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?, ?)""", (user_id, first_name, last_name, email, phone))
        conn.commit()

def add_purchase(user_id, date_of_purchase, item_id, amount):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM PurchaseHistory WHERE user_id = ? AND item_id = ? AND date_of_purchase = ?",
                   (user_id, item_id, date_of_purchase))
    if not cursor.fetchone():
        cursor.execute("""
            INSERT INTO PurchaseHistory (user_id, date_of_purchase, item_id, amount)
            VALUES (?, ?, ?, ?)""", (user_id, date_of_purchase, item_id, amount))
        conn.commit()

def add_product(product_id, product_name, price):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO Products VALUES (?, ?, ?);", (product_id, product_name, price))
    conn.commit()

# --- TYPING DEL ESTADO ---
class AgentState(TypedDict):
    query: str
    user_id: str
    item_id: str
    product_id: str
    agent_outputs: Dict[str, str]

# --- TRIAGE CON LLM ---
def triage_fn(state: AgentState) -> dict:
    with get_openai_callback() as cb:
        prompt = f"""
                Determina si esta consulta corresponde a un reembolso, una compra o una solicitud de descuento:

                Consulta: "{state['query']}"

                Devuelve solo una de estas palabras: refunds_agent, sales_agent o discount_agent
                """
        response = llm.invoke([HumanMessage(content=prompt)])
        route = response.content.strip().replace('"', '').replace("'", "")  # <- ✨ Solución
        print(f"[triage_fn] Redirigiendo a: {route}")
        return {"__next__": route, "tokens_used": cb.total_tokens}

# --- AGENTES ---

def discount_agent(state: AgentState) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    print("[discount_agent] Buscando compra del usuario...")
    cursor.execute("SELECT amount FROM PurchaseHistory WHERE user_id = ? AND item_id = ?",
                   (state["user_id"], state["item_id"]))
    resultado = cursor.fetchone()
    print(f"[discount_agent] Resultado de la compra: {resultado}")

    with get_openai_callback() as cb:
        if resultado:
            monto_original = resultado[0]
            prompt = f"""Extrae el porcentaje de descuento solicitado en esta frase. 
            Devuelve solo un número decimal sin el símbolo %. 
            Frase: '{state["query"]}'"""
            print(f"[discount_agent] Enviando prompt: {prompt}")
            descuento_str = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            print(f"[discount_agent] Resultado del LLM: {descuento_str}")
            try:
                porcentaje = float(descuento_str)
            except ValueError:
                porcentaje = 0.10

            monto_descuento = monto_original * porcentaje
            monto_final = monto_original - monto_descuento
            respuesta = f"Se aplicó un descuento de {porcentaje*100:.0f}% sobre ${monto_original:.2f}. Total: ${monto_final:.2f}."
            razonamiento = f"El usuario solicitó un descuento de {porcentaje*100:.0f}% sobre el ítem {state['item_id']}."
        else:
            respuesta = "No se encontró una compra válida para aplicar el descuento."
            razonamiento = f"No se encontró el item {state['item_id']} en el historial de compras del usuario {state['user_id']}."

        prompt_resumen = f"""
        Actuá como un asistente de compras. Generá un resumen amable y profesional explicando la acción realizada.
        Contexto: El usuario pidió aplicar un descuento del {int(porcentaje * 100)}% sobre el ítem {state['item_id']} con precio original ${monto_original:.2f}.
        El descuento fue aplicado correctamente, y el precio final es de ${monto_final:.2f}.
        """

        resumen = llm.invoke(prompt_resumen)

        print("\n\n--- RESPUESTA FINAL ---")
        print("🧠 Resumen del LLM:", resumen.content)
        print("📦 Resultado:", respuesta)
        print("📊 Tokens usados:", cb.total_tokens)
        print("🤖 Agente ejecutado:", "discount_agent")
        return {
            "__next__": END,
            **state,
            "agent_outputs": {
                "respuesta": respuesta,
                "resumen": resumen.content,
                "tokens_used": cb.total_tokens,
                "razonamiento": razonamiento,
    }
}


def refunds_agent(state: AgentState) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT amount FROM PurchaseHistory WHERE user_id = ? AND item_id = ?",
                   (state["user_id"], state["item_id"]))
    resultado = cursor.fetchone()

    with get_openai_callback() as cb:
        razonamiento = f"El usuario {state['user_id']} solicita reembolso del item {state['item_id']}."
        if resultado:
            monto = resultado[0]
            razonamiento += f" El monto a reembolsar es ${monto}."
            respuesta = f"Se procesó un reembolso de ${monto} y se notificó al cliente."
        else:
            respuesta = "No se encontró una compra para procesar el reembolso."
        
        prompt_resumen = f"""
            Actuá como un asistente de compras. Generá un resumen amable y profesional explicando la acción realizada.
            Contexto: {razonamiento}
            Resultado: {respuesta}
            """
        
        resumen = llm.invoke(prompt_resumen)
    print("\n\n--- RESPUESTA FINAL ---")
    print("🧠 Resumen del LLM:", resumen.content)
    print("📦 Resultado:", respuesta)
    print("📊 Tokens usados:", cb.total_tokens)
    print("🤖 Agente ejecutado:", "refunds_agent")
    return {
        "__next__": END,
        **state,
        "agent_outputs": {
            "respuesta": respuesta,
            "resumen": resumen.content,
            "tokens_used": cb.total_tokens,
            "razonamiento": razonamiento,
        }
}


def sales_agent(state: AgentState) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT product_name, price FROM Products WHERE product_id = ?", (state["product_id"],))
    resultado = cursor.fetchone()

    with get_openai_callback() as cb:
        if resultado:
            nombre, precio = resultado
            respuesta = f"Se ordenó el producto '{nombre}' por ${precio} para el usuario {state['user_id']}."
            razonamiento = f"El usuario {state['user_id']} quiere comprar {nombre}. Se procesó la orden."
            add_purchase(state["user_id"], datetime.datetime.now(), random.randint(100, 999), precio)
        else:
            respuesta = "El producto solicitado no existe."
            razonamiento = "No se pudo encontrar el producto en la base."

        prompt_resumen = f"""
            Actuá como un asistente de compras. Generá un resumen amable y profesional explicando la acción realizada.
            Contexto: {razonamiento}
            Resultado: {respuesta}
            """
        
        resumen = llm.invoke(prompt_resumen)
    print("\n\n--- RESPUESTA FINAL ---")
    print("🧠 Resumen del LLM:", resumen.content)
    print("📦 Resultado:", respuesta)
    print("📊 Tokens usados:", cb.total_tokens)
    print("🤖 Agente ejecutado:", "sales_agent")
    return {
        "__next__": END,
        **state,
        "agent_outputs": {
            "respuesta": respuesta,
            "resumen": resumen.content,
            "tokens_used": cb.total_tokens,
            "razonamiento": razonamiento,
        }
}

# --- GRAPH ---
initialize_database()

graph = StateGraph(AgentState)
graph.add_node("triage", RunnableLambda(triage_fn))
graph.add_node("refunds_agent", RunnableLambda(refunds_agent))
graph.add_node("sales_agent", RunnableLambda(sales_agent))
graph.add_node("discount_agent", RunnableLambda(discount_agent))
graph.set_entry_point("triage")

graph.add_conditional_edges(
    "triage",
    lambda s: s["__next__"],
    {
        "refunds_agent": "refunds_agent",
        "sales_agent": "sales_agent",
        "discount_agent": "discount_agent",
    }
)
multiagent_cv_graph = graph.compile()

# --- EJECUCIÓN EJEMPLO ---
if __name__ == "__main__":
    example_state = {
        "query": "Quiero devolver el producto que compré la semana pasada",
        "user_id": "1",
        "item_id": "101",
        "product_id": "8",
        "agent_outputs": {},
    }
    result = multiagent_cv_graph.invoke(example_state)